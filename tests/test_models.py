"""Model tests -- spec §27 "Model shape tests" plus the architecture invariants.

Spec §27 asks only that ``x -> model(x)`` yields exactly one binary logit per
sample. That is necessary but weak, so this also pins the properties that make
the architecture ladder of §12 meaningful: that sum-pooling really is
position-invariant, that BranchNet's slices read the *recent* end of the
history, and that the reported cost vector matches the actual parameters.
"""

import pytest
import torch

from src.models import CNN, MLP, BranchNet, build_model

VOCAB = 8192
H = 582


def tokens(batch=4, length=H, vocab=VOCAB, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g)


# (name, model kwargs, input length). BranchNet reads suffixes of a long
# history, so it always needs the full H regardless of its slice lengths.
ALL_MODELS = [
    ("mlp", dict(embedding_dim=8, hidden=(16,)), 64),
    ("cnn1", dict(embedding_dim=8, channels=4), 64),
    ("cnn2", dict(embedding_dim=8, channels=4), 64),
    ("branchnet", dict(embedding_dim=8, channels=(2,) * 5, hidden=(8,)), H),
    ("branchnet_mini", dict(embedding_dim=8), 602),
]


# --------------------------------------------------------------------------- #
# spec §27: one binary logit per sample
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,kwargs,length", ALL_MODELS)
def test_model_emits_one_logit_per_sample(name, kwargs, length):
    model = build_model(name, vocabulary_size=VOCAB,
                        history_length=length, **kwargs).eval()
    out = model(tokens(batch=5, length=length))
    assert out.shape == (5,)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("name,kwargs,length", ALL_MODELS)
def test_output_is_a_logit_not_a_probability(name, kwargs, length):
    """Spec §16: train on logits, sigmoid only at eval."""
    model = build_model(name, vocabulary_size=VOCAB,
                        history_length=length, **kwargs)
    # A sigmoid as the last layer would be impossible to push outside [0, 1].
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.mul_(50.0)
        out = model(tokens(batch=8, length=length))
    assert out.min() < 0.0 or out.max() > 1.0


@pytest.mark.parametrize("name,kwargs,length", ALL_MODELS)
def test_gradients_reach_the_embedding(name, kwargs, length):
    model = build_model(name, vocabulary_size=VOCAB,
                        history_length=length, **kwargs)
    x = tokens(batch=8, length=length)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(x), torch.zeros(8))
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name,kwargs,length", ALL_MODELS)
def test_batch_size_one_works(name, kwargs, length):
    """BatchNorm in train mode breaks on batch size 1; eval mode must not."""
    model = build_model(name, vocabulary_size=VOCAB,
                        history_length=length, **kwargs).eval()
    assert model(tokens(batch=1, length=length)).shape == (1,)


@pytest.mark.parametrize("name,kwargs,length", ALL_MODELS)
def test_samples_are_independent_within_a_batch(name, kwargs, length):
    """No cross-sample leakage: a row's output must not depend on its neighbours."""
    model = build_model(name, vocabulary_size=VOCAB,
                        history_length=length, **kwargs).eval()
    x = tokens(batch=4, length=length, seed=1)
    with torch.no_grad():
        together = model(x)
        alone = torch.cat([model(x[i:i + 1]) for i in range(4)])
    assert torch.allclose(together, alone, atol=1e-5)


# --------------------------------------------------------------------------- #
# the architecture ladder (spec §12)
# --------------------------------------------------------------------------- #

def test_sum_pooling_is_position_invariant():
    """Spec §10.3: sum-pooling answers "how often", not "where".

    Shifting a motif within the history must not change the pooled vector --
    that is exactly the property BranchNet relies on in a noisy history.
    """
    model = CNN(vocabulary_size=VOCAB, history_length=64, embedding_dim=8,
                channels=4, conv_width=3, pooling="sum",
                batch_norm=False).eval()

    a = torch.zeros(1, 64, dtype=torch.long)
    b = torch.zeros(1, 64, dtype=torch.long)
    motif = torch.tensor([11, 22, 33])
    a[0, 5:8] = motif
    b[0, 40:43] = motif

    with torch.no_grad():
        emb_a = model.activation(model.conv(model.embedding(a).transpose(1, 2)))
        emb_b = model.activation(model.conv(model.embedding(b).transpose(1, 2)))
        assert torch.allclose(model._pool(emb_a), model._pool(emb_b), atol=1e-5)


def test_no_pooling_is_position_sensitive():
    """The contrast that makes the previous test meaningful."""
    model = CNN(vocabulary_size=VOCAB, history_length=64, embedding_dim=8,
                channels=4, conv_width=3, pooling="none",
                batch_norm=False).eval()
    a = torch.zeros(1, 64, dtype=torch.long)
    b = torch.zeros(1, 64, dtype=torch.long)
    a[0, 5:8] = torch.tensor([11, 22, 33])
    b[0, 40:43] = torch.tensor([11, 22, 33])
    with torch.no_grad():
        assert not torch.allclose(model(a), model(b), atol=1e-4)


def test_segmented_pooling_keeps_coarse_position():
    """pooling_width > 1 trades some invariance for coarse position."""
    model = CNN(vocabulary_size=VOCAB, history_length=64, embedding_dim=8,
                channels=4, conv_width=3, pooling="sum", pooling_width=8,
                batch_norm=False).eval()
    assert model.segments > 1
    a = torch.zeros(1, 64, dtype=torch.long)
    b = torch.zeros(1, 64, dtype=torch.long)
    a[0, 2:5] = torch.tensor([11, 22, 33])
    b[0, 50:53] = torch.tensor([11, 22, 33])
    with torch.no_grad():
        assert not torch.allclose(model(a), model(b), atol=1e-4)


@pytest.mark.parametrize("pooling", ["sum", "avg", "max", "none"])
def test_all_pooling_modes_run(pooling):
    """Spec §19 ablates pooling; all four must be usable."""
    model = CNN(vocabulary_size=VOCAB, history_length=64, embedding_dim=8,
                channels=4, pooling=pooling).eval()
    assert model(tokens(batch=3, length=64)).shape == (3,)


def test_unknown_pooling_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        CNN(vocabulary_size=VOCAB, history_length=64, pooling="median")


# --------------------------------------------------------------------------- #
# BranchNet specifics
# --------------------------------------------------------------------------- #

def test_branchnet_defaults_match_the_paper():
    """Cross-checked against BranchNet-master/src/branchnet/configs/big.yaml."""
    assert BranchNet.BIG["history_lengths"] == (42, 78, 150, 294, 582)
    assert BranchNet.BIG["channels"] == (32, 32, 32, 32, 32)
    assert BranchNet.BIG["pooling_widths"] == (3, 6, 12, 24, 48)
    assert BranchNet.BIG["embedding_dim"] == 32
    assert BranchNet.BIG["hidden"] == (128, 128)
    assert all(w == 7 for w in BranchNet.BIG["conv_widths"])


def test_branchnet_slices_read_the_recent_end_of_history():
    """A short slice must see the branches nearest the target, not the oldest."""
    model = BranchNet(vocabulary_size=VOCAB, history_lengths=(8, 16),
                      channels=(2, 2), pooling_widths=(2, 4),
                      conv_widths=(3, 3), embedding_dim=4, hidden=(4,)).eval()
    short = model.slices[0]
    x = torch.arange(16, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        used = x[:, -short.history_length:]
    assert used.tolist() == [[8, 9, 10, 11, 12, 13, 14, 15]]


def test_branchnet_rejects_too_short_a_history():
    model = BranchNet(vocabulary_size=VOCAB, history_lengths=(8, 64),
                      channels=(2, 2), pooling_widths=(2, 4),
                      conv_widths=(3, 3), embedding_dim=4, hidden=(4,)).eval()
    with pytest.raises(ValueError, match="at least 64 tokens"):
        model(torch.zeros(2, 32, dtype=torch.long))


def test_branchnet_slice_lists_must_agree_in_length():
    with pytest.raises(ValueError, match="same length"):
        BranchNet(vocabulary_size=VOCAB, history_lengths=(8, 16),
                  channels=(2,), pooling_widths=(2, 4), conv_widths=(3, 3))


def test_branchnet_shares_one_embedding_across_slices():
    """Separate embeddings per slice would multiply the dominant cost."""
    model = BranchNet(vocabulary_size=VOCAB, history_lengths=(8, 16),
                      channels=(2, 2), pooling_widths=(2, 4),
                      conv_widths=(3, 3), embedding_dim=4, hidden=(4,))
    assert all(s.embedding is model.embedding for s in model.slices)


def test_mini_branchnet_is_far_smaller_than_big():
    """Only when built with its own folded vocabulary.

    Mini's saving comes mostly from pc_hash_bits=6 (vocab 128) rather than from
    fewer channels. Built at Big's vocabulary it is barely half the size, which
    is the trap dataset_config() exists to prevent.
    """
    from src.dataset import vocabulary_size as vocab_of
    from src.models import dataset_config

    big_cfg, mini_cfg = dataset_config("branchnet_big"), dataset_config("mini_2k")
    big = BranchNet(vocabulary_size=vocab_of(**big_cfg), **BranchNet.BIG)
    mini = BranchNet(vocabulary_size=vocab_of(**mini_cfg), **BranchNet.MINI_2K)
    assert mini.count_parameters()[0] < big.count_parameters()[0] / 20

    wrong = BranchNet(vocabulary_size=vocab_of(**big_cfg), **BranchNet.MINI_2K)
    assert wrong.count_parameters()[0] > 10 * mini.count_parameters()[0]


def test_dataset_config_matches_branchnet_yaml():
    """Cross-checked against BranchNet-master/src/branchnet/configs/*.yaml."""
    from src.models import dataset_config
    assert dataset_config("branchnet_big") == dict(
        pc_bits=12, pc_hash_bits=12, hash_dir_with_pc=False)
    assert dataset_config("mini_2k") == dict(
        pc_bits=12, pc_hash_bits=6, hash_dir_with_pc=False)


# --------------------------------------------------------------------------- #
# cost reporting (spec §15)
# --------------------------------------------------------------------------- #

def test_cost_matches_the_actual_parameter_count():
    model = MLP(vocabulary_size=VOCAB, history_length=64, embedding_dim=8,
                hidden=(16,))
    cost = model.cost()
    assert cost.parameters == sum(p.numel() for p in model.parameters())
    assert cost.bytes_fp32 == cost.parameters * 4
    assert cost.kib_at(8) == pytest.approx(cost.kib_fp32 / 4)
    assert cost.kib_at(4) == pytest.approx(cost.kib_fp32 / 8)
    assert cost.as_dict()["kib_int4"] == pytest.approx(cost.kib_at(4))


@pytest.mark.parametrize("name,kwargs,length", ALL_MODELS)
def test_mac_count_is_positive_and_reported(name, kwargs, length):
    model = build_model(name, vocabulary_size=VOCAB,
                        history_length=length, **kwargs)
    cost = model.cost(name=name)
    assert cost.macs_per_prediction > 0
    assert set(cost.as_dict()) >= {
        "name", "parameters", "kib_fp32", "kib_int4", "macs_per_prediction"}


def test_embedding_dominates_big_branchnet_storage():
    """8192 x 32 embedding is most of the model -- why Mini folds the PC."""
    model = BranchNet(vocabulary_size=VOCAB, **BranchNet.BIG)
    total = model.count_parameters()[0]
    assert model.embedding.weight.numel() / total > 0.4


def test_unknown_model_name_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("lstm", vocabulary_size=VOCAB, history_length=64)
