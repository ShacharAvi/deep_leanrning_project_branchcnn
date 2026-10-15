"""Neural branch predictors, built in the order spec §12 prescribes.

The ladder matters. Jumping straight to BranchNet would leave us unable to say
*which* of its ideas does the work, so each step adds exactly one thing:

    MLP      embedding -> flatten -> FC            no inductive bias at all
    CNN-1    embedding -> conv -> global pool      local motifs, position-agnostic
    CNN-2    embedding -> conv -> SUM pool         sum-pooling: "how often", not "where"
    CNN-3    several history lengths in parallel   multi-scale
    BranchNet  CNN-3 + BatchNorm + the paper's knobs

Sum-pooling (CNN-1 -> CNN-2) is the single most important idea in BranchNet
(spec §10.3): a convolution can detect a useful local pattern, but *where* it
occurred in a long noisy history usually does not matter. Summing turns
positional activations into an occurrence count, which compresses the
representation and makes the hardware tractable.

Every model takes the same input -- a tensor of ``(batch, history_length)``
token ids -- and returns one **logit** per sample. Logits, not probabilities:
spec §16 wants ``BCEWithLogitsLoss``, with sigmoid applied only at evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelCost:
    """The cost vector spec §15 requires for every model.

    Accuracy alone cannot justify a design in this project -- a model that wins
    by 0.1% at 50x the storage is not a win. These numbers get logged alongside
    every result.
    """

    name: str
    parameters: int
    trainable_parameters: int
    bytes_fp32: int
    history_length: int
    embedding_dim: int
    macs_per_prediction: int
    config: dict = field(default_factory=dict)

    @property
    def kib_fp32(self) -> float:
        return self.bytes_fp32 / 1024

    def kib_at(self, bits_per_weight: int) -> float:
        """Storage if every weight were quantised to ``bits_per_weight``."""
        return self.parameters * bits_per_weight / 8 / 1024

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "trainable_parameters": self.trainable_parameters,
            "bytes_fp32": self.bytes_fp32,
            "kib_fp32": self.kib_fp32,
            "kib_int8": self.kib_at(8),
            "kib_int4": self.kib_at(4),
            "history_length": self.history_length,
            "embedding_dim": self.embedding_dim,
            "macs_per_prediction": self.macs_per_prediction,
            **self.config,
        }


class BranchPredictorModel(nn.Module):
    """Base class: fixes the input/output contract and the cost reporting."""

    def __init__(self, *, vocabulary_size: int, history_length: int,
                 embedding_dim: int):
        super().__init__()
        self.vocabulary_size = int(vocabulary_size)
        self.history_length = int(history_length)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def count_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def macs_per_prediction(self) -> int:  # pragma: no cover - overridden
        return 0

    def cost(self, name: str | None = None, **config) -> ModelCost:
        total, trainable = self.count_parameters()
        return ModelCost(
            name=name or type(self).__name__,
            parameters=total,
            trainable_parameters=trainable,
            bytes_fp32=total * 4,
            history_length=self.history_length,
            embedding_dim=self.embedding_dim,
            macs_per_prediction=self.macs_per_prediction(),
            config=config,
        )


class MLP(BranchPredictorModel):
    """Embedding -> flatten -> FC stack. No inductive bias whatsoever.

    The floor for the architecture comparison: it can express anything the
    others can, but has to learn position-specific weights for every slot in
    the history, so it scales badly with H and generalises poorly. If a CNN
    does not beat this, the convolutional structure is not earning its place.
    """

    def __init__(self, *, vocabulary_size: int, history_length: int,
                 embedding_dim: int = 32, hidden: tuple[int, ...] = (128, 128),
                 dropout: float = 0.0):
        super().__init__(vocabulary_size=vocabulary_size,
                         history_length=history_length,
                         embedding_dim=embedding_dim)
        self.hidden = tuple(hidden)
        layers: list[nn.Module] = []
        width = history_length * embedding_dim
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).flatten(start_dim=1)
        return self.head(x).squeeze(-1)

    def macs_per_prediction(self) -> int:
        macs, width = 0, self.history_length * self.embedding_dim
        for size in self.hidden:
            macs += width * size
            width = size
        return macs + width


class CNN(BranchPredictorModel):
    """Embedding -> Conv1d -> pooling -> FC. Spec §12's CNN-1 and CNN-2.

    ``pooling`` is the whole point of having both:

    ``sum``
        BranchNet's choice. Collapses each channel to how *often* its pattern
        fired, discarding position. Cheap in hardware (a running sum) and
        robust to a pattern drifting around in a noisy history.
    ``avg`` / ``max``
        The usual alternatives, kept for the §19 pooling ablation.
    ``none``
        Flatten everything -- keeps position, and shows what sum-pooling costs
        and buys.

    ``pooling_width`` splits the sequence into segments pooled independently,
    so some coarse position survives. ``None`` pools the whole sequence.
    """

    def __init__(self, *, vocabulary_size: int, history_length: int,
                 embedding_dim: int = 32, channels: int = 32,
                 conv_width: int = 7, pooling: str = "sum",
                 pooling_width: int | None = None,
                 hidden: tuple[int, ...] = (128,), dropout: float = 0.0,
                 batch_norm: bool = True):
        super().__init__(vocabulary_size=vocabulary_size,
                         history_length=history_length,
                         embedding_dim=embedding_dim)
        if pooling not in ("sum", "avg", "max", "none"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.channels = channels
        self.conv_width = conv_width
        self.pooling = pooling
        self.pooling_width = pooling_width
        self.hidden = tuple(hidden)

        self.conv = nn.Conv1d(embedding_dim, channels, conv_width)
        self.norm = nn.BatchNorm1d(channels) if batch_norm else nn.Identity()
        self.activation = nn.ReLU()

        self.conv_out = history_length - conv_width + 1
        if pooling == "none":
            self.segments = self.conv_out
        elif pooling_width is None:
            self.segments = 1
        else:
            self.segments = max(1, self.conv_out // pooling_width)

        layers: list[nn.Module] = []
        width = channels * self.segments
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) -> (B, C * segments)."""
        if self.pooling == "none":
            return x.flatten(start_dim=1)
        if self.segments == 1:
            if self.pooling == "sum":
                return x.sum(dim=2)
            if self.pooling == "avg":
                return x.mean(dim=2)
            return x.max(dim=2).values
        usable = self.segments * (x.shape[2] // self.segments)
        x = x[:, :, :usable].reshape(x.shape[0], x.shape[1], self.segments, -1)
        if self.pooling == "sum":
            x = x.sum(dim=3)
        elif self.pooling == "avg":
            x = x.mean(dim=3)
        else:
            x = x.max(dim=3).values
        return x.flatten(start_dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)   # (B, E, H)
        x = self.activation(self.norm(self.conv(x)))
        return self.head(self._pool(x)).squeeze(-1)

    def macs_per_prediction(self) -> int:
        macs = self.conv_out * self.channels * self.embedding_dim * self.conv_width
        width = self.channels * self.segments
        for size in self.hidden:
            macs += width * size
            width = size
        return macs + width


class BranchNetSlice(nn.Module):
    """One history-length slice: embedding -> conv -> norm -> ReLU -> sum-pool.

    Slices read *suffixes* of the same history buffer, so a short slice sees
    only recent branches while a long one sees far back. That is how BranchNet
    gets a multi-scale view without running several histories (spec §10.2).
    """

    def __init__(self, *, embedding: nn.Embedding, history_length: int,
                 channels: int, conv_width: int, pooling_width: int):
        super().__init__()
        self.embedding = embedding          # shared across slices
        self.history_length = history_length
        self.channels = channels
        self.conv_width = conv_width
        self.pooling_width = pooling_width

        self.conv = nn.Conv1d(embedding.embedding_dim, channels, conv_width)
        self.norm = nn.BatchNorm1d(channels)
        self.conv_out = history_length - conv_width + 1
        self.segments = max(1, self.conv_out // pooling_width)
        self.output_size = channels * self.segments

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Most recent `history_length` tokens: the window ends at the target.
        x = tokens[:, -self.history_length:]
        x = self.embedding(x).transpose(1, 2)
        x = torch.relu(self.norm(self.conv(x)))
        usable = self.segments * (x.shape[2] // self.segments)
        x = x[:, :, :usable].reshape(x.shape[0], x.shape[1], self.segments, -1)
        return x.sum(dim=3).flatten(start_dim=1)    # sum-pooling

    def macs(self) -> int:
        return self.conv_out * self.channels * self.embedding.embedding_dim * \
            self.conv_width


class BranchNet(BranchPredictorModel):
    """Multi-slice CNN, after Zangeneh et al., MICRO 2020.

    Defaults are the paper's Big-BranchNet, cross-checked against
    BranchNet's ``src/branchnet/configs/big.yaml`` rather than the
    spec's transcription of it:

        H = [42, 78, 150, 294, 582]   C = 32   P = [3, 6, 12, 24, 48]
        conv width 7, embedding 32, hidden [128, 128], pc_bits 12

    Note the geometric history lengths: correlations may sit a few branches
    back or several hundred, and a single receptive field cannot cover both
    cheaply.
    """

    BIG = dict(
        history_lengths=(42, 78, 150, 294, 582),
        channels=(32, 32, 32, 32, 32),
        pooling_widths=(3, 6, 12, 24, 48),
        conv_widths=(7, 7, 7, 7, 7),
        embedding_dim=32,
        hidden=(128, 128),
    )

    #: Mini-BranchNet 2 KB (configs/mini_2k.yaml). Quantisation is applied in a
    #: later phase; this is the architecture only.
    MINI_2K = dict(
        history_lengths=(37, 77, 152, 302, 602),
        channels=(4, 5, 5, 4, 4),
        pooling_widths=(7, 15, 30, 60, 120),
        conv_widths=(3, 3, 3, 3, 3),
        embedding_dim=32,
        hidden=(20,),
    )

    def __init__(self, *, vocabulary_size: int,
                 history_lengths=BIG["history_lengths"],
                 channels=BIG["channels"],
                 pooling_widths=BIG["pooling_widths"],
                 conv_widths=BIG["conv_widths"],
                 embedding_dim: int = 32,
                 hidden=(128, 128),
                 dropout: float = 0.0):
        lengths = tuple(history_lengths)
        if not (len(lengths) == len(channels) == len(pooling_widths)
                == len(conv_widths)):
            raise ValueError("slice parameter lists must all be the same length")
        super().__init__(vocabulary_size=vocabulary_size,
                         history_length=max(lengths),
                         embedding_dim=embedding_dim)
        self.history_lengths = lengths
        self.slices = nn.ModuleList([
            BranchNetSlice(embedding=self.embedding, history_length=h,
                           channels=c, conv_width=w, pooling_width=p)
            for h, c, p, w in zip(lengths, channels, pooling_widths, conv_widths)
        ])
        self.hidden = tuple(hidden)

        layers: list[nn.Module] = []
        width = sum(s.output_size for s in self.slices)
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        """Concatenated per-slice pooled vectors -- the CNN's final layer.

        Split out from :meth:`forward` so a feature head can be attached to it
        (see :class:`BranchNetWithFeatures`) without duplicating the trunk.
        """
        return torch.cat([s(tokens) for s in self.slices], dim=1)

    @property
    def pooled_size(self) -> int:
        return sum(s.output_size for s in self.slices)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < self.history_length:
            raise ValueError(
                f"BranchNet needs at least {self.history_length} tokens "
                f"(longest slice), got {tokens.shape[1]}"
            )
        return self.head(self.pooled(tokens)).squeeze(-1)

    def macs_per_prediction(self) -> int:
        macs = sum(s.macs() for s in self.slices)
        width = sum(s.output_size for s in self.slices)
        for size in self.hidden:
            macs += width * size
            width = size
        return macs + width


def needs_features(name: str) -> bool:
    """Whether a model configuration consumes per-instance features."""
    return name.lower() in ("branchnet_features", "branchnet_big_features",
                            "branchnet+features")


def build_model(name: str, *, vocabulary_size: int, history_length: int,
                **overrides) -> BranchPredictorModel:
    """Construct a model by name. Keeps configs declarative."""
    name = name.lower()
    if name == "mlp":
        return MLP(vocabulary_size=vocabulary_size,
                   history_length=history_length, **overrides)
    if name in ("cnn", "cnn1", "cnn_global"):
        overrides.setdefault("pooling", "avg")
        return CNN(vocabulary_size=vocabulary_size,
                   history_length=history_length, **overrides)
    if name in ("cnn2", "cnn_sum"):
        overrides.setdefault("pooling", "sum")
        return CNN(vocabulary_size=vocabulary_size,
                   history_length=history_length, **overrides)
    if name in ("branchnet", "branchnet_big"):
        return BranchNet(vocabulary_size=vocabulary_size, **overrides)
    if name in ("branchnet_mini", "mini_2k"):
        config = dict(BranchNet.MINI_2K)
        config.update(overrides)
        return BranchNet(vocabulary_size=vocabulary_size, **config)
    if name in ("branchnet_features", "branchnet_big_features",
                "branchnet+features"):
        return BranchNetWithFeatures(vocabulary_size=vocabulary_size,
                                     **overrides)
    raise ValueError(f"unknown model {name!r}")


#: Dataset-side settings each configuration expects, taken from BranchNet's
#: own yaml files. These are NOT model arguments -- they decide how
#: ``BranchDataset`` encodes tokens, and therefore the vocabulary the embedding
#: must cover.
#:
#: This is where most of Mini-BranchNet's size reduction actually comes from.
#: Folding the PC from 12 bits to 6 shrinks the vocabulary 8192 -> 128 and the
#: embedding 262,144 -> 4,096 parameters. The smaller channel counts matter far
#: less: in Big-BranchNet the embedding is ~47% of all parameters, so halving
#: the convolutions barely moves the total. Build the dataset with the matching
#: ``pc_hash_bits`` or a "mini" model will not be mini.
DATASET_CONFIGS = {
    "branchnet_big": dict(pc_bits=12, pc_hash_bits=12, hash_dir_with_pc=False),
    "branchnet_mini_2k": dict(pc_bits=12, pc_hash_bits=6, hash_dir_with_pc=False),
    "default": dict(pc_bits=12, pc_hash_bits=12, hash_dir_with_pc=False),
}


def dataset_config(name: str) -> dict:
    """Token-encoding settings a model configuration expects."""
    name = name.lower()
    if name in ("branchnet_mini", "mini_2k", "branchnet_mini_2k"):
        return dict(DATASET_CONFIGS["branchnet_mini_2k"])
    if name in ("branchnet", "branchnet_big", "branchnet_features",
                "branchnet_big_features", "branchnet+features"):
        return dict(DATASET_CONFIGS["branchnet_big"])
    return dict(DATASET_CONFIGS["default"])


class BranchNetWithFeatures(BranchPredictorModel):
    """BranchNet whose pooled CNN output is concatenated with per-instance
    features before the fully-connected head.

    This is the "join the features to the CNN's final layer and run an FC net"
    design. The trunk is an unmodified :class:`BranchNet`, so a feature-free
    baseline and this model differ in exactly one thing -- which is what makes
    the comparison an ablation rather than a confound.

    ::

        tokens -> [BranchNet slices] -> pooled (480)  ->|
                                                        |-> concat -> FC -> logit
        per-instance features (k) -> [BatchNorm] ------>|

    Two things this gets right that a naive concat does not:

    * **Features are normalised separately.** The pooled CNN activations and
      the raw features have different scales; without this the head spends its
      early epochs just learning the scale difference.
    * **Zero-variance features are dropped at construction.** We train one
      model per static branch, so a feature that is constant for that branch
      (``loop_valid`` is constant for ``0x404d9a``, for instance) contributes
      no gradient and only adds parameters. Pass ``feature_mask`` to drop them;
      :func:`src.dataset.aux_feature_mask` computes it from the training set.
    """

    def __init__(self, *, vocabulary_size: int, num_features: int,
                 feature_mask=None,
                 feature_hidden: int = 0,
                 hidden=(128, 128), dropout: float = 0.0, **branchnet_kwargs):
        if num_features <= 0:
            raise ValueError("num_features must be positive; use BranchNet "
                             "directly for the feature-free model")
        trunk = BranchNet(vocabulary_size=vocabulary_size,
                          hidden=(), dropout=0.0, **branchnet_kwargs)
        super().__init__(vocabulary_size=vocabulary_size,
                         history_length=trunk.history_length,
                         embedding_dim=trunk.embedding_dim)
        # Replace the trunk's own head; we build our own over the concatenation.
        trunk.head = nn.Identity()
        self.trunk = trunk
        self.embedding = trunk.embedding

        mask = (torch.ones(num_features, dtype=torch.bool)
                if feature_mask is None
                else torch.as_tensor(feature_mask, dtype=torch.bool))
        if mask.numel() != num_features:
            raise ValueError(
                f"feature_mask has {mask.numel()} entries, expected "
                f"{num_features}")
        if not bool(mask.any()):
            raise ValueError(
                "feature_mask drops every feature -- all of them are constant "
                "for this branch, so there is nothing for the feature head to "
                "learn. Train plain BranchNet instead.")
        self.register_buffer("feature_mask", mask)
        self.num_features_in = int(num_features)
        self.num_features_used = int(mask.sum())

        self.feature_norm = nn.BatchNorm1d(self.num_features_used)
        feature_width = self.num_features_used
        if feature_hidden:
            self.feature_mlp = nn.Sequential(
                nn.Linear(self.num_features_used, feature_hidden),
                nn.BatchNorm1d(feature_hidden), nn.ReLU())
            feature_width = feature_hidden
        else:
            self.feature_mlp = nn.Identity()

        layers: list[nn.Module] = []
        width = trunk.pooled_size + feature_width
        self.concat_size = width
        for size in tuple(hidden):
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor,
                features: torch.Tensor | None = None) -> torch.Tensor:
        if features is None:
            raise ValueError(
                f"{type(self).__name__} requires per-instance features; got "
                "None. The dataset must be built with aux_features=... .")
        if features.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"batch mismatch: {tokens.shape[0]} token rows vs "
                f"{features.shape[0]} feature rows")
        if features.shape[1] != self.num_features_in:
            raise ValueError(
                f"expected {self.num_features_in} features, got "
                f"{features.shape[1]}")
        pooled = self.trunk.pooled(tokens)
        kept = features[:, self.feature_mask].to(pooled.dtype)
        joined = torch.cat([pooled, self.feature_mlp(self.feature_norm(kept))],
                           dim=1)
        return self.head(joined).squeeze(-1)
