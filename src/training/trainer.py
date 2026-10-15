"""Training harness for per-branch neural predictors.

Three things this does that a plain training loop does not, each because this
project's failure mode is a good-looking number rather than a crash:

1. **Reports against the bar that matters.** Accuracy on an H2P branch is
   meaningless on its own -- TAGE-SC-L is already right ~84-99% of the time on
   these. Every result is reported next to the bias floor (always-taken) and
   TAGE's own accuracy, because a model only helps if it beats *TAGE*
   (spec §29, §33 Rule 8).
2. **Selects on validation, reports on test, once.** The test set is touched
   exactly once, after early stopping has chosen a checkpoint.
3. **Logs the full reproducibility record** (spec §26): seed, config, split,
   dataset sizes, model cost vector, and per-epoch history.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..dataset import BranchDataset
from ..models import BranchPredictorModel


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a device. MPS on Apple Silicon, else CUDA, else CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Spec §26: all relevant seeds, recorded with the run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run."""

    branch_pc: int
    model: str = "cnn2"
    model_kwargs: dict = field(default_factory=dict)

    history_length: int = 582
    pc_bits: int = 12
    pc_hash_bits: int | None = None
    hash_dir_with_pc: bool = False

    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 20
    patience: int = 3
    use_pos_weight: bool = False      # spec §16: compare unweighted first

    max_train_samples: int | None = 2_000_000
    max_val_samples: int | None = 500_000   # selection only; test is never capped
    seed: int = 0
    device: str = "auto"

    #: What early stopping selects on. ``val_loss`` is the default;
    #: ``val_accuracy`` is kept only to reproduce older runs.
    #:
    #: Accuracy is a hard ``logit > 0`` threshold, so on a heavily-biased
    #: validation branch it carries almost no signal. mcf's validation trace
    #: 994B is 99.13% taken on 0x406337, which put *every* validation accuracy
    #: we recorded at or below that trace's own bias floor. Combined with seven
    #: BatchNorm layers whose eval-mode running statistics come from the
    #: training distribution, a small shift in normalised pre-activations flips
    #: a large fraction of predictions -- training loss untouched, validation
    #: accuracy destroyed. That is what made a harmless batch-size change look
    #: like a collapse.
    #:
    #: Loss is continuous, so it degrades gracefully instead of cliff-edging.
    selection_metric: str = "val_loss"

    def as_dict(self) -> dict:
        out = asdict(self)
        out["branch_pc"] = hex(self.branch_pc)
        return out


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    seconds: float
    selected: bool = False


def _unpack(batch, device, *, has_weights: bool = False):
    """Split a dataset batch into ``(model_args, labels)`` (+ weights).

    ``BranchDataset.iter_batches`` yields ``(tokens, labels)`` normally and
    ``(tokens, features, labels)`` when built with ``aux_features=``. Labels
    are always last, so this stays correct if a third input is ever added --
    except with ``with_weights=True``, where per-sample loss weights follow the
    labels and ``has_weights`` must be passed.
    """
    weights = None
    if has_weights:
        *batch, weights = batch
        weights = weights.to(device, non_blocking=True)
    *inputs, labels = batch
    out = (tuple(x.to(device, non_blocking=True) for x in inputs),
           labels.to(device, non_blocking=True))
    return out + (weights,) if has_weights else out


def evaluate(model: BranchPredictorModel, dataset: BranchDataset,
             device: torch.device, batch_size: int,
             criterion: nn.Module | None = None) -> dict:
    """Accuracy, loss, and the confusion counts, over an entire dataset.

    Returns per-instance correctness too, so Phase D can compute hybrid MPKI
    without re-running the model.
    """
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    tp = tn = fp = fn = 0
    predictions: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataset.iter_batches(batch_size):
            inputs, labels = _unpack(batch, device)
            logits = model(*inputs)
            if criterion is not None:
                loss_sum += float(criterion(logits, labels).detach()) * labels.numel()
            predicted = (logits > 0).float()      # logit > 0 <=> sigmoid > 0.5
            correct += int((predicted == labels).sum())
            total += labels.numel()
            tp += int(((predicted == 1) & (labels == 1)).sum())
            tn += int(((predicted == 0) & (labels == 0)).sum())
            fp += int(((predicted == 1) & (labels == 0)).sum())
            fn += int(((predicted == 0) & (labels == 1)).sum())
            predictions.append((predicted == labels).cpu().numpy().astype(np.uint8))
            logit_chunks.append(logits.detach().float().cpu().numpy())

    return {
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "loss": loss_sum / total if total else 0.0,
        "mispredictions": total - correct,
        "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn,
        "_correct": np.concatenate(predictions) if predictions else np.empty(0, np.uint8),
        # Raw logits in dataset order, for per-execution arbitration against
        # TAGE after training (confidence = |sigmoid(logit) - 0.5|).
        "_logits": (np.concatenate(logit_chunks) if logit_chunks
                    else np.empty(0, np.float32)),
    }


def train_branch_model(
    model: BranchPredictorModel,
    train_set: BranchDataset,
    val_set: BranchDataset,
    config: TrainConfig,
    *,
    log=print,
) -> dict:
    """Train with early stopping on validation accuracy.

    The test set is deliberately not a parameter: it must not be visible until
    after a checkpoint has been chosen.
    """
    device = resolve_device(config.device)
    model = model.to(device)
    log(f"  device {device}, {sum(p.numel() for p in model.parameters()):,} parameters")

    pos_weight = None
    if config.use_pos_weight:
        pos_weight = torch.tensor(train_set.pos_weight, device=device)
        log(f"  pos_weight {train_set.pos_weight:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Per-sample weights (e.g. emphasis on executions where TAGE is unsure)
    # apply to the TRAINING loss only; validation loss stays unweighted so
    # checkpoint selection measures the plain task.
    weighted = getattr(train_set, "has_sample_weights", False)
    weighted_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                              reduction="none")
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    if config.selection_metric not in ("val_loss", "val_accuracy"):
        raise ValueError(f"unknown selection_metric {config.selection_metric!r}")
    lower_is_better = config.selection_metric == "val_loss"

    # The validation set's own bias floor. If a run never beats it, validation
    # accuracy carries no signal and any accuracy-based selection is noise.
    val_bias_floor = max(val_set.taken_rate, 1 - val_set.taken_rate)
    log(f"  selecting on {config.selection_metric}; validation bias floor "
        f"{val_bias_floor:.4%}")

    history: list[EpochRecord] = []
    best_score = float("inf") if lower_is_better else -float("inf")
    best_epoch, best_state = -1, None
    stale = 0

    for epoch in range(1, config.max_epochs + 1):
        started = time.time()
        model.train()
        loss_sum = seen = 0.0

        for batch in train_set.iter_batches(
                config.batch_size, shuffle=True, seed=config.seed + epoch,
                drop_last=True, with_weights=weighted):
            if weighted:
                inputs, labels, weights = _unpack(batch, device,
                                                  has_weights=True)
            else:
                inputs, labels = _unpack(batch, device)
            optimiser.zero_grad(set_to_none=True)
            if weighted:
                per_sample = weighted_criterion(model(*inputs), labels)
                loss = (per_sample * weights).sum() / weights.sum()
            else:
                loss = criterion(model(*inputs), labels)
            loss.backward()
            optimiser.step()
            loss_sum += float(loss.detach()) * labels.numel()
            seen += labels.numel()

        val = evaluate(model, val_set, device, config.batch_size, criterion)
        record = EpochRecord(epoch, loss_sum / max(seen, 1), val["loss"],
                             val["accuracy"], time.time() - started)
        history.append(record)

        score = val["loss"] if lower_is_better else val["accuracy"]
        improved = score < best_score if lower_is_better else score > best_score
        log(f"  epoch {epoch:2d}  train_loss {record.train_loss:.5f}  "
            f"val_loss {record.val_loss:.5f}  val_acc {record.val_accuracy:.4%}  "
            f"{record.seconds:.0f}s{'  *' if improved else ''}")

        if improved:
            best_score, best_epoch = score, epoch
            record.selected = True
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                log(f"  early stop: no improvement for {config.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    best = next(r for r in history if r.epoch == best_epoch)
    log(f"  best epoch {best_epoch} by {config.selection_metric}: "
        f"val_loss {best.val_loss:.5f}, val_acc {best.val_accuracy:.4%}")

    # Loudly flag a validation set that cannot distinguish models. This is the
    # condition that made every earlier mcf hyperparameter conclusion unsafe.
    uninformative = all(r.val_accuracy <= val_bias_floor for r in history)
    if uninformative:
        log(f"  WARNING: no epoch beat the validation bias floor "
            f"({val_bias_floor:.4%}). Validation accuracy carries no signal on "
            f"this split; treat checkpoint selection as weakly determined.")

    return {
        "best_epoch": best_epoch,
        "selection_metric": config.selection_metric,
        "best_val_loss": best.val_loss,
        "best_val_accuracy": best.val_accuracy,
        "val_bias_floor": val_bias_floor,
        "validation_uninformative": uninformative,
        "history": [asdict(r) for r in history],
        "device": str(device),
    }


def summarise_against_baselines(test: dict, *, taken_rate: float,
                                tage_accuracy: float) -> dict:
    """Put model accuracy next to the two numbers that decide if it is useful.

    ``bias`` is the zero-knowledge floor (always-taken / always-not-taken);
    ``tage`` is what the model actually has to beat for the hybrid to improve.
    Spec §29's first failure mode is a model that clears the first and not the
    second -- high accuracy, no hybrid gain.
    """
    bias = max(taken_rate, 1 - taken_rate)
    accuracy = test["accuracy"]
    return {
        "model_accuracy": accuracy,
        "bias_accuracy": bias,
        "tage_accuracy": tage_accuracy,
        "gain_over_bias": accuracy - bias,
        "gain_over_tage": accuracy - tage_accuracy,
        "beats_bias": accuracy > bias,
        "beats_tage": accuracy > tage_accuracy,
        "mispredictions": test["mispredictions"],
        # tage_accuracy is NaN when the baseline statistics could not be found.
        # Report that rather than crashing -- the model result is still valid.
        "tage_mispredictions": (None if tage_accuracy != tage_accuracy
                                else round((1 - tage_accuracy) * test["samples"])),
        "misprediction_reduction_vs_tage": (
            ((1 - tage_accuracy) - (1 - accuracy)) / (1 - tage_accuracy)
            if tage_accuracy == tage_accuracy and tage_accuracy < 1 else float("nan")),
    }


def save_run(outdir: Path, *, config: TrainConfig, model: BranchPredictorModel,
             training: dict, test: dict, comparison: dict,
             datasets: dict, extra: dict | None = None) -> None:
    """Write the spec §26 artefact set: config, metrics, checkpoint."""
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": config.as_dict()}, outdir / "checkpoint.pt")
    (outdir / "config.json").write_text(json.dumps(config.as_dict(), indent=2))

    metrics = {
        "config": config.as_dict(),
        "model_cost": model.cost(name=config.model).as_dict(),
        "datasets": datasets,
        "training": training,
        "test": {k: v for k, v in test.items() if not k.startswith("_")},
        "comparison": comparison,
        **(extra or {}),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if "_correct" in test and test["_correct"].size:
        np.save(outdir / "test_correct.npy", test["_correct"])
