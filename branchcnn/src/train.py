"""Training and evaluation for the branch direction CNN.

Reports accuracy against three reference points, because an accuracy on its own
is not interpretable for this task:

* **majority class** -- the bias floor. A model at this number has learnt
  nothing.
* **always taken** -- the trivial hardware predictor.
* **2-bit saturating counter** -- a real, if weak, predictor, evaluated causally
  per static branch over the test split.

The counter is used only as a *baseline*; no predictor state is fed to the model
as a feature. That is deferred.

``--shuffle-labels`` is the leakage control. It permutes labels while leaving
inputs untouched, so a pipeline free of leakage cannot beat the bias floor. Any
result above it means information is reaching the model by some path other than
the intended inputs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .dataset import NUM_SCALARS, NUM_SEQ_NUMERIC, BranchDataset
from .model import BranchCNN, ModelConfig, count_parameters, tage_weights

__all__ = ["TrainConfig", "EpochStats", "train", "evaluate", "baselines"]


@dataclass
class TrainConfig:
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 10
    patience: int = 3
    #: Stop an epoch after this many batches. Useful for smoke runs.
    max_batches: Optional[int] = None
    shuffle_labels: bool = False
    device: str = "cpu"
    seed: int = 0
    threads: Optional[int] = None


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float


def _forward(model: BranchCNN, t: Dict[str, torch.Tensor]):
    return model(
        t["tokens"], t["classes"], t["seq_numeric"], t["scalars"], t["pc"],
        t.get("tage"), t.get("tage_state"), t.get("tage_sign"),
    )


def _to_torch(batch: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    extra = {k: torch.from_numpy(batch[k]).to(device) for k in ("tage", "tage_state", "tage_sign") if k in batch}
    return {
        **extra,
        "tokens": torch.from_numpy(batch["tokens"]).to(device),
        "classes": torch.from_numpy(batch["classes"]).to(device),
        "seq_numeric": torch.from_numpy(batch["seq_numeric"]).to(device),
        "scalars": torch.from_numpy(batch["scalars"]).to(device),
        "pc": torch.from_numpy(batch["pc"]).to(device),
        "label": torch.from_numpy(batch["label"]).to(device),
    }


def baselines(dataset: BranchDataset, role: str) -> Dict[str, float]:
    """Reference predictors over one split, all computed causally.

    The 2-bit counter is indexed by static PC and updated after each occurrence,
    in trace order, which is what a hardware counter does.
    """
    data = dataset.splits[role]
    total = 0
    taken = 0
    counter_correct = 0
    instructions = 0
    tage_correct = 0
    have_tage = all(e.tage is not None for e in data.traces)

    for entry in data.traces:
        features = entry.features
        rows = entry.rows
        outcomes = features.s_out[rows].astype(np.int64)
        pcs = features.s_pc[rows]
        total += outcomes.size
        taken += int(outcomes.sum())
        instructions += features.instructions_seen
        if have_tage:
            tage_correct += int((entry.tage.pred[rows].astype(np.int64) == outcomes).sum())

        state: Dict[int, int] = {}
        for pc, outcome in zip(pcs.tolist(), outcomes.tolist()):
            value = state.get(pc, 1)
            if (value >= 2) == bool(outcome):
                counter_correct += 1
            if outcome:
                state[pc] = min(3, value + 1)
            else:
                state[pc] = max(0, value - 1)

    if not total:
        return {}
    taken_rate = taken / total
    return {
        "samples": total,
        "instructions": instructions,
        "taken_rate": taken_rate,
        "majority_accuracy": max(taken_rate, 1.0 - taken_rate),
        "always_taken_accuracy": taken_rate,
        "two_bit_accuracy": counter_correct / total,
        **(
            {
                "tage_accuracy": tage_correct / total,
                "tage_mpki": 1000.0 * (total - tage_correct) / max(instructions, 1),
            }
            if have_tage
            else {}
        ),
    }


def evaluate(
    model: BranchCNN,
    dataset: BranchDataset,
    role: str,
    config: TrainConfig,
    loss_fn: nn.Module,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for index, batch in enumerate(
            dataset.iter_batches(
                role,
                batch_size=config.batch_size,
                shuffle=False,
                shuffle_labels=config.shuffle_labels,
                seed=config.seed,
            )
        ):
            if max_batches is not None and index >= max_batches:
                break
            tensors = _to_torch(batch, config.device)
            logit, _ = _forward(model, tensors)
            loss = loss_fn(logit, tensors["label"])
            size = tensors["label"].shape[0]
            loss_sum += loss.item() * size
            correct += int(((logit > 0).float() == tensors["label"]).sum())
            total += size
    if not total:
        return {"loss": float("nan"), "accuracy": float("nan"), "samples": 0}
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "samples": total,
    }


def train(
    dataset: BranchDataset,
    model_config: ModelConfig,
    config: TrainConfig = TrainConfig(),
    output_dir: Optional[Path] = None,
) -> dict:
    """Fit the model, early-stopping on validation accuracy, then test once."""
    torch.manual_seed(config.seed)
    if config.threads:
        torch.set_num_threads(config.threads)

    model = BranchCNN(model_config).to(config.device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_accuracy = -1.0
    best_state = None
    stale = 0

    print(
        f"model: {count_parameters(model):,} parameters, "
        f"head input {model.head_inputs}, conv output {model.conv_output}"
    )

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        started = time.time()
        total = 0
        correct = 0
        loss_sum = 0.0

        for index, batch in enumerate(
            dataset.iter_batches(
                "train",
                batch_size=config.batch_size,
                shuffle=True,
                shuffle_labels=config.shuffle_labels,
                seed=config.seed + epoch,
            )
        ):
            if config.max_batches is not None and index >= config.max_batches:
                break
            tensors = _to_torch(batch, config.device)
            optimiser.zero_grad(set_to_none=True)
            logit, _ = _forward(model, tensors)
            loss = loss_fn(logit, tensors["label"])
            loss.backward()
            optimiser.step()

            size = tensors["label"].shape[0]
            loss_sum += loss.item() * size
            correct += int(((logit > 0).float() == tensors["label"]).sum())
            total += size

        validation = evaluate(model, dataset, "validation", config, loss_fn)
        stats = EpochStats(
            epoch=epoch,
            train_loss=loss_sum / max(total, 1),
            train_accuracy=correct / max(total, 1),
            val_loss=validation["loss"],
            val_accuracy=validation["accuracy"],
            seconds=time.time() - started,
        )
        history.append(asdict(stats))
        print(
            f"epoch {epoch:2d}  train loss {stats.train_loss:.4f} "
            f"acc {stats.train_accuracy:.4f}  |  val loss {stats.val_loss:.4f} "
            f"acc {stats.val_accuracy:.4f}  ({stats.seconds:.0f}s, {total:,} samples)",
            flush=True,
        )

        if stats.val_accuracy > best_accuracy:
            best_accuracy = stats.val_accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                print(f"early stop: no validation gain for {stale} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, dataset, "test", config, loss_fn)
    result = {
        "dataset": dataset.describe(),
        "model_config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(model_config).items()
        },
        "train_config": asdict(config),
        "parameters": count_parameters(model),
        "history": history,
        "best_val_accuracy": best_accuracy,
        "test": test,
        "test_mpki_on_scored_branches": (
            1000.0 * (1.0 - test["accuracy"]) * test["samples"]
            / max(baselines(dataset, "test").get("instructions", 1), 1)
            if test["samples"] else None
        ),
        "tage_weights": tage_weights(model),
        "baselines": {
            "validation": baselines(dataset, "validation"),
            "test": baselines(dataset, "test"),
        },
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        torch.save(
            {"state_dict": model.state_dict(), "model_config": asdict(model_config)},
            output_dir / "model.pt",
        )
        print(f"wrote {output_dir / 'result.json'}")

    return result
