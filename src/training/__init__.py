"""Training harness (spec Phase 6 / plan Phase C)."""

from .trainer import (
    EpochRecord,
    TrainConfig,
    evaluate,
    resolve_device,
    save_run,
    seed_everything,
    summarise_against_baselines,
    train_branch_model,
)

__all__ = [
    "EpochRecord",
    "TrainConfig",
    "evaluate",
    "resolve_device",
    "save_run",
    "seed_everything",
    "summarise_against_baselines",
    "train_branch_model",
]
