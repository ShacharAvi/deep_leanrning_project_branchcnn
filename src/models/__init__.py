"""Neural branch predictors (spec §12-§14)."""

from .branch_models import (
    CNN,
    MLP,
    BranchNet,
    BranchNetSlice,
    BranchNetWithFeatures,
    BranchPredictorModel,
    DATASET_CONFIGS,
    ModelCost,
    build_model,
    dataset_config,
    needs_features,
)

__all__ = [
    "CNN",
    "MLP",
    "BranchNet",
    "BranchNetSlice",
    "BranchNetWithFeatures",
    "BranchPredictorModel",
    "DATASET_CONFIGS",
    "ModelCost",
    "build_model",
    "dataset_config",
    "needs_features",
]
