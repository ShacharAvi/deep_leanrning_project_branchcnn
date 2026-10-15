"""Per-branch dataset construction (spec Phase 5 / plan Phase B)."""

from .builder import (
    BUILDER_VERSION,
    STORAGE_PC_BITS,
    BuildStats,
    as_signed_pc,
    build_trace_dataset,
    decode_token,
    encode_tokens,
    read_hard_branches,
    token_dtype,
)
from .aux_features import (
    DIRECTION_FREE_GROUPS,
    AuxFeatureError,
    AuxFeatureSpec,
)
from .raw_features import (
    RAW_FEATURE_GROUPS,
    CombinedFeatureSpec,
    RawFeatureSpec,
    load_raw_features,
)
from .pytorch_dataset import (
    BranchDataset,
    clear_history_cache,
    load_history,
    preprocess_history,
    remove_incomplete_histories,
    vocabulary_size,
)

__all__ = [
    "BUILDER_VERSION",
    "AuxFeatureError",
    "AuxFeatureSpec",
    "DIRECTION_FREE_GROUPS",
    "BranchDataset",
    "BuildStats",
    "as_signed_pc",
    "CombinedFeatureSpec",
    "RAW_FEATURE_GROUPS",
    "RawFeatureSpec",
    "load_raw_features",
    "clear_history_cache",
    "load_history",
    "STORAGE_PC_BITS",
    "build_trace_dataset",
    "decode_token",
    "encode_tokens",
    "preprocess_history",
    "read_hard_branches",
    "remove_incomplete_histories",
    "token_dtype",
    "vocabulary_size",
]
