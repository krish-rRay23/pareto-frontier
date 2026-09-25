"""Features module exports."""

from src.features.consensus import compute_s2_s3_consensus
from src.features.pairwise_features import (
    FEATURE_COLUMNS,
    build_pairwise_feature_matrix,
    extract_pair_features,
    precompute_record_views,
)

__all__ = [
    "FEATURE_COLUMNS",
    "extract_pair_features",
    "precompute_record_views",
    "build_pairwise_feature_matrix",
    "compute_s2_s3_consensus",
]
