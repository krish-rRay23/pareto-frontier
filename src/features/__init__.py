"""Feature engineering module exports."""

from src.features.pairwise_features import (
    FEATURE_COLUMNS,
    extract_pair_features,
    build_pairwise_feature_dataframe,
    jaro_winkler_similarity,
    sequence_match_ratio,
    jaccard_similarity,
    containment_ratio,
)
from src.features.consensus import compute_s2_s3_consensus
from src.features.context import compute_context_and_competition_features

__all__ = [
    "FEATURE_COLUMNS",
    "extract_pair_features",
    "build_pairwise_feature_dataframe",
    "jaro_winkler_similarity",
    "sequence_match_ratio",
    "jaccard_similarity",
    "containment_ratio",
    "compute_s2_s3_consensus",
    "compute_context_and_competition_features",
]
