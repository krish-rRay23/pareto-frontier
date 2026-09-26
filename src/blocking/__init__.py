"""Candidate blocking and retrieval module."""

from src.blocking.blocker import (
    MultiPassBlocker,
    evaluate_blocking,
    export_candidate_pairs_tsv,
)
from src.blocking.retrieval import (
    MultiChannelBidirectionalRetriever,
    RetrievalProvenance,
    ALL_CHANNELS,
)

__all__ = [
    "MultiPassBlocker",
    "evaluate_blocking",
    "export_candidate_pairs_tsv",
    "MultiChannelBidirectionalRetriever",
    "RetrievalProvenance",
    "ALL_CHANNELS",
]
