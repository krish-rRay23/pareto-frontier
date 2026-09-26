"""Entity-level context, target competition, and sibling support features."""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def compute_context_and_competition_features(
    pairs: List[Tuple[str, str]],
    s1_precomputed: Dict[str, Dict[str, Any]],
    target_precomputed: Dict[str, Dict[str, Any]],
    initial_scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute entity-level context, candidate density, and target competition features."""
    s1_counts: Dict[str, int] = Counter()
    target_counts: Dict[str, int] = Counter()

    for sid, tid in pairs:
        s1_counts[sid] += 1
        target_counts[tid] += 1

    # Track target scores for margin computation if available
    target_scores: Dict[str, List[float]] = defaultdict(list)
    if initial_scores is not None and len(initial_scores) == len(pairs):
        for (sid, tid), sc in zip(pairs, initial_scores):
            target_scores[tid].append(float(sc))
        for tid in target_scores:
            target_scores[tid].sort(reverse=True)

    rows: List[Dict[str, float]] = []

    for idx, (sid, tid) in enumerate(pairs):
        s1_cands = s1_counts.get(sid, 1)
        tgt_comp = target_counts.get(tid, 1)

        # Margin: difference between top-1 candidate probability for this target and second top
        target_margin = 0.0
        if tid in target_scores:
            sc_list = target_scores[tid]
            if len(sc_list) >= 2:
                target_margin = float(sc_list[0] - sc_list[1])
            else:
                target_margin = float(sc_list[0])

        s1_toks = s1_precomputed[sid]["name"]["tokens"]
        tgt_toks = target_precomputed[tid]["name"]["tokens"]

        s1_first_num = s1_precomputed[sid]["addr"]["first_num"]
        tgt_first_num = target_precomputed[tid]["addr"]["first_num"]
        hnum_match = 1.0 if (s1_first_num and s1_first_num == tgt_first_num) else 0.0

        rows.append(
            {
                "s1_candidate_density": float(s1_cands),
                "s1_has_single_candidate": 1.0 if s1_cands == 1 else 0.0,
                "target_competition_count": float(tgt_comp),
                "target_has_single_claimant": 1.0 if tgt_comp == 1 else 0.0,
                "target_score_margin": target_margin,
                "sibling_hnum_support": hnum_match,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "s1_candidate_density",
                "s1_has_single_candidate",
                "target_competition_count",
                "target_has_single_claimant",
                "target_score_margin",
                "sibling_hnum_support",
            ]
        )

    return pd.DataFrame(rows, dtype=np.float32)
