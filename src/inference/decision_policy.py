"""Precision-biased decision policy and threshold optimizer for Macro F0.5."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from src.validation.metrics import compute_macro_f05


class PrecisionDecisionPolicy:
    """Configurable multi-tier decision policy optimized for Macro F0.5.
    
    Tiers:
    1. Singleton Guard: If max(P) < singleton_threshold -> predict empty set [].
    2. Primary Match Threshold: Candidate c accepted only if P(c) >= match_threshold.
    3. Score Margin Gate: For secondary candidates, P(c) >= P_top - score_margin.
    """

    def __init__(
        self,
        match_threshold: float = 0.65,
        singleton_threshold: float = 0.55,
        score_margin: float = 0.20,
        max_matches_per_entity: Optional[int] = 10,
    ):
        self.match_threshold = match_threshold
        self.singleton_threshold = singleton_threshold
        self.score_margin = score_margin
        self.max_matches_per_entity = max_matches_per_entity

    def apply(
        self,
        pairs: List[Tuple[str, str]],
        scores: np.ndarray,
        all_s1_ids: Optional[List[str]] = None,
    ) -> Dict[str, Set[str]]:
        """Apply decision policy to candidate pair scores and generate final entity matches."""
        # Group scored candidates by S1 ID
        entity_cands: Dict[str, List[Tuple[str, float]]] = {}
        if all_s1_ids:
            for sid in all_s1_ids:
                entity_cands[sid] = []

        for (sid, tid), score in zip(pairs, scores):
            if sid not in entity_cands:
                entity_cands[sid] = []
            entity_cands[sid].append((tid, float(score)))

        predictions: Dict[str, Set[str]] = {}

        for sid, cand_list in entity_cands.items():
            if not cand_list:
                predictions[sid] = set()
                continue

            # Sort descending by predicted probability
            cand_list.sort(key=lambda x: x[1], reverse=True)
            top_cand, top_score = cand_list[0]

            # 1. Singleton Guard
            if top_score < self.singleton_threshold or top_score < self.match_threshold:
                predictions[sid] = set()
                continue

            # 2. Select candidates satisfying both match threshold and relative score margin
            matched: Set[str] = {top_cand}
            cutoff_score = max(self.match_threshold, top_score - self.score_margin)

            for tid, sc in cand_list[1:]:
                if sc >= cutoff_score:
                    matched.add(tid)
                    if self.max_matches_per_entity and len(matched) >= self.max_matches_per_entity:
                        break

            predictions[sid] = matched

        return predictions


def optimize_decision_policy(
    pairs: List[Tuple[str, str]],
    val_scores: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str],
    match_thresholds: Optional[List[float]] = None,
    singleton_thresholds: Optional[List[float]] = None,
    score_margins: Optional[List[float]] = None,
    verbose: bool = False,
) -> Tuple[PrecisionDecisionPolicy, Dict[str, float]]:
    """Grid-search decision policy parameters to maximize Macro F0.5 on OOF validation.

    Ensures zero leakage and optimal precision-recall trade-off directly on the competition metric.
    """
    if match_thresholds is None:
        match_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if singleton_thresholds is None:
        singleton_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    if score_margins is None:
        score_margins = [0.10, 0.15, 0.20, 0.25, 0.35]

    best_f05 = -1.0
    best_params = {}
    best_metrics = {}

    for mt in match_thresholds:
        for st in singleton_thresholds:
            if st > mt:
                continue  # Singleton threshold should not exceed match threshold
            for sm in score_margins:
                policy = PrecisionDecisionPolicy(
                    match_threshold=mt,
                    singleton_threshold=st,
                    score_margin=sm,
                )
                preds = policy.apply(pairs, val_scores, all_s1_ids=all_s1_ids)
                eval_res = compute_macro_f05(ground_truth, preds)

                if eval_res["macro_f05"] > best_f05:
                    best_f05 = eval_res["macro_f05"]
                    best_params = {"match_threshold": mt, "singleton_threshold": st, "score_margin": sm}
                    best_metrics = eval_res
                    if verbose:
                        print(f"New Best OOF F0.5: {best_f05:.4f} with {best_params}")

    optimal_policy = PrecisionDecisionPolicy(
        match_threshold=best_params.get("match_threshold", 0.65),
        singleton_threshold=best_params.get("singleton_threshold", 0.55),
        score_margin=best_params.get("score_margin", 0.20),
    )
    return optimal_policy, best_metrics
