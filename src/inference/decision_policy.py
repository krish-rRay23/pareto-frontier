"""Precision-biased decision policies, Expected-F0.5 subset decoder, and target exclusivity resolver.

Implements components from Amazon ML Challenge 2026 99.99+ Plan:
1. PrecisionDecisionPolicy (Proven 3-tier baseline: threshold, margin, singleton guard)
2. ExpectedF05DecisionDecoder (Evaluates expected F0.5 across candidate subsets: empty, top-1, top-2, ...)
3. TargetExclusivityResolver (Enforces 1 target -> at most 1 S1 structural constraint)
4. ContradictionChecker (Conservative guards against country, address, and numeric contradictions)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
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

            cand_list.sort(key=lambda x: x[1], reverse=True)
            top_cand, top_score = cand_list[0]

            # 1. Singleton Guard
            if top_score < self.singleton_threshold or top_score < self.match_threshold:
                predictions[sid] = set()
                continue

            # 2. Select candidates satisfying match threshold and relative score margin
            matched: Set[str] = {top_cand}
            cutoff_score = max(self.match_threshold, top_score - self.score_margin)

            for tid, sc in cand_list[1:]:
                if sc >= cutoff_score:
                    matched.add(tid)
                    if self.max_matches_per_entity and len(matched) >= self.max_matches_per_entity:
                        break

            predictions[sid] = matched

        return predictions


class ExpectedF05DecisionDecoder:
    """Evaluates expected Macro F0.5 across candidate subsets for each S1 entity.

    Given candidate probabilities p_1 >= p_2 >= ... >= p_k:
    For any subset S = {1, ..., m} of size m:
    Expected True Positives: E[TP] = sum_{i=1}^m p_i
    Expected Precision: E[P] = E[TP] / m
    Estimated Recall: E[R] = E[TP] / max(1.0, sum_{all} p_i)
    Expected F0.5:
        1.25 * E[P] * E[R] / (0.25 * E[P] + E[R])
    """

    def __init__(
        self,
        singleton_prior_threshold: float = 0.45,
        min_candidate_prob: float = 0.25,
        max_subset_size: int = 8,
    ):
        self.singleton_prior_threshold = singleton_prior_threshold
        self.min_candidate_prob = min_candidate_prob
        self.max_subset_size = max_subset_size

    def _expected_f05(self, probs: List[float], subset_size: int, total_weight: float) -> float:
        if subset_size == 0:
            # If true total weight is very small, empty set is correct (F0.5 = 1.0)
            return 1.0 if total_weight < self.singleton_prior_threshold else 0.0

        sub_sum = sum(probs[:subset_size])
        exp_prec = sub_sum / float(subset_size)
        exp_rec = sub_sum / max(1.0, total_weight)

        denom = 0.25 * exp_prec + exp_rec
        if denom <= 0.0:
            return 0.0
        return float(1.25 * exp_prec * exp_rec / denom)

    def apply(
        self,
        pairs: List[Tuple[str, str]],
        scores: np.ndarray,
        all_s1_ids: Optional[List[str]] = None,
    ) -> Dict[str, Set[str]]:
        """Decode entity matches by finding the subset that maximizes expected F0.5."""
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

            cand_list.sort(key=lambda x: x[1], reverse=True)
            probs = [sc for _, sc in cand_list]
            total_prob = sum(probs)

            # Singleton guard: if total evidence is weak, abstain to empty
            if not probs or probs[0] < self.singleton_prior_threshold:
                predictions[sid] = set()
                continue

            # Evaluate empty set vs top-1, top-2, ..., top-M
            best_size = 0
            best_exp_f = self._expected_f05(probs, 0, total_prob)

            max_m = min(len(probs), self.max_subset_size)
            for m in range(1, max_m + 1):
                if probs[m - 1] < self.min_candidate_prob:
                    break
                exp_f = self._expected_f05(probs, m, total_prob)
                if exp_f > best_exp_f:
                    best_exp_f = exp_f
                    best_size = m

            if best_size == 0:
                predictions[sid] = set()
            else:
                predictions[sid] = {cand_list[i][0] for i in range(best_size)}

        return predictions


class TargetExclusivityResolver:
    """Enforces target exclusivity: each target T in S2/S3 can link to at most one S1."""

    def __init__(self, greedy: bool = True):
        self.greedy = greedy

    def resolve(
        self,
        predictions: Dict[str, Set[str]],
        pairs: List[Tuple[str, str]],
        scores: np.ndarray,
    ) -> Dict[str, Set[str]]:
        """Resolve competing assignments so that each target is claimed by the highest-probability S1."""
        pair_scores: Dict[Tuple[str, str], float] = {
            p: float(s) for p, s in zip(pairs, scores)
        }

        # Track which S1s claim which targets
        target_claims: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for sid, t_set in predictions.items():
            for tid in t_set:
                sc = pair_scores.get((sid, tid), 0.5)
                target_claims[tid].append((sid, sc))

        # Identify collisions
        final_preds: Dict[str, Set[str]] = {sid: set() for sid in predictions}

        for tid, claimants in target_claims.items():
            if len(claimants) == 1:
                final_preds[claimants[0][0]].add(tid)
            else:
                # Collision: retain highest-confidence claimant, reject weaker ones
                claimants.sort(key=lambda x: x[1], reverse=True)
                winner_sid, _ = claimants[0]
                final_preds[winner_sid].add(tid)

        return final_preds


class ContradictionChecker:
    """Conservative safety checks against obvious geographical or numeric contradictions."""

    def __init__(self, strict_country: bool = True):
        self.strict_country = strict_country

    def filter_predictions(
        self,
        predictions: Dict[str, Set[str]],
        s1_precomputed: Dict[str, Dict[str, Any]],
        target_precomputed: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Set[str]]:
        """Filter out candidate edges violating hard physical contradictions."""
        clean_preds: Dict[str, Set[str]] = {}

        for sid, t_set in predictions.items():
            s1_pre = s1_precomputed.get(sid)
            if not s1_pre:
                clean_preds[sid] = t_set
                continue

            filtered_targets: Set[str] = set()
            s1_country = s1_pre["country"]
            s1_first_num = s1_pre["addr"]["first_num"]
            s1_sal = s1_pre["addr"]["first_salient"]

            for tid in t_set:
                tgt_pre = target_precomputed.get(tid)
                if not tgt_pre:
                    filtered_targets.add(tid)
                    continue

                # 1. Country conflict
                if self.strict_country and s1_country != "unknown" and tgt_pre["country"] != "unknown":
                    if s1_country != tgt_pre["country"]:
                        continue

                # 2. Extreme house number conflict on completely different street
                tgt_first_num = tgt_pre["addr"]["first_num"]
                tgt_sal = tgt_pre["addr"]["first_salient"]
                if s1_sal and tgt_sal and s1_sal != tgt_sal and s1_first_num and tgt_first_num:
                    try:
                        diff = abs(int(s1_first_num) - int(tgt_first_num))
                        if diff > 500:
                            continue
                    except ValueError:
                        pass

                filtered_targets.add(tid)

            clean_preds[sid] = filtered_targets

        return clean_preds


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
    """Grid-search decision policy parameters to maximize Macro F0.5 on OOF validation."""
    if match_thresholds is None:
        match_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if singleton_thresholds is None:
        singleton_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    if score_margins is None:
        score_margins = [0.10, 0.15, 0.20, 0.25, 0.35]

    best_f05 = -1.0
    best_params = {}

    for mt in match_thresholds:
        for st in singleton_thresholds:
            for sm in score_margins:
                policy = PrecisionDecisionPolicy(
                    match_threshold=mt,
                    singleton_threshold=st,
                    score_margin=sm,
                )
                preds = policy.apply(pairs, val_scores, all_s1_ids)
                metrics = compute_macro_f05(preds, ground_truth)
                score = metrics["macro_f05"]

                if score > best_f05:
                    best_f05 = score
                    best_params = {
                        "match_threshold": mt,
                        "singleton_threshold": st,
                        "score_margin": sm,
                        "macro_f05": score,
                        "precision": metrics.get("macro_precision", metrics.get("precision", 0.0)),
                        "recall": metrics.get("macro_recall", metrics.get("recall", 0.0)),
                    }

    best_policy = PrecisionDecisionPolicy(
        match_threshold=best_params["match_threshold"],
        singleton_threshold=best_params["singleton_threshold"],
        score_margin=best_params["score_margin"],
    )
    return best_policy, best_params
