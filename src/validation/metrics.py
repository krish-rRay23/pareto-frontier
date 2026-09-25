"""Official ML Challenge 2026 Macro-Averaged F0.5 Metric Module.

Precision-heavy metric that penalizes false merges more than missed matches:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Evaluation rules:
- Macro-averaged across ALL Source 1 entities in the evaluation set.
- Singletons: If ground truth has no matches, correctly predicting an empty list
  scores 1.0; predicting any false match scores 0.0.
- Entities with true matches: standard precision, recall, and F0.5.
  If predicted list is empty or tp == 0, score is 0.0.
"""

from typing import Dict, List, Set, Union, Optional
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score from precision and recall.
    
    Default beta=0.5 gives 2x weight to precision over recall:
        beta_sq = 0.25
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> Dict[str, float]:
    """Compute F0.5, precision, recall for a single Source 1 entity."""
    is_singleton = len(true_ids) == 0
    pred_count = len(pred_ids)

    if is_singleton:
        # Singleton entity: 1.0 if empty list, 0.0 if any match predicted
        score = 1.0 if pred_count == 0 else 0.0
        return {
            "f05": score,
            "precision": 1.0 if pred_count == 0 else 0.0,
            "recall": 1.0 if pred_count == 0 else 0.0,
            "tp": 0,
            "fp": pred_count,
            "fn": 0,
            "is_singleton": 1.0,
        }

    # Non-singleton entity
    if pred_count == 0:
        return {
            "f05": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(true_ids),
            "is_singleton": 0.0,
        }

    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    if tp == 0:
        return {
            "f05": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": fp,
            "fn": fn,
            "is_singleton": 0.0,
        }

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f05 = compute_f_beta(prec, rec, beta=0.5)

    return {
        "f05": f05,
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "is_singleton": 0.0,
    }


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    metadata: Optional[Dict[str, Dict]] = None,
) -> Dict[str, float]:
    """Compute official competition Macro F0.5 across all S1 entities.

    Args:
        ground_truth: Mapping from source1_entity_id to set of true matching IDs.
        predictions: Mapping from source1_entity_id to set of predicted matching IDs.
        metadata: Optional dictionary mapping source1_entity_id to metadata (e.g. 'country').

    Returns:
        Structured evaluation metrics dictionary.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "total_entities": 0,
            "false_merges": 0,
        }

    scores = []
    precisions = []
    recalls = []
    singleton_scores = []
    matched_scores = []
    total_fp = 0
    total_tp = 0
    total_fn = 0

    country_scores = {}
    bucket_scores = {}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        res = compute_entity_f05(true_set, pred_set)

        scores.append(res["f05"])
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        total_fp += res["fp"]
        total_tp += res["tp"]
        total_fn += res["fn"]

        if res["is_singleton"] > 0.5:
            singleton_scores.append(res["f05"])
        else:
            matched_scores.append(res["f05"])

        # Match-count bucket categorization
        n_true = len(true_set)
        if n_true == 0:
            b_key = "bucket_0"
        elif n_true == 1:
            b_key = "bucket_1"
        elif n_true in {2, 3}:
            b_key = "bucket_2_3"
        else:
            b_key = "bucket_4_plus"

        if b_key not in bucket_scores:
            bucket_scores[b_key] = []
        bucket_scores[b_key].append(res["f05"])

        if metadata and s1_id in metadata:
            c = metadata[s1_id].get("country", "Unknown")
            if c not in country_scores:
                country_scores[c] = []
            country_scores[c].append(res["f05"])

    summary = {
        "macro_f05": float(np.mean(scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "matched_entity_f05": float(np.mean(matched_scores)) if matched_scores else 0.0,
        "total_entities": total_entities,
        "singleton_count": len(singleton_scores),
        "false_merges": int(total_fp),
        "true_positives": int(total_tp),
        "false_negatives": int(total_fn),
    }

    for c, c_scores in country_scores.items():
        summary[f"macro_f05_{c.lower()}"] = float(np.mean(c_scores))

    for b, b_scores in bucket_scores.items():
        summary[f"macro_f05_{b}"] = float(np.mean(b_scores))
        summary[f"count_{b}"] = len(b_scores)

    return summary

