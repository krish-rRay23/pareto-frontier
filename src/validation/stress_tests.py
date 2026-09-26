"""Stress-testing and error slice auditing suite for entity resolution."""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.validation.metrics import compute_macro_f05


def run_error_slice_audit(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    s1_precomputed: Dict[str, Dict[str, Any]],
    target_precomputed: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit performance across all critical error slices.

    Slices:
    1. Overall Macro F0.5, Precision, Recall
    2. Singleton Accuracy
    3. Missing Name Slice
    4. Missing Address Slice
    5. Transliteration (Indic script) Slice
    6. Non-US / Foreign / France Slice
    7. Multi-Match S1 Slice (S1 entities with >= 2 true targets)
    """
    total_eval = compute_macro_f05(predictions, ground_truth)

    slice_sids: Dict[str, List[str]] = {
        "missing_name": [],
        "missing_addr": [],
        "transliteration": [],
        "foreign_domain": [],
        "multi_match": [],
        "singletons": [],
    }

    for sid, true_set in ground_truth.items():
        s1_pre = s1_precomputed.get(sid)
        if not s1_pre:
            continue

        nv = s1_pre["name"]
        av = s1_pre["addr"]
        country = s1_pre["country"]

        if not nv["clean"]:
            slice_sids["missing_name"].append(sid)
        if not av["clean"]:
            slice_sids["missing_addr"].append(sid)
        if nv["is_indic"]:
            slice_sids["transliteration"].append(sid)
        if country not in ("us", "unknown"):
            slice_sids["foreign_domain"].append(sid)
        if len(true_set) >= 2:
            slice_sids["multi_match"].append(sid)
        if len(true_set) == 0:
            slice_sids["singletons"].append(sid)

    slice_results: Dict[str, Any] = {
        "overall": total_eval,
    }

    for s_name, sids in slice_sids.items():
        if not sids:
            slice_results[s_name] = {"count": 0, "macro_f05": 0.0}
            continue
        sub_gt = {sid: ground_truth[sid] for sid in sids}
        sub_pred = {sid: predictions.get(sid, set()) for sid in sids}
        sub_metrics = compute_macro_f05(sub_pred, sub_gt)
        slice_results[s_name] = {
            "count": len(sids),
            "macro_f05": sub_metrics["macro_f05"],
            "precision": sub_metrics.get("macro_precision", sub_metrics.get("precision", 0.0)),
            "recall": sub_metrics.get("macro_recall", sub_metrics.get("recall", 0.0)),
        }

    return slice_results
