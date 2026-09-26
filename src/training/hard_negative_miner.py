"""Iterative Hard-Negative Mining Engine for Entity Resolution.

Mines structured negative buckets specified in Amazon ML Challenge 2026 99.99+ Plan:
1. Same name / different address
2. Same address / different name
3. Same locality / nearby house numbers
4. High TF-IDF / BM25 false positives
5. Reverse retrieval false positives
6. High-confidence model false positives (hard false alarms)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class HardNegativeMiner:
    """Mines diverse, density-matched hard negatives to train high-precision entity matchers."""

    def __init__(
        self,
        max_negatives_per_positive: int = 15,
        min_model_prob_threshold: float = 0.35,
        random_state: int = 42,
    ):
        self.max_negatives_per_positive = max_negatives_per_positive
        self.min_model_prob_threshold = min_model_prob_threshold
        self.random_state = random_state

    def mine_hard_negatives(
        self,
        candidate_pairs: List[Tuple[str, str]],
        ground_truth: Dict[str, Set[str]],
        s1_precomputed: Dict[str, Dict[str, Any]],
        target_precomputed: Dict[str, Dict[str, Any]],
        model_probabilities: Optional[np.ndarray] = None,
        retrieval_provenances: Optional[Dict[Tuple[str, str], Any]] = None,
    ) -> List[Tuple[str, str]]:
        """Mine categorized hard negative pairs from candidate pool."""
        rng = np.random.RandomState(self.random_state)

        positives_set = set()
        for sid, t_set in ground_truth.items():
            for tid in t_set:
                positives_set.add((sid, tid))

        # Categorize candidates
        neg_buckets: Dict[str, List[Tuple[str, str]]] = {
            "model_fp": [],
            "same_name_diff_addr": [],
            "same_addr_diff_name": [],
            "nearby_hnum_locality": [],
            "bm25_fp": [],
            "reverse_fp": [],
            "general_cands": [],
        }

        for idx, (sid, tid) in enumerate(candidate_pairs):
            if (sid, tid) in positives_set:
                continue

            s1_pre = s1_precomputed.get(sid)
            tgt_pre = target_precomputed.get(tid)
            if not s1_pre or not tgt_pre:
                continue

            prob = float(model_probabilities[idx]) if model_probabilities is not None else 0.0

            # Bucket 1: High model probability false positive
            if prob >= self.min_model_prob_threshold:
                neg_buckets["model_fp"].append((sid, tid))
                continue

            # Bucket 2: Same/near name, different address
            name_sim = (s1_pre["name"]["clean"] == tgt_pre["name"]["clean"]) or (
                s1_pre["name"]["sorted_tokens"] and s1_pre["name"]["sorted_tokens"] == tgt_pre["name"]["sorted_tokens"]
            )
            addr_diff = s1_pre["addr"]["clean"] != tgt_pre["addr"]["clean"]
            if name_sim and addr_diff:
                neg_buckets["same_name_diff_addr"].append((sid, tid))
                continue

            # Bucket 3: Same address, different name
            addr_same = s1_pre["addr"]["clean"] and (s1_pre["addr"]["clean"] == tgt_pre["addr"]["clean"])
            if addr_same and not name_sim:
                neg_buckets["same_addr_diff_name"].append((sid, tid))
                continue

            # Bucket 4: Locality / Postal match with different house number
            postal_same = (
                s1_pre["addr"]["first_postal"]
                and s1_pre["addr"]["first_postal"] == tgt_pre["addr"]["first_postal"]
            )
            hnum_diff = (
                s1_pre["addr"]["first_num"]
                and tgt_pre["addr"]["first_num"]
                and s1_pre["addr"]["first_num"] != tgt_pre["addr"]["first_num"]
            )
            if postal_same and hnum_diff:
                neg_buckets["nearby_hnum_locality"].append((sid, tid))
                continue

            # Bucket 5: Retrieval-specific FP
            if retrieval_provenances:
                prov = retrieval_provenances.get((sid, tid))
                if prov:
                    if "bm25_name" in prov.channels or "bm25_addr" in prov.channels:
                        neg_buckets["bm25_fp"].append((sid, tid))
                        continue
                    if "reverse" in prov.channels:
                        neg_buckets["reverse_fp"].append((sid, tid))
                        continue

            neg_buckets["general_cands"].append((sid, tid))

        # Sample across buckets with priority to hard errors
        selected_negatives: List[Tuple[str, str]] = []
        n_pos = len(positives_set)
        target_neg_count = max(100, n_pos * self.max_negatives_per_positive)

        # Allocate quota to hard categories first
        for cat in [
            "model_fp",
            "same_name_diff_addr",
            "same_addr_diff_name",
            "nearby_hnum_locality",
            "bm25_fp",
            "reverse_fp",
        ]:
            b_list = neg_buckets[cat]
            if b_list:
                take = min(len(b_list), target_neg_count // 4)
                idx_choice = rng.choice(len(b_list), size=take, replace=False)
                selected_negatives.extend([b_list[i] for i in idx_choice])

        # Fill remaining with general candidate pool negatives
        rem = target_neg_count - len(selected_negatives)
        if rem > 0 and neg_buckets["general_cands"]:
            gen_list = neg_buckets["general_cands"]
            take = min(len(gen_list), rem)
            idx_choice = rng.choice(len(gen_list), size=take, replace=False)
            selected_negatives.extend([gen_list[i] for i in idx_choice])

        return selected_negatives
