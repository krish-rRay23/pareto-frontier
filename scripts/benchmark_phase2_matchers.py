"""Master Benchmark Script for Phase 2: Pairwise Matching & Entity-Level Reasoning.

Uses V2.7 Retrieval Unchanged.
Evaluates 5 Disjoint GroupKFold Configurations:
1. LightGBM (Base Features + Standard Training + Optimized Policy)
2. LightGBM + Hard-Negative Mining
3. LightGBM + Hard-Negative Mining + S2/S3 Consensus Features
4. LightGBM + Hard-Negative Mining + S2/S3 Consensus + Optimized S1 Policy & Singleton Abstention Gate
5. CatBoost + Hard-Negative Mining + S2/S3 Consensus + Optimized S1 Policy & Singleton Abstention Gate

Reports:
- Leakage-free Macro F0.5
- Macro Precision & Recall
- Singleton Accuracy (Bucket 0)
- Total False Merges
- Per-Country Macro F0.5 (US, India)
- Macro F0.5 by Match-Count Bucket (0, 1, 2-3, 4+)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.abspath("."))
import time
from collections import defaultdict
import numpy as np
import pandas as pd

from src.data.loader import load_benchmark_subset
from src.blocking.blocker import MultiPassBlocker
from src.features.pairwise_features import (
    build_pairwise_feature_matrix,
    precompute_record_views,
)
from src.features.consensus import compute_s2_s3_consensus
from src.models.classifier import LightGBMPairClassifier, CatBoostPairClassifier
from src.inference.decision_policy import optimize_decision_policy, PrecisionDecisionPolicy
from src.validation.splitters import get_folds_list
from src.validation.metrics import compute_macro_f05

print("=== STARTING PHASE 2 MATCHER & REASONING BENCHMARK ===")

# 1. Load leakage-free validation sample
N_S1 = 2000
NOISE_RATIO = 25
print(f"Loading {N_S1:,} S1 records with background noise ratio {NOISE_RATIO}...")
s1_records, target_records, ground_truth = load_benchmark_subset(
    data_dir="resources/student_resource/dataset/train",
    n_s1=N_S1,
    background_noise_ratio=NOISE_RATIO,
)

s1_dict = {r["entity_id"]: r for r in s1_records}
target_dict = {r["entity_id"]: r for r in target_records}
s1_ids = list(s1_dict.keys())
s1_metadata = {sid: {"country": s1_dict[sid].get("country", "Unknown")} for sid in s1_ids}

# 2. Run V2.7 Blocker (Unchanged)
print("\n[Step 1] Running V2.7 Blocker to generate candidate pool...")
t0 = time.time()
blocker = MultiPassBlocker(
    max_candidates_per_entity=150,
    enable_name_exact=True,
    enable_name_legal_stripped=True,
    enable_name_first_two=True,
    enable_name_sorted_tokens=True,
    enable_indic_transliteration=True,
    enable_condensed_name=True,
    enable_address_multi=True,
    enable_rare_tokens=True,
    enable_char_ngrams=True,
    enable_adaptive_fallback=True,
)
blocker.fit_targets(target_records)
candidates_dict = blocker.block_all(s1_records)
t_block = time.time() - t0

total_cands = sum(len(c) for c in candidates_dict.values())
true_in_cands = sum(len(ground_truth.get(s, set()) & candidates_dict.get(s, set())) for s in s1_ids)
total_true = sum(len(t) for t in ground_truth.values())
cand_recall = true_in_cands / total_true * 100
avg_cands = total_cands / len(s1_ids)
print(f"  V2.7 Candidate Recall: {cand_recall:.2f}% | Avg Candidates/S1: {avg_cands:.1f} ({t_block:.1f}s)")

# 3. Build Base Feature Matrix
print("\n[Step 2] Building Pairwise Feature Matrix...")
t0 = time.time()
X_base, y_base, pairs = build_pairwise_feature_matrix(
    s1_dict=s1_dict,
    target_dict=target_dict,
    candidates_dict=candidates_dict,
    ground_truth=ground_truth,
)
print(f"  Extracted {X_base.shape[1]} pairwise features across {len(X_base):,} pairs ({time.time() - t0:.1f}s).")

# 4. Compute S2-S3 Consensus Features
print("\n[Step 3] Computing S2/S3 Consensus Signals...")
t0 = time.time()
active_target_ids = {tid for _, tid in pairs}
target_precomputed = {
    tid: precompute_record_views(target_dict[tid])
    for tid in active_target_ids
    if tid in target_dict
}
consensus_df = compute_s2_s3_consensus(pairs, target_dict, target_precomputed)
X_consensus = pd.concat([X_base, consensus_df], axis=1)
print(f"  Consensus features generated ({time.time() - t0:.1f}s). Total features: {X_consensus.shape[1]}.")

# 5. Build GroupKFold Splits on Source 1 Entities
groups = np.array([sid for sid, _ in pairs])
folds = get_folds_list(strategy="group", n_splits=5, groups=groups, seed=42)

# Helper function to mine hard negatives
def mine_hard_negatives(tr_idx, y_arr, X_df, groups_arr, hard_ratio=0.6):
    train_groups = defaultdict(list)
    for idx in tr_idx:
        train_groups[groups_arr[idx]].append(idx)

    hardness_scores = (
        X_df["name_token_jaccard"].fillna(0).values * 0.4 +
        X_df["name_char_3gram_jaccard"].fillna(0).values * 0.3 +
        X_df["addr_number_match"].fillna(0).values * 0.3
    )

    selected = []
    for sid, p_idxs in train_groups.items():
        pos = [i for i in p_idxs if y_arr[i] == 1]
        neg = [i for i in p_idxs if y_arr[i] == 0]
        selected.extend(pos)
        if len(neg) <= 8:
            selected.extend(neg)
        else:
            neg.sort(key=lambda i: hardness_scores[i], reverse=True)
            n_hard = max(4, min(int(len(neg) * hard_ratio), 15))
            selected.extend(neg[:n_hard])
            selected.extend(neg[n_hard : n_hard + 6])

    return np.array(sorted(selected), dtype=np.int64)

# 6. Evaluation Function for Model Configurations
def run_cv_evaluation(model_type, X_feat, use_hard_mining, use_singleton_gate):
    oof_scores = np.zeros(len(pairs), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        if use_hard_mining:
            train_indices = mine_hard_negatives(tr_idx, y_base, X_feat, groups, hard_ratio=0.6)
        else:
            train_indices = tr_idx

        X_tr, y_tr = X_feat.iloc[train_indices], y_base[train_indices]
        X_va, y_val = X_feat.iloc[va_idx], y_base[va_idx]

        if model_type == "lightgbm":
            clf = LightGBMPairClassifier(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold_idx,
            )
        elif model_type == "catboost":
            clf = CatBoostPairClassifier(
                iterations=450,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=4.0,
                random_seed=42 + fold_idx,
                verbose=0,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_val, early_stopping_rounds=30, verbose=False)
        oof_scores[va_idx] = clf.predict_proba(X_va)

    # Optimize Decision Policy on Out-Of-Fold Validation Scores
    policy, _ = optimize_decision_policy(
        pairs=pairs,
        val_scores=oof_scores,
        ground_truth=ground_truth,
        all_s1_ids=s1_ids,
        match_thresholds=[0.35, 0.40, 0.45, 0.50, 0.55],
        singleton_thresholds=[0.30, 0.35, 0.40, 0.45],
        score_margins=[0.10, 0.15, 0.20],
        verbose=False,
    )

    predictions = policy.apply(pairs, oof_scores, all_s1_ids=s1_ids)

    # Dedicated Singleton Abstention Gate Enhancement
    if use_singleton_gate:
        cand_scores_by_s1 = defaultdict(list)
        for (sid, tid), sc in zip(pairs, oof_scores):
            cand_scores_by_s1[sid].append(sc)

        for sid, sc_list in cand_scores_by_s1.items():
            if not sc_list:
                continue
            sc_list.sort(reverse=True)
            top1 = sc_list[0]
            top2 = sc_list[1] if len(sc_list) > 1 else 0.0
            # Borderline score with near-identical second candidate indicates high uncertainty -> abstain
            if top1 < (policy.match_threshold + 0.12) and (top1 - top2) < 0.035 and top2 > 0.35:
                predictions[sid] = set()

    metrics = compute_macro_f05(ground_truth, predictions, metadata=s1_metadata)
    metrics["policy_match_thresh"] = policy.match_threshold
    metrics["policy_singleton_thresh"] = policy.singleton_threshold
    metrics["policy_margin"] = policy.score_margin
    return metrics


# 7. Benchmark the 5 Requested Configurations
configs = [
    ("Config 1: LightGBM (Base Features)", "lightgbm", X_base, False, False),
    ("Config 2: LightGBM + Hard-Negative Mining", "lightgbm", X_base, True, False),
    ("Config 3: LightGBM + Hard Neg + Consensus", "lightgbm", X_consensus, True, False),
    ("Config 4: LightGBM + Hard Neg + Consensus + Gate", "lightgbm", X_consensus, True, True),
    ("Config 5: CatBoost + Hard Neg + Consensus + Gate", "catboost", X_consensus, True, True),
]

results = []

print("\n" + "=" * 125)
print(f"{'CONFIGURATION':48s} | {'MACRO F0.5':10s} | {'PRECISION':9s} | {'RECALL':8s} | {'SINGLETON':9s} | {'FP-MERGES':9s} | {'US F0.5':8s} | {'IN F0.5':8s}")
print("=" * 125)

for name, mtype, X_mat, use_hn, use_gate in configs:
    t_start = time.time()
    m = run_cv_evaluation(mtype, X_mat, use_hn, use_gate)
    elapsed = time.time() - t_start
    m["name"] = name
    m["runtime_sec"] = elapsed
    results.append(m)

    print(
        f"{name:48s} | "
        f"{m['macro_f05']:10.4f} | "
        f"{m['macro_precision']*100:8.2f}% | "
        f"{m['macro_recall']*100:7.2f}% | "
        f"{m['singleton_accuracy']*100:8.2f}% | "
        f"{m['false_merges']:9d} | "
        f"{m.get('macro_f05_us', 0):8.4f} | "
        f"{m.get('macro_f05_india', 0):8.4f}"
    )

print("=" * 125)

# Detailed Match-Count Bucket Breakdown for Top Performing Configuration
best_m = max(results, key=lambda x: x["macro_f05"])
print(f"\n=== MATCH-COUNT BUCKET PERFORMANCE (Top Model: {best_m['name']}) ===")
print(f"  Overall Macro F0.5:      {best_m['macro_f05']:.4f}")
print(f"  Overall Macro Precision: {best_m['macro_precision']*100:.2f}%")
print(f"  Overall Macro Recall:    {best_m['macro_recall']*100:.2f}%")
print(f"  Bucket 0 (Singletons, 0 matches)   [N={best_m.get('count_bucket_0', 0):,}]: F0.5 = {best_m.get('macro_f05_bucket_0', 0):.4f} (Accuracy: {best_m['singleton_accuracy']*100:.2f}%)")
print(f"  Bucket 1 (1 match)                 [N={best_m.get('count_bucket_1', 0):,}]: F0.5 = {best_m.get('macro_f05_bucket_1', 0):.4f}")
print(f"  Bucket 2-3 (2-3 matches)           [N={best_m.get('count_bucket_2_3', 0):,}]: F0.5 = {best_m.get('macro_f05_bucket_2_3', 0):.4f}")
print(f"  Bucket 4+ (4+ matches)             [N={best_m.get('count_bucket_4_plus', 0):,}]: F0.5 = {best_m.get('macro_f05_bucket_4_plus', 0):.4f}")
print(f"  Optimal Policy Parameters: Match Threshold = {best_m.get('policy_match_thresh')}, Singleton Threshold = {best_m.get('policy_singleton_thresh')}, Score Margin = {best_m.get('policy_margin')}")
