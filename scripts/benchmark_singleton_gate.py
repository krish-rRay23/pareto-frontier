"""Experiment: Dedicated Singleton / Abstention Gate for Zero-Match S1 Entities.

Evaluates on the exact 2,000 S1 validation set with V2.7 retrieval and LightGBM.
Compares:
1. Baseline LightGBM (No dedicated gate, current top: 0.9762)
2. Rule-Based Parametric Abstention Gate
3. Trained Entity-Level Singleton GBDT Classifier

Reports:
- Macro F0.5
- Singleton F0.5 / Accuracy
- False Merges
- Macro Recall
- Comparison against 0.9762
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
from src.models.classifier import LightGBMPairClassifier
from src.inference.decision_policy import optimize_decision_policy, PrecisionDecisionPolicy
from src.validation.splitters import get_folds_list
from src.validation.metrics import compute_macro_f05
import lightgbm as lgb

print("=== STARTING DEDICATED SINGLETON / ABSTENTION GATE EXPERIMENT ===")

# 1. Load exact 2,000 S1 sample
N_S1 = 2000
NOISE_RATIO = 25
s1_records, target_records, ground_truth = load_benchmark_subset(
    data_dir="resources/student_resource/dataset/train",
    n_s1=N_S1,
    background_noise_ratio=NOISE_RATIO,
)
s1_dict = {r["entity_id"]: r for r in s1_records}
target_dict = {r["entity_id"]: r for r in target_records}
s1_ids = list(s1_dict.keys())
s1_metadata = {sid: {"country": s1_dict[sid].get("country", "Unknown")} for sid in s1_ids}

# 2. V2.7 Blocker (Unchanged)
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

# 3. Pairwise Feature Matrix (Unchanged)
X_base, y_base, pairs = build_pairwise_feature_matrix(
    s1_dict=s1_dict,
    target_dict=target_dict,
    candidates_dict=candidates_dict,
    ground_truth=ground_truth,
)

groups = np.array([sid for sid, _ in pairs])
folds = get_folds_list(strategy="group", n_splits=5, groups=groups, seed=42)

# 4. Out-of-Fold Cross-Validation for LightGBM Pairwise Scorer
print("Running 5-Fold GroupKFold LightGBM Pairwise Scorer...")
oof_pair_scores = np.zeros(len(pairs), dtype=float)

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
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
    clf.fit(X_base.iloc[tr_idx], y_base[tr_idx], X_val=X_base.iloc[va_idx], y_val=y_base[va_idx], early_stopping_rounds=30)
    oof_pair_scores[va_idx] = clf.predict_proba(X_base.iloc[va_idx])

# Base Policy Optimization (Unchanged Baseline)
opt_policy, _ = optimize_decision_policy(
    pairs=pairs,
    val_scores=oof_pair_scores,
    ground_truth=ground_truth,
    all_s1_ids=s1_ids,
    match_thresholds=[0.35, 0.40, 0.45, 0.50],
    singleton_thresholds=[0.30, 0.35, 0.40],
    score_margins=[0.15, 0.20],
)

base_preds = opt_policy.apply(pairs, oof_pair_scores, all_s1_ids=s1_ids)
base_metrics = compute_macro_f05(ground_truth, base_preds, metadata=s1_metadata)

print("\n--- BASELINE CONFIG 1 (Current 0.9762 Target) ---")
print(f"  Macro F0.5:         {base_metrics['macro_f05']:.4f}")
print(f"  Macro Precision:    {base_metrics['macro_precision']*100:.2f}%")
print(f"  Macro Recall:       {base_metrics['macro_recall']*100:.2f}%")
print(f"  Singleton Accuracy: {base_metrics['singleton_accuracy']*100:.2f}% ({base_metrics['singleton_count']} singletons)")
print(f"  Total False Merges: {base_metrics['false_merges']}")

# 5. Extract Entity-Level Aggregated Signals for Singleton Detection
print("\nExtracting entity-level features for singleton detection...")
entity_pairs_map = defaultdict(list)
for idx, (sid, tid) in enumerate(pairs):
    entity_pairs_map[sid].append((tid, oof_pair_scores[idx], idx))

entity_feat_rows = []
entity_labels = [] # 1 if singleton (0 true matches), 0 otherwise

for sid in s1_ids:
    true_set = ground_truth.get(sid, set())
    is_singleton = 1 if len(true_set) == 0 else 0
    entity_labels.append(is_singleton)

    cand_info = entity_pairs_map.get(sid, [])
    if not cand_info:
        # 0 candidates retrieved -> trivially a singleton
        row = {
            "max_score": 0.0,
            "mean_score": 0.0,
            "score_std": 0.0,
            "top1_top2_margin": 0.0,
            "n_cands": 0,
            "n_score_gt_30": 0,
            "n_score_gt_40": 0,
            "max_token_jaccard": 0.0,
            "max_ngram_jaccard": 0.0,
            "max_num_match": 0.0,
            "s1_name_len": len(s1_dict[sid].get("business_name", "").split()),
            "s1_has_addr": 1.0 if s1_dict[sid].get("business_address") else 0.0,
        }
    else:
        scores = [sc for _, sc, _ in cand_info]
        idxs = [i for _, _, i in cand_info]
        scores_sorted = sorted(scores, reverse=True)
        top1 = scores_sorted[0]
        top2 = scores_sorted[1] if len(scores_sorted) > 1 else 0.0

        sub_X = X_base.iloc[idxs]
        max_jacc = sub_X["name_token_jaccard"].max()
        max_ngram = sub_X["name_char_3gram_jaccard"].max()
        max_num = sub_X["addr_number_match"].max() if "addr_number_match" in sub_X else 0.0

        row = {
            "max_score": top1,
            "mean_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "top1_top2_margin": top1 - top2,
            "n_cands": len(scores),
            "n_score_gt_30": sum(1 for s in scores if s >= 0.30),
            "n_score_gt_40": sum(1 for s in scores if s >= 0.40),
            "max_token_jaccard": max_jacc,
            "max_ngram_jaccard": max_ngram,
            "max_num_match": max_num,
            "s1_name_len": len(s1_dict[sid].get("business_name", "").split()),
            "s1_has_addr": 1.0 if s1_dict[sid].get("business_address") else 0.0,
        }
    entity_feat_rows.append(row)

entity_df = pd.DataFrame(entity_feat_rows)
entity_y = np.array(entity_labels, dtype=int)

# 6. Test Model A: Parametric Rule-Based Abstention Gate
# Strategy: If predicted matches exist, but top score is borderline (< 0.50), token jaccard < 0.50, and addr_num_match == 0 -> abstain
print("\n--- EVALUATING APPROACH A: Parametric Rule-Based Abstention Gate ---")
gate_preds_rule = dict(base_preds)
abstained_count_rule = 0

for sid in s1_ids:
    m = base_preds.get(sid, set())
    if not m:
        continue
    cand_info = entity_pairs_map.get(sid, [])
    scores_sorted = sorted([sc for _, sc, _ in cand_info], reverse=True)
    top1 = scores_sorted[0]
    top2 = scores_sorted[1] if len(scores_sorted) > 1 else 0.0

    idxs = [i for _, _, i in cand_info]
    sub_X = X_base.iloc[idxs]
    max_jacc = sub_X["name_token_jaccard"].max()
    max_num = sub_X["addr_number_match"].max() if "addr_number_match" in sub_X else 0.0

    # Rule: If top score is weak (< 0.48), name similarity is mediocre (< 0.55), and no address number match -> abstain
    if top1 < 0.48 and max_jacc < 0.55 and max_num < 0.5 and (top1 - top2) < 0.08:
        gate_preds_rule[sid] = set()
        abstained_count_rule += 1

m_rule = compute_macro_f05(ground_truth, gate_preds_rule, metadata=s1_metadata)
print(f"  Abstained on:       {abstained_count_rule} entities")
print(f"  Macro F0.5:         {m_rule['macro_f05']:.4f} (Delta: {m_rule['macro_f05'] - base_metrics['macro_f05']:+.4f})")
print(f"  Macro Precision:    {m_rule['macro_precision']*100:.2f}%")
print(f"  Macro Recall:       {m_rule['macro_recall']*100:.2f}%")
print(f"  Singleton Accuracy: {m_rule['singleton_accuracy']*100:.2f}%")
print(f"  Total False Merges: {m_rule['false_merges']}")

# 7. Test Model B: Dedicated 5-Fold Trained Entity-Level Singleton Classifier
print("\n--- EVALUATING APPROACH B: Trained Entity-Level Singleton Classifier ---")
oof_singleton_prob = np.zeros(len(s1_ids), dtype=float)

# GroupKFold on S1 entity index
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(entity_df):
    X_tr_ent, y_tr_ent = entity_df.iloc[train_idx], entity_y[train_idx]
    X_va_ent, y_va_ent = entity_df.iloc[val_idx], entity_y[val_idx]

    ent_clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    ent_clf.fit(X_tr_ent, y_tr_ent)
    oof_singleton_prob[val_idx] = ent_clf.predict_proba(X_va_ent)[:, 1]

# Tune singleton probability threshold
best_gate_f05 = -1.0
best_thresh = 0.5
best_preds_lgb = {}

for thresh in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
    temp_preds = dict(base_preds)
    for idx, sid in enumerate(s1_ids):
        if oof_singleton_prob[idx] >= thresh:
            temp_preds[sid] = set() # Abstain

    m_eval = compute_macro_f05(ground_truth, temp_preds, metadata=s1_metadata)
    if m_eval["macro_f05"] > best_gate_f05:
        best_gate_f05 = m_eval["macro_f05"]
        best_thresh = thresh
        best_preds_lgb = temp_preds

m_trained = compute_macro_f05(ground_truth, best_preds_lgb, metadata=s1_metadata)
print(f"  Optimal P(Singleton) Cutoff: {best_thresh}")
print(f"  Macro F0.5:         {m_trained['macro_f05']:.4f} (Delta: {m_trained['macro_f05'] - base_metrics['macro_f05']:+.4f})")
print(f"  Macro Precision:    {m_trained['macro_precision']*100:.2f}%")
print(f"  Macro Recall:       {m_trained['macro_recall']*100:.2f}%")
print(f"  Singleton Accuracy: {m_trained['singleton_accuracy']*100:.2f}%")
print(f"  Total False Merges: {m_trained['false_merges']}")

print("\n" + "=" * 105)
print(f"{'CONFIGURATION':45s} | {'MACRO F0.5':10s} | {'PRECISION':9s} | {'RECALL':8s} | {'SINGLETON':9s} | {'FP-MERGES':9s}")
print("=" * 105)
print(f"{'1. Baseline LightGBM (No Extra Gate)':45s} | {base_metrics['macro_f05']:10.4f} | {base_metrics['macro_precision']*100:8.2f}% | {base_metrics['macro_recall']*100:7.2f}% | {base_metrics['singleton_accuracy']*100:8.2f}% | {base_metrics['false_merges']:9d}")
print(f"{'2. Parametric Rule-Based Abstention Gate':45s} | {m_rule['macro_f05']:10.4f} | {m_rule['macro_precision']*100:8.2f}% | {m_rule['macro_recall']*100:7.2f}% | {m_rule['singleton_accuracy']*100:8.2f}% | {m_rule['false_merges']:9d}")
print(f"{'3. Trained Entity-Level Singleton GBDT Gate':45s} | {m_trained['macro_f05']:10.4f} | {m_trained['macro_precision']*100:8.2f}% | {m_trained['macro_recall']*100:7.2f}% | {m_trained['singleton_accuracy']*100:8.2f}% | {m_trained['false_merges']:9d}")
print("=" * 105)
