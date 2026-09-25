"""In-depth Error Autopsy & Quantitative Pareto Analysis for Baseline V1."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.abspath("."))
import csv
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

from src.data.loader import load_benchmark_subset
from src.inference.pipeline import EntityResolutionPipeline
from src.validation.metrics import compute_entity_f05, compute_macro_f05

print("=== STARTING DEEP ERROR AUTOPSY ON OOF PREDICTIONS ===")

# 1. Load 2,000 S1 entities with representative background
N_SAMPLE = 2000
s1_records, target_records, ground_truth = load_benchmark_subset(
    data_dir="resources/student_resource/dataset/train",
    n_s1=N_SAMPLE,
    background_noise_ratio=25,
)

s1_dict = {r["entity_id"]: r for r in s1_records}
target_dict = {r["entity_id"]: r for r in target_records}
s1_ids = list(s1_dict.keys())

print(f"Loaded {len(s1_records):,} S1 records and {len(target_records):,} target records.")

# 2. Run Baseline V1 Pipeline to get OOF predictions and candidate sets
pipeline = EntityResolutionPipeline(
    max_candidates_per_entity=50,
    enable_consensus_features=True,
    lgb_params={"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6},
)

# Step 1: Blocker
print("Fitting blocker and generating candidates...")
pipeline.blocker.fit_targets(target_records)
candidates_dict = pipeline.blocker.block_all(s1_records)

# Step 2: Fit and get OOF probabilities
metrics = pipeline.fit_and_validate(
    s1_records=s1_records,
    target_records=target_records,
    ground_truth=ground_truth,
    n_splits=5,
    optimize_policy=True,
    verbose=False,
)

print(f"\nBaseline V1 OOF Results:")
print(f"  Macro F0.5:      {metrics['macro_f05']:.4f}")
print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
print(f"  Singleton Acc:   {metrics['singleton_accuracy']:.4f}")
print(f"  Candidate Recall:{metrics['candidate_recall']*100:.2f}%")
print(f"  False Merges:    {metrics['false_merges']}")

# 3. Analyze every entity with F0.5 < 1.0
# We need to know: for each entity, what was true, what was predicted, and what was in candidates!
# Let's extract the pair probabilities from the fitted pipeline
from src.features.pairwise_features import build_pairwise_feature_matrix, precompute_record_views
from src.features.consensus import compute_s2_s3_consensus

X, y, pairs = build_pairwise_feature_matrix(
    s1_dict=s1_dict,
    target_dict=target_dict,
    candidates_dict=candidates_dict,
    ground_truth=ground_truth,
)
if pipeline.enable_consensus_features and not X.empty:
    active_target_ids = {tid for _, tid in pairs}
    target_precomputed = {
        tid: precompute_record_views(target_dict[tid])
        for tid in active_target_ids
        if tid in target_dict
    }
    consensus_df = compute_s2_s3_consensus(pairs, target_dict, target_precomputed)
    X = pd.concat([X, consensus_df], axis=1)

scores = pipeline.model.predict_proba(X)
predicted_matches = pipeline.policy.apply(pairs, scores, all_s1_ids=s1_ids)

# Group scored pairs for analysis
pair_scores = {(sid, tid): sc for (sid, tid), sc in zip(pairs, scores)}

# Helper functions for taxonomy classification
NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]+')

def is_transliteration(s1_name, tgt_name):
    # Check if either has non-ascii characters (Devanagari, Bengali, Tamil, etc.)
    has_non_ascii_1 = bool(NON_ASCII_PATTERN.search(s1_name))
    has_non_ascii_2 = bool(NON_ASCII_PATTERN.search(tgt_name))
    return has_non_ascii_1 or has_non_ascii_2

def is_missing_address(addr):
    if not addr or not str(addr).strip() or str(addr).strip().lower() in {'nan', 'none', 'null'}:
        return True
    return False

# Taxonomy tally
lost_f05_by_category = defaultdict(float)
count_by_category = Counter()
detailed_failures = defaultdict(list)

total_entities = len(s1_ids)

for sid in s1_ids:
    true_set = ground_truth.get(sid, set())
    pred_set = predicted_matches.get(sid, set())
    cand_set = candidates_dict.get(sid, set())

    eval_res = compute_entity_f05(true_set, pred_set)
    f05 = eval_res["f05"]
    deficit = 1.0 - f05

    if deficit <= 1e-6:
        continue  # Perfect prediction (1.0)

    s1_rec = s1_dict[sid]
    s1_name = s1_rec["business_name"]
    s1_addr = s1_rec["business_address"]
    is_singleton = len(true_set) == 0

    # Categorize Error
    # A. False Merges on Singletons
    if is_singleton and len(pred_set) > 0:
        cat = "singleton_false_merge"
        lost_f05_by_category[cat] += deficit
        count_by_category[cat] += 1
        detailed_failures[cat].append({
            "sid": sid,
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "pred_mids": list(pred_set),
            "pred_details": [target_dict.get(p, {}) for p in pred_set],
            "deficit": deficit,
        })
        continue

    # B. False Merges on Non-Singletons (Precision loss)
    false_positives = pred_set - true_set
    false_negatives = true_set - pred_set

    # Analyze False Negatives (Missed matches)
    for fn_id in false_negatives:
        tgt_rec = target_dict.get(fn_id, {})
        tgt_name = tgt_rec.get("business_name", "")
        tgt_addr = tgt_rec.get("business_address", "")

        # Did blocker miss it entirely?
        if fn_id not in cand_set:
            if is_transliteration(s1_name, tgt_name):
                cat = "blocker_miss_transliteration"
            elif is_missing_address(s1_addr) or is_missing_address(tgt_addr):
                cat = "blocker_miss_missing_address"
            else:
                cat = "blocker_miss_lexical"
        else:
            # Candidate was present in blocker, but model rejected it (Threshold error / model scoring)
            sc = pair_scores.get((sid, fn_id), 0.0)
            if sc >= pipeline.policy.match_threshold:
                cat = "margin_gate_cut"  # Top score was higher and secondary cut by margin
            else:
                cat = "model_under_threshold"

        lost_f05_by_category[cat] += deficit * (1.0 / max(1, len(false_negatives) + len(false_positives)))
        count_by_category[cat] += 1
        detailed_failures[cat].append({
            "sid": sid,
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "tgt_id": fn_id,
            "tgt_name": tgt_name,
            "tgt_addr": tgt_addr,
            "score": pair_scores.get((sid, fn_id), 0.0),
            "deficit": deficit,
        })

    # Analyze False Positives (Wrong Merges)
    for fp_id in false_positives:
        tgt_rec = target_dict.get(fp_id, {})
        tgt_name = tgt_rec.get("business_name", "")
        tgt_addr = tgt_rec.get("business_address", "")

        # Check if same address but different name (address collision)
        if s1_addr and tgt_addr and s1_addr.lower() == tgt_addr.lower():
            cat = "address_collision_different_business"
        # Check if same name but different address (name collision / chain)
        elif s1_name and tgt_name and s1_name.lower() == tgt_name.lower():
            cat = "name_collision_different_location"
        else:
            cat = "hard_negative_scoring_error"

        lost_f05_by_category[cat] += deficit * (1.0 / max(1, len(false_negatives) + len(false_positives)))
        count_by_category[cat] += 1
        detailed_failures[cat].append({
            "sid": sid,
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "fp_id": fp_id,
            "fp_name": tgt_name,
            "fp_addr": tgt_addr,
            "score": pair_scores.get((sid, fp_id), 0.0),
            "deficit": deficit,
        })

# 4. Summary Table & Pareto Breakdown
total_lost_f05 = sum(lost_f05_by_category.values())
mean_lost_macro = total_lost_f05 / total_entities

print("\n" + "=" * 95)
print(f"{'FAILURE TAXONOMY CLASS':40s} | {'COUNT':6s} | {'LOST F0.5 PTS':14s} | {'% OF TOTAL LOSS':16s} | {'CUMULATIVE %'}")
print("=" * 95)

sorted_cats = sorted(lost_f05_by_category.items(), key=lambda x: x[1], reverse=True)
cum_pct = 0.0

for cat, lost in sorted_cats:
    pct = (lost / total_lost_f05) * 100.0 if total_lost_f05 > 0 else 0.0
    cum_pct += pct
    cnt = count_by_category[cat]
    print(f"{cat:40s} | {cnt:6d} | {lost:14.4f} | {pct:14.2f}% | {cum_pct:10.2f}%")

print("=" * 95)
print(f"Total Lost Macro F0.5 Points across sample: {total_lost_f05:.4f} (translates to {mean_lost_macro:.4f} macro F0.5 gap)")

# Print concrete examples for the top 3 categories
print("\n=== CONCRETE EXAMPLES OF TOP ERROR CATEGORIES ===")
for cat, _ in sorted_cats[:3]:
    print(f"\n--- Category: {cat} (Count: {count_by_category[cat]}) ---")
    examples = detailed_failures[cat][:4]
    for i, ex in enumerate(examples, 1):
        if "pred_mids" in ex:
            print(f"[{i}] Singleton S1: [{ex['sid']}] \"{ex['s1_name']}\" | \"{ex['s1_addr']}\"")
            print(f"    False Merges: {ex['pred_mids']}")
        elif "tgt_id" in ex:
            print(f"[{i}] S1: [{ex['sid']}] \"{ex['s1_name']}\" | \"{ex['s1_addr']}\"")
            print(f"    Missed Match: [{ex['tgt_id']}] \"{ex['tgt_name']}\" | \"{ex['tgt_addr']}\" (Model score: {ex.get('score', 0):.4f})")
        elif "fp_id" in ex:
            print(f"[{i}] S1: [{ex['sid']}] \"{ex['s1_name']}\" | \"{ex['s1_addr']}\"")
            print(f"    Wrong Merge: [{ex['fp_id']}] \"{ex['fp_name']}\" | \"{ex['fp_addr']}\" (Model score: {ex.get('score', 0):.4f})")
