"""Rigorous Retrieval Benchmark across Retrieval Versions V1 to V2.7.

Measures:
- Overall candidate recall
- Recall on transliteration cases (Indic scripts)
- Recall on Baseline V1 blocker-miss cases
- Recall by country (US, India)
- Candidate volume: average, p95, p99, max candidates per S1
- Runtime (fit + query)
- Memory usage (MB)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.abspath("."))
import time
import tracemalloc
from collections import defaultdict
import numpy as np

from src.data.loader import load_benchmark_subset
from src.blocking.blocker import MultiPassBlocker
from src.normalization.transliteration import has_indic_script

print("=== STARTING COMPREHENSIVE RETRIEVAL BENCHMARK ===")

# 1. Load large leakage-free validation sample
N_S1 = 5000
NOISE_RATIO = 25
print(f"Loading {N_S1:,} S1 records with background noise ratio {NOISE_RATIO}...")
s1_records, target_records, ground_truth = load_benchmark_subset(
    data_dir="resources/student_resource/dataset/train",
    n_s1=N_S1,
    background_noise_ratio=NOISE_RATIO,
)

s1_dict = {r["entity_id"]: r for r in s1_records}
target_dict = {r["entity_id"]: r for r in target_records}
total_targets = len(target_records)
total_true_matches = sum(len(mids) for mids in ground_truth.values())

print(f"Loaded {len(s1_records):,} S1 entities, {total_targets:,} target records.")
print(f"Total ground truth pairs in validation set: {total_true_matches:,}")

# Identify Ground Truth Slices:
# A. By Country
gt_by_country = defaultdict(dict)
for sid, tids in ground_truth.items():
    country = s1_dict[sid].get("country", "Unknown")
    gt_by_country[country][sid] = tids

# B. Transliteration Cases (ground truth targets that contain Indic characters)
gt_translit = defaultdict(set)
for sid, tids in ground_truth.items():
    for tid in tids:
        tgt_name = target_dict.get(tid, {}).get("business_name", "")
        if has_indic_script(tgt_name):
            gt_translit[sid].add(tid)
n_translit_pairs = sum(len(tids) for tids in gt_translit.values())
print(f"Transliteration ground truth pairs: {n_translit_pairs:,} (across {len(gt_translit):,} S1s)")

# 2. Run Baseline V1 first to capture initial blocker misses
print("\n[Step 1] Running Baseline V1 to capture initial blocker misses...")
blocker_v1 = MultiPassBlocker(
    max_candidates_per_entity=50,
    enable_name_exact=True,
    enable_name_legal_stripped=True,
    enable_name_first_two=True,
    enable_name_sorted_tokens=False,
    enable_indic_transliteration=False,
    enable_condensed_name=False,
    enable_address_multi=False,
    enable_rare_tokens=False,
    enable_char_ngrams=False,
    enable_adaptive_fallback=False,
)
blocker_v1.fit_targets(target_records)
cands_v1 = blocker_v1.block_all(s1_records)

# Identify Baseline V1 Blocker Misses
gt_v1_misses = defaultdict(set)
for sid, tids in ground_truth.items():
    found = cands_v1.get(sid, set())
    missed = tids - found
    if missed:
        gt_v1_misses[sid] = missed
n_v1_misses = sum(len(tids) for tids in gt_v1_misses.values())
print(f"Baseline V1 missed {n_v1_misses:,} true pairs ({n_v1_misses/total_true_matches*100:.2f}% of all matches).")

# 3. Define the sequential experimental configurations
configs = [
    (
        "V1: Baseline V1",
        dict(
            max_candidates_per_entity=50,
            enable_name_sorted_tokens=False,
            enable_indic_transliteration=False,
            enable_condensed_name=False,
            enable_address_multi=False,
            enable_rare_tokens=False,
            enable_char_ngrams=False,
            enable_adaptive_fallback=False,
        ),
    ),
    (
        "V2.1: + Sorted-Tokens",
        dict(
            max_candidates_per_entity=50,
            enable_name_sorted_tokens=True,
            enable_indic_transliteration=False,
            enable_condensed_name=False,
            enable_address_multi=False,
            enable_rare_tokens=False,
            enable_char_ngrams=False,
            enable_adaptive_fallback=False,
        ),
    ),
    (
        "V2.2: + Indic Transliteration",
        dict(
            max_candidates_per_entity=50,
            enable_name_sorted_tokens=True,
            enable_indic_transliteration=True,
            enable_condensed_name=False,
            enable_address_multi=False,
            enable_rare_tokens=False,
            enable_char_ngrams=False,
            enable_adaptive_fallback=False,
        ),
    ),
    (
        "V2.3: + Stronger Address Multi",
        dict(
            max_candidates_per_entity=100,
            enable_name_sorted_tokens=True,
            enable_indic_transliteration=True,
            enable_condensed_name=False,
            enable_address_multi=True,
            enable_rare_tokens=False,
            enable_char_ngrams=False,
            enable_adaptive_fallback=False,
        ),
    ),
    (
        "V2.4: + Condensed & Domain Normalization",
        dict(
            max_candidates_per_entity=100,
            enable_name_sorted_tokens=True,
            enable_indic_transliteration=True,
            enable_condensed_name=True,
            enable_address_multi=True,
            enable_rare_tokens=False,
            enable_char_ngrams=False,
            enable_adaptive_fallback=False,
        ),
    ),
    (
        "V2.5: + Rare-Token & N-Grams",
        dict(
            max_candidates_per_entity=100,
            enable_name_sorted_tokens=True,
            enable_indic_transliteration=True,
            enable_condensed_name=True,
            enable_address_multi=True,
            enable_rare_tokens=True,
            enable_char_ngrams=True,
            enable_adaptive_fallback=False,
        ),
    ),
    (
        "V2.6: + Adaptive Fallback (Cap=150)",
        dict(
            max_candidates_per_entity=150,
            enable_name_sorted_tokens=True,
            enable_indic_transliteration=True,
            enable_condensed_name=True,
            enable_address_multi=True,
            enable_rare_tokens=True,
            enable_char_ngrams=True,
            enable_adaptive_fallback=True,
        ),
    ),
    (
        "V2.7: + Full Unconstrained Frontier",
        dict(
            max_candidates_per_entity=None,
            enable_name_sorted_tokens=True,
            enable_indic_transliteration=True,
            enable_condensed_name=True,
            enable_address_multi=True,
            enable_rare_tokens=True,
            enable_char_ngrams=True,
            enable_adaptive_fallback=True,
        ),
    ),
]

# Helper to compute slice recall
def compute_slice_recall(cands, gt_slice):
    total = sum(len(tids) for tids in gt_slice.values())
    if total == 0:
        return 1.0
    hits = sum(len(tids & cands.get(sid, set())) for sid, tids in gt_slice.items())
    return hits / total

results = []

print("\n" + "=" * 115)
print(f"{'VERSION':35s} | {'RECALL':8s} | {'TRANSLIT':9s} | {'V1-REC':8s} | {'US-REC':8s} | {'IN-REC':8s} | {'AVG-C':6s} | {'P95':5s} | {'P99':5s} | {'TIME':6s} | {'RAM':6s}")
print("=" * 115)

for name, flags in configs:
    tracemalloc.start()
    t0 = time.time()

    blocker = MultiPassBlocker(**flags)
    blocker.fit_targets(target_records)
    cands = blocker.block_all(s1_records)

    elapsed = time.time() - t0
    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_ram / (1024 * 1024)

    # Metrics
    overall_recall = compute_slice_recall(cands, ground_truth)
    translit_recall = compute_slice_recall(cands, gt_translit) if n_translit_pairs > 0 else 1.0
    v1_recovery = compute_slice_recall(cands, gt_v1_misses) if n_v1_misses > 0 else 1.0
    us_recall = compute_slice_recall(cands, gt_by_country.get("US", {}))
    in_recall = compute_slice_recall(cands, gt_by_country.get("India", {}))

    counts = [len(cands.get(s["entity_id"], set())) for s in s1_records]
    avg_c = np.mean(counts)
    p95_c = np.percentile(counts, 95)
    p99_c = np.percentile(counts, 99)

    res = {
        "version": name,
        "recall": overall_recall,
        "translit_recall": translit_recall,
        "v1_recovery": v1_recovery,
        "us_recall": us_recall,
        "in_recall": in_recall,
        "avg_cands": avg_c,
        "p95_cands": p95_c,
        "p99_cands": p99_c,
        "time_sec": elapsed,
        "ram_mb": peak_mb,
    }
    results.append(res)

    print(f"{name:35s} | {overall_recall*100:7.2f}% | {translit_recall*100:8.2f}% | {v1_recovery*100:7.2f}% | {us_recall*100:7.2f}% | {in_recall*100:7.2f}% | {avg_c:6.1f} | {p95_c:5.0f} | {p99_c:5.0f} | {elapsed:5.1f}s | {peak_mb:5.1f}M")

print("=" * 115)
