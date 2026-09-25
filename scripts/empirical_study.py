import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

print("=== STARTING EMPIRICAL ANALYSIS ON REAL DATA ===")

# 1. Load a solid sample of S1 and ground truth
N_SAMPLE = 5000
s1_df = pd.read_csv('resources/student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=N_SAMPLE)
s1_dict = {r['entity_id']: r for _, r in s1_df.iterrows()}
s1_ids = set(s1_dict.keys())

# Load ground truth for these S1 IDs
gt_map = {}
for chunk in pd.read_csv('resources/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', chunksize=200000):
    sub = chunk[chunk['source1_entity_id'].isin(s1_ids)]
    for _, r in sub.iterrows():
        mids = [x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip() and pd.notna(r['matched_entity_ids'])]
        gt_map[r['source1_entity_id']] = mids
    if len(gt_map) >= len(s1_ids):
        break

print(f"Loaded {len(s1_dict)} S1 entities and found {len(gt_map)} in ground truth.")

# Analyze match breakdown: S2 vs S3
s2_counts = []
s3_counts = []
singletons = 0
total_matches = 0
both_s2_s3 = 0
only_s2 = 0
only_s3 = 0

all_target_ids = set()
for s1_id, mids in gt_map.items():
    if not mids:
        singletons += 1
        continue
    total_matches += len(mids)
    all_target_ids.update(mids)
    s2_m = [m for m in mids if m.startswith('S2-')]
    s3_m = [m for m in mids if m.startswith('S3-')]
    s2_counts.append(len(s2_m))
    s3_counts.append(len(s3_m))
    if s2_m and s3_m:
        both_s2_s3 += 1
    elif s2_m:
        only_s2 += 1
    elif s3_m:
        only_s3 += 1

n_entities = len(gt_map)
print(f"\n--- Ground Truth Structure (N={n_entities}) ---")
print(f"Singletons: {singletons} ({singletons/n_entities*100:.2f}%)")
print(f"Entities with matches: {n_entities - singletons} ({(n_entities - singletons)/n_entities*100:.2f}%)")
print(f"  - Matches in BOTH S2 and S3: {both_s2_s3} ({both_s2_s3/(n_entities - singletons)*100:.2f}% of matched)")
print(f"  - Matches in S2 ONLY: {only_s2} ({only_s2/(n_entities - singletons)*100:.2f}% of matched)")
print(f"  - Matches in S3 ONLY: {only_s3} ({only_s3/(n_entities - singletons)*100:.2f}% of matched)")
print(f"Mean S2 matches per matched S1: {np.mean(s2_counts):.2f}")
print(f"Mean S3 matches per matched S1: {np.mean(s3_counts):.2f}")
print(f"Total target IDs across sample: {len(all_target_ids)}")

print("\n--- Empirical Analysis Complete Step 1 ---")
