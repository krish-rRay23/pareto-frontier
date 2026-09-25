import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# Load first 2000 rows of S1
s1 = pd.read_csv('resources/student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=2000)
s1_map = {r['entity_id']: r for _, r in s1.iterrows()}

# Find GT for these S1 IDs
gt_map = {}
for chunk in pd.read_csv('resources/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', chunksize=200000):
    subset = chunk[chunk['source1_entity_id'].isin(s1_map)]
    for _, r in subset.iterrows():
        if pd.notna(r['matched_entity_ids']) and str(r['matched_entity_ids']).strip():
            gt_map[r['source1_entity_id']] = [x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()]
    if len(gt_map) >= 50:
        break

target_ids = set()
for ids in gt_map.values():
    target_ids.update(ids)

print(f"S1 count: {len(gt_map)}, Target IDs: {len(target_ids)}")

targets = {}
for chunk in pd.read_csv('resources/student_resource/dataset/train/train_source2.tsv', sep='\t', chunksize=200000):
    subset = chunk[chunk['entity_id'].isin(target_ids)]
    for _, r in subset.iterrows():
        targets[r['entity_id']] = r.to_dict()
    if len(targets) >= len(target_ids):
        break

for chunk in pd.read_csv('resources/student_resource/dataset/train/train_source3.tsv', sep='\t', chunksize=200000):
    subset = chunk[chunk['entity_id'].isin(target_ids)]
    for _, r in subset.iterrows():
        targets[r['entity_id']] = r.to_dict()
    if len(targets) >= len(target_ids):
        break

count = 0
for s1_id, match_list in gt_map.items():
    found_targets = [targets[mid] for mid in match_list if mid in targets]
    if found_targets and count < 8:
        count += 1
        s1_rec = s1_map[s1_id]
        print(f"=== MATCH EXAMPLE {count} | Country: {s1_rec['country']} ===")
        print(f"  S1:    [{s1_rec['entity_id']}]")
        print(f"         Name: \"{s1_rec['business_name']}\"")
        print(f"         Addr: \"{s1_rec['business_address']}\"")
        for t in found_targets:
            print(f"  MATCH: [{t['entity_id']}]")
            print(f"         Name: \"{t['business_name']}\"")
            print(f"         Addr: \"{t['business_address']}\"")
        print()
