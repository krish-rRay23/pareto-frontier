import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import re
from collections import defaultdict
import numpy as np

print("=== FAST EMPIRICAL BLOCKING BENCHMARK (USING CSV STREAMING) ===")

# 1. Load 1000 S1 records
s1_dict = {}
with open('resources/student_resource/dataset/train/train_source1.tsv', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, row in enumerate(reader):
        if i >= 1000:
            break
        s1_dict[row['entity_id']] = row

s1_ids = set(s1_dict.keys())

# 2. Load ground truth for these 1000 S1
gt_map = {}
with open('resources/student_resource/dataset/train/train_ground_truth.tsv', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        sid = row['source1_entity_id']
        if sid in s1_ids:
            raw = row['matched_entity_ids'].strip()
            mids = [x.strip() for x in raw.split(',') if x.strip()] if raw else []
            gt_map[sid] = mids
            if len(gt_map) >= len(s1_ids):
                break

all_gt_target_ids = set()
for mids in gt_map.values():
    all_gt_target_ids.update(mids)

print(f"Loaded {len(s1_dict)} S1 entities. True Target IDs in GT: {len(all_gt_target_ids)}")

# 3. Load all true targets + 150,000 background records from S2 and S3 using fast csv.reader
targets = {}
bg_loaded = 0
MAX_BG = 150000

for filename in ['train_source2.tsv', 'train_source3.tsv']:
    path = f'resources/student_resource/dataset/train/{filename}'
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            eid = row['entity_id']
            if eid in all_gt_target_ids:
                targets[eid] = row
            elif bg_loaded < MAX_BG:
                targets[eid] = row
                bg_loaded += 1

print(f"Search Pool Size: {len(targets)} records ({len(all_gt_target_ids)} GT matches + {bg_loaded} background noise).")

# 4. Normalization utilities
LEGAL_TERMS = {
    'llc', 'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'pllc', 'pc', 'co', 'company', 'sarl', 'sasu', 'sas', 'eurl', 'sa', 'gmbh', 'dba',
    'services', 'service', 'center', 'group', 'holdings', 'enterprises'
}

def clean_text(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'[\W_]+', ' ', s)
    return s.strip()

def strip_legal(s):
    toks = clean_text(s).split()
    f = [t for t in toks if t not in LEGAL_TERMS]
    return " ".join(f) if f else " ".join(toks)

def extract_numbers(s):
    if not s:
        return []
    return re.findall(r'\b\d{1,6}\b', s)

# 5. Build Blockers
print("\nBuilding Inverted Indices...")

# Blocker 1: Exact Clean Name (Country Isolated)
idx_b1 = defaultdict(list)
# Blocker 2: Legal-Stripped Name (Country Isolated)
idx_b2 = defaultdict(list)
# Blocker 3: First 2 tokens of Legal-Stripped Name
idx_b3 = defaultdict(list)
# Blocker 4: House Number + First Name Token
idx_b4 = defaultdict(list)
# Blocker 5: Rare Token Index (Tokens with length >= 4)
idx_b5 = defaultdict(list)

for eid, r in targets.items():
    c = r['country']
    bname = r['business_name']
    baddr = r['business_address']
    
    c_name = clean_text(bname)
    s_name = strip_legal(bname)
    nums = extract_numbers(baddr)
    toks = s_name.split()
    
    idx_b1[(c, c_name)].append(eid)
    idx_b2[(c, s_name)].append(eid)
    
    if len(toks) >= 2:
        idx_b3[(c, toks[0], toks[1])].append(eid)
        
    if nums and toks:
        idx_b4[(c, nums[0], toks[0][:4])].append(eid)
        
    for t in set(toks):
        if len(t) >= 4 and t not in LEGAL_TERMS:
            idx_b5[(c, t)].append(eid)

print("Indices built successfully. Running candidate queries...")

# Query blockers for each S1
cand_b1 = defaultdict(set)
cand_b2 = defaultdict(set)
cand_b3 = defaultdict(set)
cand_b4 = defaultdict(set)
cand_b5 = defaultdict(set)

for s1_id, r in s1_dict.items():
    c = r['country']
    bname = r['business_name']
    baddr = r['business_address']
    
    c_name = clean_text(bname)
    s_name = strip_legal(bname)
    nums = extract_numbers(baddr)
    toks = s_name.split()
    
    cand_b1[s1_id].update(idx_b1.get((c, c_name), []))
    cand_b2[s1_id].update(idx_b2.get((c, s_name), []))
    
    if len(toks) >= 2:
        cand_b3[s1_id].update(idx_b3.get((c, toks[0], toks[1]), []))
        
    if nums and toks:
        cand_b4[s1_id].update(idx_b4.get((c, nums[0], toks[0][:4]), []))
        
    # For rare tokens, only take tokens appearing in <= 100 targets to prevent blowing up candidate size
    for t in set(toks):
        if len(t) >= 4 and t not in LEGAL_TERMS:
            matches = idx_b5.get((c, t), [])
            if len(matches) <= 80:  # IDF filter
                cand_b5[s1_id].update(matches)

# Evaluate each blocker
def evaluate(cand_dict, name):
    hits = 0
    total_true = len(all_gt_target_ids)
    c_counts = []
    for s1_id, true_mids in gt_map.items():
        found = cand_dict.get(s1_id, set())
        c_counts.append(len(found))
        hits += len(set(true_mids) & found)
    rec = hits / total_true if total_true > 0 else 0
    avg_c = np.mean(c_counts)
    p95_c = np.percentile(c_counts, 95)
    print(f"{name:38s} | Recall: {hits:4d}/{total_true:4d} ({rec*100:6.2f}%) | Avg Cand: {avg_c:6.2f} | P95: {p95_c:5.0f}")
    return rec, avg_c

print("\n" + "="*85)
print(f"{'BLOCKER STRATEGY':38s} | {'TRUE RECALL':20s} | {'AVG CANDS':10s} | {'P95 CANDS'}")
print("="*85)
evaluate(cand_b1, "1. Exact Clean Name")
evaluate(cand_b2, "2. Legal-Stripped Name")
evaluate(cand_b3, "3. First 2 Tokens")
evaluate(cand_b4, "4. House Num + First Token Prefix")
evaluate(cand_b5, "5. Selective Rare Token Index (IDF<=80)")

# Multi-pass Unions
u12 = defaultdict(set)
u123 = defaultdict(set)
u1234 = defaultdict(set)
u_all = defaultdict(set)

for sid in s1_dict:
    u12[sid] = cand_b1[sid] | cand_b2[sid]
    u123[sid] = u12[sid] | cand_b3[sid]
    u1234[sid] = u123[sid] | cand_b4[sid]
    u_all[sid] = u1234[sid] | cand_b5[sid]

print("-"*85)
evaluate(u12, "Union (1 + 2)")
evaluate(u123, "Union (1 + 2 + 3)")
evaluate(u1234, "Union (1 + 2 + 3 + 4)")
evaluate(u_all, "FULL UNION (1 + 2 + 3 + 4 + 5)")
print("="*85)

# Inspect missed matches from full union
missed_records = []
for sid, true_mids in gt_map.items():
    found = u_all.get(sid, set())
    missed = set(true_mids) - found
    for m in missed:
        if m in targets:
            missed_records.append((s1_dict[sid], targets[m]))

print(f"\nRemaining Missed Matches in Full Union: {len(missed_records)}/{len(all_gt_target_ids)} ({len(missed_records)/len(all_gt_target_ids)*100:.2f}%)")
if missed_records:
    print("Examples of remaining misses:")
    for i, (s1_r, tgt_r) in enumerate(missed_records[:6]):
        print(f"[{i+1}] S1:   \"{s1_r['business_name']}\" | \"{s1_r['business_address']}\"")
        print(f"    TGT:  \"{tgt_r['business_name']}\" | \"{tgt_r['business_address']}\"")
