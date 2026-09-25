import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import re
from collections import defaultdict
import numpy as np

# Load 1000 S1 records
s1_dict = {}
with open('resources/student_resource/dataset/train/train_source1.tsv', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, row in enumerate(reader):
        if i >= 1000: break
        s1_dict[row['entity_id']] = row

s1_ids = set(s1_dict.keys())

# Ground truth
gt_map = {}
with open('resources/student_resource/dataset/train/train_ground_truth.tsv', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        sid = row['source1_entity_id']
        if sid in s1_ids:
            raw = row['matched_entity_ids'].strip()
            mids = [x.strip() for x in raw.split(',') if x.strip()] if raw else []
            gt_map[sid] = mids
            if len(gt_map) >= len(s1_ids): break

all_gt_target_ids = set()
for mids in gt_map.values():
    all_gt_target_ids.update(mids)

# Load targets + background
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

LEGAL_TERMS = {
    'llc', 'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'pllc', 'pc', 'co', 'company', 'sarl', 'sasu', 'sas', 'eurl', 'sa', 'gmbh', 'dba',
    'services', 'service', 'center', 'group', 'holdings', 'enterprises'
}

ADDR_STOPWORDS = {
    'st', 'street', 'rd', 'road', 'dr', 'drive', 'ave', 'avenue', 'ln', 'lane', 'blvd', 'boulevard',
    'ct', 'court', 'cir', 'circle', 'hwy', 'highway', 'fl', 'floor', 'ste', 'suite', 'apt', 'apartment',
    'unit', 'po', 'box', 'near', 'opp', 'opposite', 'behind', 'beside', 'block', 'sector', 'plot', 'h', 'no'
}

def clean_text(s):
    if not s: return ""
    return re.sub(r'[\W_]+', ' ', s.lower()).strip()

def strip_legal(s):
    toks = clean_text(s).split()
    f = [t for t in toks if t not in LEGAL_TERMS]
    return " ".join(f) if f else " ".join(toks)

def extract_numbers(s):
    if not s: return []
    return re.findall(r'\b\d{1,6}\b', s)

def extract_addr_salient_tokens(addr):
    toks = clean_text(addr).split()
    salient = [t for t in toks if len(t) >= 4 and t not in ADDR_STOPWORDS and not t.isdigit()]
    return salient

# Build Blocker: Address Number + Salient Address Token
print("Building Address Number + Salient Address Token Index...")
idx_addr_num_tok = defaultdict(list)
# Build Blocker: 2 Salient Address Tokens (for address without numbers)
idx_addr_pair = defaultdict(list)
# Build Blocker: 3-gram char prefix of name (for typo resilience: Netw0rk -> netw)
idx_name_prefix = defaultdict(list)

for eid, r in targets.items():
    c = r['country']
    nums = extract_numbers(r['business_address'])
    addr_toks = extract_addr_salient_tokens(r['business_address'])
    s_name = strip_legal(r['business_name'])
    name_toks = s_name.split()
    
    if nums and addr_toks:
        for t in addr_toks[:3]:
            idx_addr_num_tok[(c, nums[0], t)].append(eid)
            
    if len(addr_toks) >= 2:
        idx_addr_pair[(c, addr_toks[0], addr_toks[1])].append(eid)
        
    if name_toks and len(name_toks[0]) >= 4:
        idx_name_prefix[(c, name_toks[0][:4])].append(eid)

print("Querying address-based blockers...")
cand_addr_num = defaultdict(set)
cand_addr_pair = defaultdict(set)
cand_name_pref_num = defaultdict(set)

for s1_id, r in s1_dict.items():
    c = r['country']
    nums = extract_numbers(r['business_address'])
    addr_toks = extract_addr_salient_tokens(r['business_address'])
    s_name = strip_legal(r['business_name'])
    name_toks = s_name.split()
    
    if nums and addr_toks:
        for t in addr_toks[:3]:
            matches = idx_addr_num_tok.get((c, nums[0], t), [])
            if len(matches) <= 50:
                cand_addr_num[s1_id].update(matches)
                
    if len(addr_toks) >= 2:
        matches = idx_addr_pair.get((c, addr_toks[0], addr_toks[1]), [])
        if len(matches) <= 30:
            cand_addr_pair[s1_id].update(matches)
            
    if nums and name_toks and len(name_toks[0]) >= 4:
        matches = idx_name_prefix.get((c, name_toks[0][:4]), [])
        # Only take if house number also matches
        for m in matches[:100]:
            t_nums = extract_numbers(targets[m]['business_address'])
            if t_nums and t_nums[0] == nums[0]:
                cand_name_pref_num[s1_id].add(m)

# Load previous candidates from Blocker 1,2,3,4,5
# Let's import or re-evaluate union with address blockers
idx_b1 = defaultdict(list)
idx_b2 = defaultdict(list)
idx_b3 = defaultdict(list)

for eid, r in targets.items():
    c = r['country']
    c_name = clean_text(r['business_name'])
    s_name = strip_legal(r['business_name'])
    toks = s_name.split()
    idx_b1[(c, c_name)].append(eid)
    idx_b2[(c, s_name)].append(eid)
    if len(toks) >= 2:
        idx_b3[(c, toks[0], toks[1])].append(eid)

cand_u_prev = defaultdict(set)
for s1_id, r in s1_dict.items():
    c = r['country']
    c_name = clean_text(r['business_name'])
    s_name = strip_legal(r['business_name'])
    toks = s_name.split()
    cand_u_prev[s1_id].update(idx_b1.get((c, c_name), []))
    cand_u_prev[s1_id].update(idx_b2.get((c, s_name), []))
    if len(toks) >= 2:
        cand_u_prev[s1_id].update(idx_b3.get((c, toks[0], toks[1]), []))

def evaluate(cand_dict, name):
    hits = 0
    total_true = len(all_gt_target_ids)
    c_counts = [len(cand_dict.get(sid, set())) for sid in gt_map]
    for sid, true_mids in gt_map.items():
        hits += len(set(true_mids) & cand_dict.get(sid, set()))
    rec = hits / total_true if total_true > 0 else 0
    avg_c = np.mean(c_counts)
    p95_c = np.percentile(c_counts, 95)
    print(f"{name:45s} | Recall: {hits:4d}/{total_true:4d} ({rec*100:6.2f}%) | Avg Cand: {avg_c:6.2f} | P95: {p95_c:5.0f}")

print("\n" + "="*95)
evaluate(cand_u_prev, "Previous Name Union (Clean + Legal + 2-Tok)")
evaluate(cand_addr_num, "New Blocker: House Num + Salient Addr Tok")
evaluate(cand_addr_pair, "New Blocker: 2 Salient Addr Tokens")
evaluate(cand_name_pref_num, "New Blocker: Name Prefix 4-char + House Num")

# Grand Union
grand_union = defaultdict(set)
for sid in s1_dict:
    grand_union[sid] = cand_u_prev[sid] | cand_addr_num[sid] | cand_addr_pair[sid] | cand_name_pref_num[sid]

print("-"*95)
evaluate(grand_union, "GRAND HYBRID UNION (Name + Address Blockers)")
print("="*95)
