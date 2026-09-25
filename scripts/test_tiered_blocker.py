"""Test Tiered Candidate Retrieval with All Core Passes Restored."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.abspath("."))
import re
from collections import defaultdict, Counter
import numpy as np

from src.data.loader import load_benchmark_subset
from src.normalization.address_normalizer import get_multi_view_address
from src.normalization.text_normalizer import get_multi_view_name

s1_records, target_records, ground_truth = load_benchmark_subset(
    data_dir="resources/student_resource/dataset/train",
    n_s1=2000,
    background_noise_ratio=25,
)

print(f"Loaded {len(s1_records)} S1s and {len(target_records)} targets...")

def get_condensed_name(raw):
    s = re.sub(r"\.(com|org|net|in|co|io|biz|info|gov)\b", "", str(raw).lower())
    s = re.sub(r"^[@#*]+", "", s)
    return re.sub(r"[\W_]+", "", s)

def extract_all_digits(text):
    return [n.lstrip("0") or "0" for n in re.findall(r"\d{1,7}", str(text))]

idx_tier1 = defaultdict(list)
idx_tier2 = defaultdict(list)
idx_tier3 = defaultdict(list)

token_counts = Counter()
for r in target_records:
    c = r.get("country", "")
    nv = get_multi_view_name(r.get("business_name", ""))
    for t in nv.get("tokens", []):
        if len(t) >= 4:
            token_counts[(c, t)] += 1

for r in target_records:
    eid = r["entity_id"]
    c = r.get("country", "")
    bname = r.get("business_name", "")
    baddr = r.get("business_address", "")
    nv = get_multi_view_name(bname)
    av = get_multi_view_address(baddr)
    cond = get_condensed_name(bname)

    # Tier 1: Names
    if nv["clean"]:
        idx_tier1[(c, "clean", nv["clean"])].append(eid)
    if nv["legal_stripped"]:
        idx_tier1[(c, "legal", nv["legal_stripped"])].append(eid)
    if nv["first_two"]:
        idx_tier1[(c, "first_two", nv["first_two"])].append(eid)
    if nv.get("sorted_tokens"):
        idx_tier1[(c, "sorted", nv["sorted_tokens"])].append(eid)
    if cond and len(cond) >= 5:
        idx_tier1[(c, "condensed", cond)].append(eid)
    if nv.get("is_indic") and nv.get("romanized"):
        idx_tier1[(c, "legal", nv["romanized"])].append(eid)
        rom_toks = nv.get("romanized_tokens", [])
        if len(rom_toks) >= 2:
            idx_tier1[(c, "first_two", f"{rom_toks[0]} {rom_toks[1]}")].append(eid)
        rom_s = "_".join(sorted([t for t in rom_toks if len(t) >= 3][:4]))
        if rom_s:
            idx_tier1[(c, "sorted", rom_s)].append(eid)

    # Tier 2: Address Number + ALL Salient Tokens & Postal Codes & Rare Tokens
    nums = extract_all_digits(baddr)
    sals = av.get("salient_tokens", [])
    pins = av.get("postal_codes", [])
    for n in nums[:4]:
        for s in sals[:12]:
            idx_tier2[(c, "num_sal", n, s)].append(eid)
    for p in pins[:2]:
        for n in nums[:2]:
            idx_tier2[(c, "pin_num", p, n)].append(eid)
        for s in sals[:4]:
            idx_tier2[(c, "pin_sal", p, s)].append(eid)

    for t in nv.get("tokens", []):
        if len(t) >= 4 and token_counts.get((c, t), 0) <= 40:
            idx_tier2[(c, "rare", t)].append(eid)

    # Tier 3: Salient pairs
    for i in range(min(5, len(sals))):
        for j in range(i + 1, min(5, len(sals))):
            idx_tier3[(c, "sal_pair", min(sals[i], sals[j]), max(sals[i], sals[j]))].append(eid)

total = sum(len(tids) for tids in ground_truth.values())

for max_cap in [100, 150, 200, None]:
    hits = 0
    cand_counts = []
    for s1 in s1_records:
        sid = s1["entity_id"]
        c = s1.get("country", "")
        bname = s1.get("business_name", "")
        baddr = s1.get("business_address", "")
        nv = get_multi_view_name(bname)
        av = get_multi_view_address(baddr)
        cond = get_condensed_name(bname)

        t1, t2, t3 = set(), set(), set()
        def _add(idx, key, target_set, cap=150):
            m = idx.get(key, [])
            if 0 < len(m) <= cap:
                target_set.update(m)

        if nv["clean"]:
            _add(idx_tier1, (c, "clean", nv["clean"]), t1)
        if nv["legal_stripped"]:
            _add(idx_tier1, (c, "legal", nv["legal_stripped"]), t1)
        if nv["first_two"]:
            _add(idx_tier1, (c, "first_two", nv["first_two"]), t1)
        if nv.get("sorted_tokens"):
            _add(idx_tier1, (c, "sorted", nv["sorted_tokens"]), t1)
        if cond and len(cond) >= 5:
            _add(idx_tier1, (c, "condensed", cond), t1)

        nums = extract_all_digits(baddr)
        sals = av.get("salient_tokens", [])
        pins = av.get("postal_codes", [])
        for n in nums[:4]:
            for s in sals[:12]:
                _add(idx_tier2, (c, "num_sal", n, s), t2, cap=100)
        for p in pins[:2]:
            for n in nums[:2]:
                _add(idx_tier2, (c, "pin_num", p, n), t2, cap=80)
            for s in sals[:4]:
                _add(idx_tier2, (c, "pin_sal", p, s), t2, cap=60)

        for t in nv.get("tokens", []):
            if len(t) >= 4:
                _add(idx_tier2, (c, "rare", t), t2, cap=40)

        for i in range(min(5, len(sals))):
            for j in range(i + 1, min(5, len(sals))):
                _add(idx_tier3, (c, "sal_pair", min(sals[i], sals[j]), max(sals[i], sals[j])), t3, cap=50)

        if len(t1) + len(t2) <= 2:
            for s in sals[:2]:
                if len(s) >= 5:
                    _add(idx_tier3, (c, "rare_sal", s), t3, cap=35)

        cands = list(t1)
        if max_cap is None:
            cands.extend([x for x in t2 if x not in t1])
            rem_set = set(cands)
            cands.extend([x for x in t3 if x not in rem_set])
        else:
            if len(cands) < max_cap:
                cands.extend([x for x in sorted(t2) if x not in t1][: max_cap - len(cands)])
            if len(cands) < max_cap:
                rem_set = set(cands)
                cands.extend([x for x in sorted(t3) if x not in rem_set][: max_cap - len(cands)])

        true_set = ground_truth.get(sid, set())
        hits += len(true_set & set(cands))
        cand_counts.append(len(cands))

    print(f"MaxCap={str(max_cap):5s} -> Recall: {hits/total*100:.2f}% | Avg: {np.mean(cand_counts):5.1f} | P95: {np.percentile(cand_counts, 95):5.1f} | P99: {np.percentile(cand_counts, 99):5.1f}")
