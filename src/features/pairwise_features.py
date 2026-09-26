"""High-performance comprehensive pairwise feature engineering for entity resolution.

Implements the ~80-100 complementary pairwise feature families specified in
Amazon ML Challenge 2026 99.99+ Architecture:
1. Name Evidence (Exacts, Levenshtein, Jaro-Winkler, N-Grams, Tokens, Acronym, Transliteration)
2. Address Evidence (Exacts, Edit distances, Tokens, Postal, House number, City/State/Locality)
3. Numeric Evidence (Intersection, Sequence match, Postal agreement, Conflicting numbers)
4. Retrieval Provenance Evidence (Channel flags, Agreement count, Ranks, Scores, Gaps)
5. Cross-Field Interactions (Name x Address, Harmonic mean, Min/Max evidence)
"""

import difflib
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.blocking.retrieval import ALL_CHANNELS, RetrievalProvenance
from src.normalization.address_normalizer import get_multi_view_address
from src.normalization.text_normalizer import get_multi_view_name


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Fast Jaro-Winkler string similarity implementation."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return float(jaro + prefix * p * (1.0 - jaro))


def sequence_match_ratio(str_a: str, str_b: str) -> float:
    """Compute normalized sequence similarity ratio using difflib."""
    if not str_a and not str_b:
        return 1.0
    if not str_a or not str_b:
        return 0.0
    if str_a == str_b:
        return 1.0
    return float(difflib.SequenceMatcher(None, str_a, str_b).ratio())


def jaccard_similarity(set_a: Any, set_b: Any) -> float:
    """Compute Jaccard similarity between two sets or iterables."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    s_a = set(set_a)
    s_b = set(set_b)
    intersection = len(s_a & s_b)
    union = len(s_a | s_b)
    return float(intersection / union) if union > 0 else 0.0


def containment_ratio(set_a: Any, set_b: Any) -> float:
    """Compute containment: |A & B| / min(|A|, |B|)."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    s_a = set(set_a)
    s_b = set(set_b)
    min_size = min(len(s_a), len(s_b))
    return float(len(s_a & s_b) / min_size) if min_size > 0 else 0.0


def token_sort_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute similarity of alphabetically sorted token strings."""
    sorted_a = " ".join(sorted(tokens_a))
    sorted_b = " ".join(sorted(tokens_b))
    return sequence_match_ratio(sorted_a, sorted_b)


def token_set_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute token set similarity comparing common vs remainder tokens."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    if not intersection:
        return sequence_match_ratio(" ".join(tokens_a), " ".join(tokens_b))

    sorted_inter = " ".join(sorted(intersection))
    sorted_rem_a = " ".join(sorted(set_a - intersection))
    sorted_rem_b = " ".join(sorted(set_b - intersection))

    sim1 = sequence_match_ratio(sorted_inter, f"{sorted_inter} {sorted_rem_a}".strip())
    sim2 = sequence_match_ratio(sorted_inter, f"{sorted_inter} {sorted_rem_b}".strip())
    sim3 = sequence_match_ratio(f"{sorted_inter} {sorted_rem_a}".strip(), f"{sorted_inter} {sorted_rem_b}".strip())
    return float(max(sim1, sim2, sim3))


def precompute_record_views(record: Dict[str, Any]) -> Dict[str, Any]:
    """Precompute normalized name and address multi-views for a record."""
    name_view = get_multi_view_name(record.get("business_name", ""))
    addr_view = get_multi_view_address(record.get("business_address", ""))
    country = str(record.get("country", "")).strip().lower()
    return {
        "name": name_view,
        "addr": addr_view,
        "country": country,
        "name_tokens": name_view["tokens"],
    }


FEATURE_COLUMNS: List[str] = [
    # 1. Name Family (24 features)
    "exact_raw_name",
    "exact_clean_name",
    "exact_legal_name",
    "exact_compact_name",
    "name_seq_ratio",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_containment",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_prefix_match_4",
    "name_suffix_match_4",
    "name_first_token_match",
    "name_last_token_match",
    "name_acronym_match",
    "name_char_len_diff",
    "name_char_len_ratio",
    "name_token_count_diff",
    "name_transliteration_sim",
    "name_missing_indicator",
    "name_num_token_match",

    # 2. Address Family (23 features)
    "exact_raw_addr",
    "exact_clean_addr",
    "addr_seq_ratio",
    "addr_jaro_winkler",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_salient_jaccard",
    "addr_token_set_ratio",
    "addr_char_3gram_jaccard",
    "postal_exact_match",
    "postal_prefix_match",
    "house_num_exact_match",
    "house_num_abs_diff",
    "house_num_rel_diff",
    "house_num_digit_sim",
    "city_similarity",
    "state_similarity",
    "locality_similarity",
    "addr_comp_agree_count",
    "addr_comp_conflict_count",
    "addr_char_len_diff",
    "addr_missing_indicator",
    "addr_number_jaccard",

    # 3. Numeric Evidence Family (8 features)
    "num_set_intersection_count",
    "num_set_jaccard",
    "exact_num_seq_match",
    "house_num_agreement",
    "conflicting_num_count",
    "postal_agreement",
    "numeric_only_match_flag",
    "numeric_conflict_flag",

    # 4. Retrieval Provenance Family (20 features)
    "retrieved_by_exact_name",
    "retrieved_by_exact_addr",
    "retrieved_by_exact_both",
    "retrieved_by_rare_token",
    "retrieved_by_bm25_name",
    "retrieved_by_bm25_addr",
    "retrieved_by_numeric_loc",
    "retrieved_by_cross_script",
    "retrieved_by_missing_field",
    "retrieved_by_sibling",
    "retrieved_by_reverse",
    "retrieval_agreement_count",
    "retrieval_best_rank",
    "retrieval_second_best_rank",
    "retrieval_rank_gap",
    "retrieval_best_score",
    "retrieval_second_best_score",
    "retrieval_score_gap",
    "retrieval_reciprocal_rank",
    "forward_reverse_agreement",

    # 5. Cross-Field Interaction Family (11 features)
    "name_x_addr_sim",
    "name_addr_harmonic_mean",
    "min_name_addr_evidence",
    "max_name_addr_evidence",
    "name_addr_evidence_diff",
    "exact_name_partial_addr",
    "exact_num_partial_name",
    "name_x_num_match",
    "country_match",
    "is_s2",
    "is_s3",
]


def extract_pair_features(
    s1_record: Dict[str, Any],
    target_record: Dict[str, Any],
    s1_precomputed: Dict[str, Any],
    target_precomputed: Dict[str, Any],
    provenance: Optional[RetrievalProvenance] = None,
) -> List[float]:
    """Compute full 86-feature vector for a single (S1, Target) candidate pair."""
    s1_nv = s1_precomputed["name"]
    tgt_nv = target_precomputed["name"]
    s1_av = s1_precomputed["addr"]
    tgt_av = target_precomputed["addr"]

    # ----------------------------------------------------
    # 1. NAME FEATURES
    # ----------------------------------------------------
    s1_raw_name = s1_nv["raw"]
    tgt_raw_name = tgt_nv["raw"]
    s1_clean_name = s1_nv["clean"]
    tgt_clean_name = tgt_nv["clean"]
    s1_legal_name = s1_nv["legal_stripped"]
    tgt_legal_name = tgt_nv["legal_stripped"]
    s1_compact_name = s1_nv["compact"]
    tgt_compact_name = tgt_nv["compact"]

    exact_raw_name = 1.0 if (s1_raw_name and s1_raw_name == tgt_raw_name) else 0.0
    exact_clean_name = 1.0 if (s1_clean_name and s1_clean_name == tgt_clean_name) else 0.0
    exact_legal_name = 1.0 if (s1_legal_name and s1_legal_name == tgt_legal_name) else 0.0
    exact_compact_name = 1.0 if (s1_compact_name and s1_compact_name == tgt_compact_name) else 0.0

    name_seq = sequence_match_ratio(s1_clean_name, tgt_clean_name)
    name_jw = jaro_winkler_similarity(s1_clean_name, tgt_clean_name)

    s1_name_toks = s1_nv["tokens"]
    tgt_name_toks = tgt_nv["tokens"]
    s1_name_set = set(s1_name_toks)
    tgt_name_set = set(tgt_name_toks)

    name_tok_jaccard = jaccard_similarity(s1_name_set, tgt_name_set)
    name_tok_overlap = float(len(s1_name_set & tgt_name_set))
    name_tok_containment = containment_ratio(s1_name_set, tgt_name_set)
    name_tok_sort = token_sort_ratio(s1_name_toks, tgt_name_toks)
    name_tok_set = token_set_ratio(s1_name_toks, tgt_name_toks)

    name_c3_jaccard = jaccard_similarity(s1_nv["char_3grams"], tgt_nv["char_3grams"])
    name_c4_jaccard = jaccard_similarity(s1_nv["char_4grams"], tgt_nv["char_4grams"])

    name_prefix_match = 1.0 if (s1_nv["prefix_4"] and s1_nv["prefix_4"] == tgt_nv["prefix_4"]) else 0.0
    name_suffix_match = 1.0 if (s1_nv["suffix_4"] and s1_nv["suffix_4"] == tgt_nv["suffix_4"]) else 0.0
    name_first_tok_match = 1.0 if (s1_nv["first_token"] and s1_nv["first_token"] == tgt_nv["first_token"]) else 0.0
    name_last_tok_match = 1.0 if (s1_nv["last_token"] and s1_nv["last_token"] == tgt_nv["last_token"]) else 0.0

    # Acronym match
    name_acronym = 0.0
    if s1_nv["acronym"] and tgt_nv["acronym"] and s1_nv["acronym"] == tgt_nv["acronym"]:
        name_acronym = 1.0
    elif s1_nv["acronym"] and s1_nv["acronym"] in tgt_compact_name:
        name_acronym = 0.8
    elif tgt_nv["acronym"] and tgt_nv["acronym"] in s1_compact_name:
        name_acronym = 0.8

    name_char_len_diff = float(abs(len(s1_clean_name) - len(tgt_clean_name)))
    max_name_len = max(len(s1_clean_name), len(tgt_clean_name), 1)
    name_char_len_ratio = float(min(len(s1_clean_name), len(tgt_clean_name)) / max_name_len)
    name_token_count_diff = float(abs(len(s1_name_toks) - len(tgt_name_toks)))

    # Transliteration similarity
    name_translit = 0.0
    if s1_nv["is_indic"] or tgt_nv["is_indic"]:
        rom_s1 = s1_nv["romanized"] or s1_clean_name
        rom_tgt = tgt_nv["romanized"] or tgt_clean_name
        name_translit = sequence_match_ratio(rom_s1, rom_tgt)

    name_missing = 1.0 if (len(s1_clean_name) == 0 or len(tgt_clean_name) == 0) else 0.0

    # Numeric tokens in name
    s1_name_nums = set(s1_nv["numeric_tokens"])
    tgt_name_nums = set(tgt_nv["numeric_tokens"])
    name_num_token_match = 1.0 if (s1_name_nums and tgt_name_nums and (s1_name_nums & tgt_name_nums)) else 0.0

    # ----------------------------------------------------
    # 2. ADDRESS FEATURES
    # ----------------------------------------------------
    s1_raw_addr = s1_av["raw"]
    tgt_raw_addr = tgt_av["raw"]
    s1_clean_addr = s1_av["clean"]
    tgt_clean_addr = tgt_av["clean"]

    exact_raw_addr = 1.0 if (s1_raw_addr and s1_raw_addr == tgt_raw_addr) else 0.0
    exact_clean_addr = 1.0 if (s1_clean_addr and s1_clean_addr == tgt_clean_addr) else 0.0

    addr_seq = sequence_match_ratio(s1_clean_addr, tgt_clean_addr)
    addr_jw = jaro_winkler_similarity(s1_clean_addr, tgt_clean_addr)

    s1_addr_toks = s1_av["tokens"]
    tgt_addr_toks = tgt_av["tokens"]
    s1_addr_set = s1_av["token_set"]
    tgt_addr_set = tgt_av["token_set"]

    addr_tok_jaccard = jaccard_similarity(s1_addr_set, tgt_addr_set)
    addr_tok_overlap = float(len(s1_addr_set & tgt_addr_set))
    addr_salient_jaccard = jaccard_similarity(s1_av["salient_set"], tgt_av["salient_set"])
    addr_tok_set = token_set_ratio(s1_addr_toks, tgt_addr_toks)
    addr_c3_jaccard = jaccard_similarity(s1_av["char_3grams"], tgt_av["char_3grams"])

    # Postal exact / prefix
    postal_exact = 0.0
    postal_prefix = 0.0
    s1_postals = s1_av["postal_codes"]
    tgt_postals = tgt_av["postal_codes"]
    if s1_postals and tgt_postals:
        if set(s1_postals) & set(tgt_postals):
            postal_exact = 1.0
            postal_prefix = 1.0
        elif any(p1[:3] == p2[:3] for p1 in s1_postals for p2 in tgt_postals if len(p1) >= 3 and len(p2) >= 3):
            postal_prefix = 1.0

    # House number geometry
    s1_hnum = s1_av["first_num"]
    tgt_hnum = tgt_av["first_num"]
    house_num_exact = 1.0 if (s1_hnum and s1_hnum == tgt_hnum) else 0.0
    house_num_abs_diff = 0.0
    house_num_rel_diff = 0.0
    house_num_digit_sim = 0.0

    if s1_hnum and tgt_hnum:
        try:
            n1 = int(s1_hnum)
            n2 = int(tgt_hnum)
            house_num_abs_diff = float(abs(n1 - n2))
            house_num_rel_diff = float(abs(n1 - n2) / max(n1, n2, 1))
        except ValueError:
            pass
        house_num_digit_sim = sequence_match_ratio(s1_hnum, tgt_hnum)

    # City / Locality similarity
    s1_sal = s1_av["salient"]
    tgt_sal = tgt_av["salient"]
    city_sim = sequence_match_ratio(s1_av["first_salient"], tgt_av["first_salient"])
    state_sim = sequence_match_ratio(s1_av["second_salient"], tgt_av["second_salient"])
    locality_sim = jaccard_similarity(set(s1_sal[:3]), set(tgt_sal[:3]))

    # Component agreement vs conflict
    comp_agree = 0.0
    comp_conflict = 0.0
    if postal_exact:
        comp_agree += 1.0
    elif s1_postals and tgt_postals:
        comp_conflict += 1.0

    if house_num_exact:
        comp_agree += 1.0
    elif s1_hnum and tgt_hnum and house_num_abs_diff > 10:
        comp_conflict += 1.0

    addr_char_len_diff = float(abs(len(s1_clean_addr) - len(tgt_clean_addr)))
    addr_missing = 1.0 if (len(s1_clean_addr) == 0 or len(tgt_clean_addr) == 0) else 0.0
    addr_number_jaccard = jaccard_similarity(s1_av["number_set"], tgt_av["number_set"])

    # ----------------------------------------------------
    # 3. NUMERIC EVIDENCE
    # ----------------------------------------------------
    s1_nums = s1_av["number_set"] | s1_name_nums
    tgt_nums = tgt_av["number_set"] | tgt_name_nums

    num_inter_count = float(len(s1_nums & tgt_nums))
    num_jaccard = jaccard_similarity(s1_nums, tgt_nums)
    exact_num_seq = 1.0 if (s1_av["numbers"] and s1_av["numbers"] == tgt_av["numbers"]) else 0.0
    house_num_agree = house_num_exact
    conflicting_num = float(len(s1_nums - tgt_nums) + len(tgt_nums - s1_nums))
    postal_agree = postal_exact
    num_only_match = 1.0 if (num_inter_count > 0 and name_seq < 0.3) else 0.0
    num_conflict_flag = 1.0 if (comp_conflict > 0) else 0.0

    # ----------------------------------------------------
    # 4. RETRIEVAL PROVENANCE EVIDENCE
    # ----------------------------------------------------
    if provenance is not None:
        prov_dict = provenance.to_feature_dict()
    else:
        # Default neutral provenance if not provided
        prov_dict = {
            f"retrieved_by_{ch}": 0.0 for ch in ALL_CHANNELS
        }
        prov_dict["retrieval_agreement_count"] = 1.0
        prov_dict["retrieval_best_rank"] = 0.0
        prov_dict["retrieval_second_best_rank"] = 0.0
        prov_dict["retrieval_rank_gap"] = 0.0
        prov_dict["retrieval_best_score"] = 0.5
        prov_dict["retrieval_second_best_score"] = 0.0
        prov_dict["retrieval_score_gap"] = 0.5
        prov_dict["retrieval_reciprocal_rank"] = 1.0
        prov_dict["forward_reverse_agreement"] = 0.0

    # ----------------------------------------------------
    # 5. CROSS-FIELD INTERACTIONS & METADATA
    # ----------------------------------------------------
    name_x_addr = float(name_seq * addr_seq)
    harmonic_mean = (
        float(2.0 * name_seq * addr_seq / (name_seq + addr_seq))
        if (name_seq + addr_seq) > 0
        else 0.0
    )
    min_evidence = float(min(name_seq, addr_seq))
    max_evidence = float(max(name_seq, addr_seq))
    evidence_diff = float(abs(name_seq - addr_seq))

    exact_name_part_addr = 1.0 if (exact_clean_name and addr_seq >= 0.5) else 0.0
    exact_num_part_name = 1.0 if (house_num_exact and name_seq >= 0.7) else 0.0
    name_x_num = float(name_seq * (1.0 if num_inter_count > 0 else 0.0))

    country_match = 1.0 if (s1_precomputed["country"] == target_precomputed["country"]) else 0.0
    tid_str = str(target_record.get("entity_id", ""))
    is_s2 = 1.0 if tid_str.startswith("S2") else 0.0
    is_s3 = 1.0 if tid_str.startswith("S3") else 0.0

    # Assemble vector matching FEATURE_COLUMNS order
    row: List[float] = [
        # Name
        exact_raw_name, exact_clean_name, exact_legal_name, exact_compact_name,
        name_seq, name_jw, name_tok_jaccard, name_tok_overlap, name_tok_containment,
        name_tok_sort, name_tok_set, name_c3_jaccard, name_c4_jaccard,
        name_prefix_match, name_suffix_match, name_first_tok_match, name_last_tok_match,
        name_acronym, name_char_len_diff, name_char_len_ratio, name_token_count_diff,
        name_translit, name_missing, name_num_token_match,
        # Address
        exact_raw_addr, exact_clean_addr, addr_seq, addr_jw, addr_tok_jaccard,
        addr_tok_overlap, addr_salient_jaccard, addr_tok_set, addr_c3_jaccard,
        postal_exact, postal_prefix, house_num_exact, house_num_abs_diff,
        house_num_rel_diff, house_num_digit_sim, city_sim, state_sim, locality_sim,
        comp_agree, comp_conflict, addr_char_len_diff, addr_missing, addr_number_jaccard,
        # Numeric
        num_inter_count, num_jaccard, exact_num_seq, house_num_agree,
        conflicting_num, postal_agree, num_only_match, num_conflict_flag,
        # Retrieval Provenance
        prov_dict["retrieved_by_exact_name"],
        prov_dict["retrieved_by_exact_addr"],
        prov_dict["retrieved_by_exact_both"],
        prov_dict["retrieved_by_rare_token"],
        prov_dict["retrieved_by_bm25_name"],
        prov_dict["retrieved_by_bm25_addr"],
        prov_dict["retrieved_by_numeric_loc"],
        prov_dict["retrieved_by_cross_script"],
        prov_dict["retrieved_by_missing_field"],
        prov_dict["retrieved_by_sibling"],
        prov_dict["retrieved_by_reverse"],
        prov_dict["retrieval_agreement_count"],
        prov_dict["retrieval_best_rank"],
        prov_dict["retrieval_second_best_rank"],
        prov_dict["retrieval_rank_gap"],
        prov_dict["retrieval_best_score"],
        prov_dict["retrieval_second_best_score"],
        prov_dict["retrieval_score_gap"],
        prov_dict["retrieval_reciprocal_rank"],
        prov_dict["forward_reverse_agreement"],
        # Cross-Field
        name_x_addr, harmonic_mean, min_evidence, max_evidence, evidence_diff,
        exact_name_part_addr, exact_num_part_name, name_x_num, country_match,
        is_s2, is_s3,
    ]

    return row


def build_pairwise_feature_dataframe(
    pairs: List[Tuple[str, str]],
    s1_dict: Dict[str, Dict[str, Any]],
    target_dict: Dict[str, Dict[str, Any]],
    s1_precomputed: Dict[str, Dict[str, Any]],
    target_precomputed: Dict[str, Dict[str, Any]],
    provenance_map: Optional[Dict[Tuple[str, str], RetrievalProvenance]] = None,
) -> pd.DataFrame:
    """Vectorized build of pairwise feature matrix across candidate pairs."""
    rows: List[List[float]] = []

    for sid, tid in pairs:
        s1 = s1_dict[sid]
        tgt = target_dict[tid]
        s1_pre = s1_precomputed[sid]
        tgt_pre = target_precomputed[tid]
        prov = provenance_map.get((sid, tid)) if provenance_map else None

        feat_row = extract_pair_features(s1, tgt, s1_pre, tgt_pre, prov)
        rows.append(feat_row)

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=np.float32)
