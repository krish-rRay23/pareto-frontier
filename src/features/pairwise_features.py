"""High-performance pairwise feature engineering for business entity matching."""

import difflib
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from src.normalization.address_normalizer import get_multi_view_address
from src.normalization.text_normalizer import get_multi_view_name


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from a string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(intersection / union) if union > 0 else 0.0


def containment_ratio(set_a: Set, set_b: Set) -> float:
    """Compute containment: |A & B| / min(|A|, |B|)."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    min_size = min(len(set_a), len(set_b))
    return float(len(set_a & set_b) / min_size) if min_size > 0 else 0.0


def sequence_match_ratio(str_a: str, str_b: str) -> float:
    """Compute normalized sequence similarity ratio."""
    if not str_a and not str_b:
        return 1.0
    if not str_a or not str_b:
        return 0.0
    # Fast path for exact match
    if str_a == str_b:
        return 1.0
    return float(difflib.SequenceMatcher(None, str_a, str_b).ratio())


FEATURE_COLUMNS = [
    "exact_clean_name",
    "exact_legal_name",
    "name_char_len_diff",
    "name_char_len_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_containment",
    "name_char_3gram_jaccard",
    "name_seq_ratio",
    "name_prefix_match_4",
    "exact_clean_address",
    "addr_char_len_diff",
    "addr_token_jaccard",
    "addr_salient_jaccard",
    "addr_number_match",
    "addr_number_jaccard",
    "addr_missing_indicator",
    "is_s2",
    "is_s3",
    "country_match",
    "name_x_addr_jaccard",
    "name_x_num_match",
    "exact_name_partial_addr",
    "exact_num_partial_name",
]


def extract_pair_features(
    s1_record: Dict[str, object],
    target_record: Dict[str, object],
    s1_precomputed: Dict[str, object],
    target_precomputed: Dict[str, object],
) -> List[float]:
    """Compute 24 numeric features for a single (S1, Target) candidate pair."""
    # Precomputed views
    s1_nv = s1_precomputed["name"]
    tgt_nv = target_precomputed["name"]
    s1_av = s1_precomputed["addr"]
    tgt_av = target_precomputed["addr"]

    # 1. Names
    s1_clean_name = s1_nv["clean"]
    tgt_clean_name = tgt_nv["clean"]
    s1_legal_name = s1_nv["legal_stripped"]
    tgt_legal_name = tgt_nv["legal_stripped"]

    exact_clean_name = 1.0 if (s1_clean_name and s1_clean_name == tgt_clean_name) else 0.0
    exact_legal_name = 1.0 if (s1_legal_name and s1_legal_name == tgt_legal_name) else 0.0

    len1 = len(s1_legal_name)
    len2 = len(tgt_legal_name)
    name_char_len_diff = float(abs(len1 - len2))
    name_char_len_ratio = float(min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0

    s1_tokens = s1_precomputed["name_tokens"]
    tgt_tokens = target_precomputed["name_tokens"]
    name_token_jaccard = jaccard_similarity(s1_tokens, tgt_tokens)
    name_token_overlap = float(len(s1_tokens & tgt_tokens))
    name_token_containment = containment_ratio(s1_tokens, tgt_tokens)

    s1_ngrams = s1_precomputed["name_ngrams"]
    tgt_ngrams = target_precomputed["name_ngrams"]
    name_char_3gram_jaccard = jaccard_similarity(s1_ngrams, tgt_ngrams)

    name_seq_ratio = sequence_match_ratio(s1_legal_name, tgt_legal_name)
    name_prefix_match_4 = 1.0 if (len1 >= 4 and len2 >= 4 and s1_legal_name[:4] == tgt_legal_name[:4]) else 0.0

    # 2. Addresses
    s1_clean_addr = s1_av["clean"]
    tgt_clean_addr = tgt_av["clean"]
    exact_clean_address = 1.0 if (s1_clean_addr and s1_clean_addr == tgt_clean_addr) else 0.0

    addr_len1 = len(s1_clean_addr)
    addr_len2 = len(tgt_clean_addr)
    addr_char_len_diff = float(abs(addr_len1 - addr_len2))

    s1_addr_tokens = s1_precomputed["addr_tokens"]
    tgt_addr_tokens = target_precomputed["addr_tokens"]
    addr_token_jaccard = jaccard_similarity(s1_addr_tokens, tgt_addr_tokens)

    s1_salient = s1_precomputed["salient_tokens"]
    tgt_salient = target_precomputed["salient_tokens"]
    addr_salient_jaccard = jaccard_similarity(s1_salient, tgt_salient)

    s1_nums = s1_precomputed["addr_numbers"]
    tgt_nums = target_precomputed["addr_numbers"]
    s1_prim_num = s1_av["primary_number"]
    tgt_prim_num = tgt_av["primary_number"]
    addr_number_match = 1.0 if (s1_prim_num and s1_prim_num == tgt_prim_num) else 0.0
    addr_number_jaccard = jaccard_similarity(s1_nums, tgt_nums)

    is_missing = 1.0 if (not s1_clean_addr or not tgt_clean_addr or tgt_clean_addr == "nan") else 0.0

    # 3. Source & Country
    target_id = target_record.get("entity_id", "")
    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0
    country_match = 1.0 if s1_record.get("country") == target_record.get("country") else 0.0

    # 4. Interactions
    name_x_addr_jaccard = name_token_jaccard * addr_token_jaccard
    name_x_num_match = name_token_jaccard * addr_number_match
    exact_name_partial_addr = 1.0 if (exact_legal_name == 1.0 and addr_token_jaccard >= 0.25) else 0.0
    exact_num_partial_name = 1.0 if (addr_number_match == 1.0 and name_token_jaccard >= 0.40) else 0.0

    return [
        exact_clean_name,
        exact_legal_name,
        name_char_len_diff,
        name_char_len_ratio,
        name_token_jaccard,
        name_token_overlap,
        name_token_containment,
        name_char_3gram_jaccard,
        name_seq_ratio,
        name_prefix_match_4,
        exact_clean_address,
        addr_char_len_diff,
        addr_token_jaccard,
        addr_salient_jaccard,
        addr_number_match,
        addr_number_jaccard,
        is_missing,
        is_s2,
        is_s3,
        country_match,
        name_x_addr_jaccard,
        name_x_num_match,
        exact_name_partial_addr,
        exact_num_partial_name,
    ]


def precompute_record_views(record: Dict[str, str]) -> Dict[str, object]:
    """Precompute normalized views and token sets for high-throughput feature extraction."""
    bname = record.get("business_name", "")
    baddr = record.get("business_address", "")

    nv = get_multi_view_name(bname)
    av = get_multi_view_address(baddr)

    name_tokens = set(nv["legal_stripped"].split())
    name_ngrams = char_ngrams(nv["legal_stripped"], n=3)
    addr_tokens = set(av["clean"].split())
    salient_tokens = set(av["salient_tokens"])
    addr_numbers = set(av["numbers"])

    return {
        "name": nv,
        "addr": av,
        "name_tokens": name_tokens,
        "name_ngrams": name_ngrams,
        "addr_tokens": addr_tokens,
        "salient_tokens": salient_tokens,
        "addr_numbers": addr_numbers,
    }


def build_pairwise_feature_matrix(
    s1_dict: Dict[str, Dict[str, str]],
    target_dict: Dict[str, Dict[str, str]],
    candidates_dict: Dict[str, Set[str]],
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[Tuple[str, str]]]:
    """Build feature matrix X, binary labels y, and list of (s1_id, target_id) pairs."""
    # 1. Precompute all S1 views
    s1_precomputed = {sid: precompute_record_views(r) for sid, r in s1_dict.items()}

    # 2. Collect unique target IDs appearing in candidates and precompute their views
    active_target_ids = set()
    for cands in candidates_dict.values():
        active_target_ids.update(cands)

    target_precomputed = {
        tid: precompute_record_views(target_dict[tid])
        for tid in active_target_ids
        if tid in target_dict
    }

    # 3. Compute pairwise rows
    feature_rows = []
    labels = []
    pairs = []

    for sid, cands in candidates_dict.items():
        if sid not in s1_dict:
            continue
        s1_r = s1_dict[sid]
        s1_pv = s1_precomputed[sid]
        true_mids = ground_truth.get(sid, set()) if ground_truth is not None else None

        for tid in cands:
            if tid not in target_dict:
                continue
            tgt_r = target_dict[tid]
            tgt_pv = target_precomputed[tid]

            feats = extract_pair_features(s1_r, tgt_r, s1_pv, tgt_pv)
            feature_rows.append(feats)
            pairs.append((sid, tid))

            if true_mids is not None:
                labels.append(1 if tid in true_mids else 0)

    if not feature_rows:
        df_empty = pd.DataFrame(columns=FEATURE_COLUMNS)
        return df_empty, (np.array([], dtype=int) if ground_truth is not None else None), []

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    y = np.array(labels, dtype=int) if ground_truth is not None else None
    return X, y, pairs
