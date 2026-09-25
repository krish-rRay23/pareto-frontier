"""Production High-Recall Country-Partitioned Multi-Pass Blocker.

Implements:
1. Order-invariant sorted-token blocking
2. Rare-token inverted retrieval
3. Character 3/4-gram retrieval
4. Stronger address multi-number and multi-salient retrieval
5. Indic script romanization and transliteration
6. Condensed web/social handle normalization
7. Tiered candidate prioritization (protecting high-confidence matches from truncation)
8. Adaptive fallback for unresolved/low-confidence S1s
"""

from collections import Counter, defaultdict
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.normalization.address_normalizer import get_multi_view_address
from src.normalization.text_normalizer import get_multi_view_name


def get_condensed_name(raw: str) -> str:
    """Normalize web URLs, social handles, and strip spaces/symbols."""
    if not raw or not isinstance(raw, str):
        return ""
    # Strip URL schemes and domain extensions
    s = re.sub(r"https?://(?:www\.)?", "", raw.lower())
    s = re.sub(r"\.(com|org|net|in|co|io|biz|info|gov)\b", "", s)
    # Strip leading social symbols and asterisks
    s = re.sub(r"^[@#*]+", "", s)
    # Keep only alphanumeric characters
    return re.sub(r"[\W_]+", "", s)


def extract_all_digits(text: str) -> List[str]:
    """Extract all digit sequences with leading zeros stripped."""
    if not text or not isinstance(text, str):
        return []
    raw = re.findall(r"\d{1,7}", str(text))
    return [n.lstrip("0") or "0" for n in raw]


class MultiPassBlocker:
    """Configurable high-recall country-partitioned multi-pass candidate generator."""

    def __init__(
        self,
        max_candidates_per_entity: Optional[int] = 150,
        max_bucket_size: int = 150,
        max_rare_token_freq: int = 40,
        enable_name_exact: bool = True,
        enable_name_legal_stripped: bool = True,
        enable_name_first_two: bool = True,
        enable_name_sorted_tokens: bool = True,
        enable_indic_transliteration: bool = True,
        enable_condensed_name: bool = True,
        enable_address_multi: bool = True,
        enable_rare_tokens: bool = True,
        enable_char_ngrams: bool = True,
        enable_adaptive_fallback: bool = True,
    ):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.max_bucket_size = max_bucket_size
        self.max_rare_token_freq = max_rare_token_freq

        # Flags for ablation benchmarking
        self.enable_name_exact = enable_name_exact
        self.enable_name_legal_stripped = enable_name_legal_stripped
        self.enable_name_first_two = enable_name_first_two
        self.enable_name_sorted_tokens = enable_name_sorted_tokens
        self.enable_indic_transliteration = enable_indic_transliteration
        self.enable_condensed_name = enable_condensed_name
        self.enable_address_multi = enable_address_multi
        self.enable_rare_tokens = enable_rare_tokens
        self.enable_char_ngrams = enable_char_ngrams
        self.enable_adaptive_fallback = enable_adaptive_fallback

        # Tier 1 Inverted Indexes: High-Precision Name Keys
        self.idx_name_clean: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_name_legal: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_name_first_two: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_name_sorted: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_name_condensed: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Tier 2 Inverted Indexes: Address Numbers, Postal Codes, and Rare Tokens
        self.idx_addr_num_sal: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_addr_pin_num: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_addr_pin_sal: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_addr_num_pair: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_rare_token: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Tier 3 Inverted Indexes: Salient Address Pairs, N-Grams, and Fallback
        self.idx_addr_sal_pair: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_char_ngram: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_fallback_sal: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Internal corpus frequency table for rare-token filtering
        self.token_corpus_freq: Dict[Tuple[str, str], int] = Counter()

    def fit_targets(self, target_records: List[Dict[str, str]]) -> "MultiPassBlocker":
        """Index all candidate target records."""
        # 1. Compute corpus token frequencies for rare-token indexing
        if self.enable_rare_tokens:
            for r in target_records:
                country = r.get("country", "")
                nv = get_multi_view_name(r.get("business_name", ""))
                for t in nv.get("tokens", []):
                    if len(t) >= 4:
                        self.token_corpus_freq[(country, t)] += 1
                if self.enable_indic_transliteration and nv.get("is_indic"):
                    for t in nv.get("romanized_tokens", []):
                        if len(t) >= 4:
                            self.token_corpus_freq[(country, t)] += 1

        # 2. Populate tiered inverted index tables
        for r in target_records:
            eid = r["entity_id"]
            country = r.get("country", "")
            bname = r.get("business_name", "")
            baddr = r.get("business_address", "")

            nv = get_multi_view_name(bname)
            av = get_multi_view_address(baddr)
            cond = get_condensed_name(bname) if self.enable_condensed_name else ""

            # Tier 1: Name Indexing
            if self.enable_name_exact and nv["clean"]:
                self.idx_name_clean[(country, nv["clean"])].append(eid)
            if self.enable_name_legal_stripped and nv["legal_stripped"]:
                self.idx_name_legal[(country, nv["legal_stripped"])].append(eid)
            if self.enable_name_first_two and nv["first_two"]:
                self.idx_name_first_two[(country, nv["first_two"])].append(eid)
            if self.enable_name_sorted_tokens and nv.get("sorted_tokens"):
                self.idx_name_sorted[(country, nv["sorted_tokens"])].append(eid)
            if self.enable_condensed_name and cond and len(cond) >= 5:
                self.idx_name_condensed[(country, cond)].append(eid)

            # Indic Script Romanization & Transliteration
            if self.enable_indic_transliteration and nv.get("is_indic"):
                rom = nv.get("romanized", "")
                if rom:
                    self.idx_name_legal[(country, rom)].append(eid)
                rom_toks = nv.get("romanized_tokens", [])
                if len(rom_toks) >= 2:
                    self.idx_name_first_two[(country, f"{rom_toks[0]} {rom_toks[1]}")].append(eid)
                if self.enable_name_sorted_tokens:
                    rom_sorted = "_".join(sorted([t for t in rom_toks if len(t) >= 3][:4]))
                    if rom_sorted:
                        self.idx_name_sorted[(country, rom_sorted)].append(eid)

            # Tier 2: Address Indexing
            nums = extract_all_digits(baddr) if self.enable_address_multi else av.get("numbers", [])
            sals = av.get("salient_tokens", [])
            pins = av.get("postal_codes", [])

            if self.enable_address_multi:
                # Number + Salient Address Token
                for n in nums[:4]:
                    for s in sals[:12]:
                        self.idx_addr_num_sal[(country, n, s)].append(eid)
                # Postal PIN/ZIP Code + Number & Salient
                for p in pins[:2]:
                    for n in nums[:2]:
                        self.idx_addr_pin_num[(country, p, n)].append(eid)
                    for s in sals[:4]:
                        self.idx_addr_pin_sal[(country, p, s)].append(eid)
                # Number Pairs (e.g. 718 94, 390 391)
                if len(nums) >= 2:
                    n1, n2 = sorted(nums[:2])
                    self.idx_addr_num_pair[(country, n1, n2)].append(eid)
            else:
                # Baseline Address Indexing
                primary_num = av.get("primary_number", "")
                if primary_num and sals:
                    for s in sals[:3]:
                        self.idx_addr_num_sal[(country, primary_num, s)].append(eid)

            # Rare-Token Indexing
            if self.enable_rare_tokens:
                cand_toks = list(nv.get("tokens", []))
                if self.enable_indic_transliteration and nv.get("is_indic"):
                    cand_toks.extend(nv.get("romanized_tokens", []))
                for t in cand_toks:
                    if len(t) >= 4 and self.token_corpus_freq.get((country, t), 0) <= self.max_rare_token_freq:
                        self.idx_rare_token[(country, t)].append(eid)

            # Tier 3: Salient Address Token Pairs
            if sals and len(sals) >= 2:
                pair_limit = 5 if self.enable_address_multi else 2
                for i in range(min(pair_limit, len(sals))):
                    for j in range(i + 1, min(pair_limit, len(sals))):
                        self.idx_addr_sal_pair[(country, min(sals[i], sals[j]), max(sals[i], sals[j]))].append(eid)

            # Character N-Gram Prefix Pairs
            if self.enable_char_ngrams:
                tokens = nv.get("tokens", [])
                if len(tokens) >= 2 and len(tokens[0]) >= 3 and len(tokens[1]) >= 3:
                    p1, p2 = tokens[0][:3], tokens[1][:3]
                    self.idx_char_ngram[(country, min(p1, p2), max(p1, p2))].append(eid)

            # Adaptive Fallback Index
            if self.enable_adaptive_fallback and sals:
                for s in sals[:2]:
                    if len(s) >= 5:
                        self.idx_fallback_sal[(country, s)].append(eid)

        return self

    def query_entity(self, s1_record: Dict[str, str]) -> Set[str]:
        """Retrieve candidate target IDs using tiered prioritization."""
        country = s1_record.get("country", "")
        bname = s1_record.get("business_name", "")
        baddr = s1_record.get("business_address", "")

        nv = get_multi_view_name(bname)
        av = get_multi_view_address(baddr)
        cond = get_condensed_name(bname) if self.enable_condensed_name else ""

        tier1: Set[str] = set()
        tier2: Set[str] = set()
        tier3: Set[str] = set()

        def _add(index_dict, key, target_set, cap=150):
            matches = index_dict.get(key, [])
            if 0 < len(matches) <= cap:
                target_set.update(matches)

        # -----------------------------------------------------------------
        # Tier 1: High-Confidence Name Views
        # -----------------------------------------------------------------
        if self.enable_name_exact and nv["clean"]:
            _add(self.idx_name_clean, (country, nv["clean"]), tier1, cap=self.max_bucket_size)
        if self.enable_name_legal_stripped and nv["legal_stripped"]:
            _add(self.idx_name_legal, (country, nv["legal_stripped"]), tier1, cap=self.max_bucket_size)
        if self.enable_name_first_two and nv["first_two"]:
            _add(self.idx_name_first_two, (country, nv["first_two"]), tier1, cap=self.max_bucket_size)
        if self.enable_name_sorted_tokens and nv.get("sorted_tokens"):
            _add(self.idx_name_sorted, (country, nv["sorted_tokens"]), tier1, cap=self.max_bucket_size)
        if self.enable_condensed_name and cond and len(cond) >= 5:
            _add(self.idx_name_condensed, (country, cond), tier1, cap=self.max_bucket_size)

        # -----------------------------------------------------------------
        # Tier 2: Address Numbers, Postal Codes, and Rare Tokens
        # -----------------------------------------------------------------
        nums = extract_all_digits(baddr) if self.enable_address_multi else av.get("numbers", [])
        sals = av.get("salient_tokens", [])
        pins = av.get("postal_codes", [])

        if self.enable_address_multi:
            for n in nums[:4]:
                for s in sals[:12]:
                    _add(self.idx_addr_num_sal, (country, n, s), tier2, cap=100)
            for p in pins[:2]:
                for n in nums[:2]:
                    _add(self.idx_addr_pin_num, (country, p, n), tier2, cap=80)
                for s in sals[:4]:
                    _add(self.idx_addr_pin_sal, (country, p, s), tier2, cap=60)
            if len(nums) >= 2:
                n1, n2 = sorted(nums[:2])
                _add(self.idx_addr_num_pair, (country, n1, n2), tier2, cap=60)
        else:
            primary_num = av.get("primary_number", "")
            if primary_num and sals:
                for s in sals[:3]:
                    _add(self.idx_addr_num_sal, (country, primary_num, s), tier2, cap=100)

        if self.enable_rare_tokens:
            for t in nv.get("tokens", []):
                if len(t) >= 4:
                    _add(self.idx_rare_token, (country, t), tier2, cap=self.max_rare_token_freq)

        # -----------------------------------------------------------------
        # Tier 3: Salient Pairs, Character N-Grams, and Adaptive Fallback
        # -----------------------------------------------------------------
        if sals and len(sals) >= 2:
            pair_limit = 5 if self.enable_address_multi else 2
            for i in range(min(pair_limit, len(sals))):
                for j in range(i + 1, min(pair_limit, len(sals))):
                    _add(self.idx_addr_sal_pair, (country, min(sals[i], sals[j]), max(sals[i], sals[j])), tier3, cap=50)

        if self.enable_char_ngrams:
            tokens = nv.get("tokens", [])
            if len(tokens) >= 2 and len(tokens[0]) >= 3 and len(tokens[1]) >= 3:
                p1, p2 = tokens[0][:3], tokens[1][:3]
                _add(self.idx_char_ngram, (country, min(p1, p2), max(p1, p2)), tier3, cap=50)

        # Adaptive Fallback for Unresolved Entities (<= 2 candidates found)
        if self.enable_adaptive_fallback and (len(tier1) + len(tier2)) <= 2:
            for s in sals[:2]:
                if len(s) >= 5:
                    _add(self.idx_fallback_sal, (country, s), tier3, cap=35)

        # -----------------------------------------------------------------
        # Tiered Candidate Assembly (Priority Tier 1 > Tier 2 > Tier 3)
        # -----------------------------------------------------------------
        if self.max_candidates_per_entity is None:
            final_cands = set(tier1) | set(tier2) | set(tier3)
            return final_cands

        max_cap = self.max_candidates_per_entity
        cands: List[str] = list(tier1)
        if len(cands) < max_cap:
            cands.extend([x for x in sorted(tier2) if x not in tier1][: max_cap - len(cands)])
        if len(cands) < max_cap:
            rem_set = set(cands)
            cands.extend([x for x in sorted(tier3) if x not in rem_set][: max_cap - len(cands)])

        return set(cands)

    def block_all(
        self, s1_records: List[Dict[str, str]]
    ) -> Dict[str, Set[str]]:
        """Run blocking for a list of S1 records."""
        result: Dict[str, Set[str]] = {}
        for r in s1_records:
            sid = r["entity_id"]
            result[sid] = self.query_entity(r)
        return result


def evaluate_blocking(
    candidates_dict: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    total_target_pool_size: int,
) -> Dict[str, float]:
    """Calculate blocking quality metrics: recall, avg candidates, reduction ratio."""
    hits = 0
    total_true_matches = sum(len(mids) for mids in ground_truth.values())
    cand_counts = []

    for s1_id, true_mids in ground_truth.items():
        found = candidates_dict.get(s1_id, set())
        cand_counts.append(len(found))
        hits += len(true_mids & found)

    recall = (hits / total_true_matches) if total_true_matches > 0 else 0.0
    avg_cands = float(np.mean(cand_counts)) if cand_counts else 0.0
    p95_cands = float(np.percentile(cand_counts, 95)) if cand_counts else 0.0
    p99_cands = float(np.percentile(cand_counts, 99)) if cand_counts else 0.0
    max_cands = int(np.max(cand_counts)) if cand_counts else 0

    total_pairs = sum(cand_counts)
    total_possible = len(ground_truth) * total_target_pool_size
    reduction_ratio = 1.0 - (total_pairs / total_possible) if total_possible > 0 else 1.0

    return {
        "candidate_recall": recall,
        "hits": hits,
        "total_true_matches": total_true_matches,
        "avg_candidates_per_s1": avg_cands,
        "p95_candidates_per_s1": p95_cands,
        "p99_candidates_per_s1": p99_cands,
        "max_candidates_per_s1": max_cands,
        "reduction_ratio": reduction_ratio,
        "total_candidate_pairs": total_pairs,
    }


def export_candidate_pairs_tsv(
    candidates_dict: Dict[str, Set[str]],
    output_path: str,
    s1_ordered_ids: Optional[List[str]] = None,
) -> None:
    """Write candidate_pairs.tsv in the exact official tab-separated format."""
    if s1_ordered_ids is None:
        s1_ordered_ids = sorted(candidates_dict.keys())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ordered_ids:
            cands = candidates_dict.get(s1_id, set())
            cand_str = ",".join(sorted(cands)) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
