"""High-Recall Multi-Channel Bidirectional Retrieval Engine.

Implements the complete retrieval specifications from Amazon ML Challenge 2026 99.99+ Architecture:
1. Open-set Country Partitioning (US, India, France, and global)
2. Multi-Channel Retrieval:
   - Exact normalized name
   - Exact normalized address
   - Exact combined (name + address)
   - Rare-token inverted index
   - BM25 / TF-IDF name channel
   - BM25 / TF-IDF address channel
   - Numeric / postal / locality channel
   - Cross-script Brahmic transliteration channel
   - Missing-field adaptive retrieval
   - Sibling / cluster expansion
3. Bidirectional Retrieval:
   - Forward retrieval: S1 -> S2/S3
   - Reverse retrieval: S2/S3 -> S1
4. Candidate Provenance Tracking (Channels, Ranks, Scores, Gaps, Reciprocal Ranks, Agreement)
5. Adaptive Candidate Budgets (dynamic K based on entity difficulty)
"""

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.normalization.address_normalizer import get_multi_view_address
from src.normalization.text_normalizer import get_multi_view_name


# Channel IDs
CH_EXACT_NAME = "exact_name"
CH_EXACT_ADDR = "exact_addr"
CH_EXACT_BOTH = "exact_both"
CH_RARE_TOKEN = "rare_token"
CH_BM25_NAME = "bm25_name"
CH_BM25_ADDR = "bm25_addr"
CH_NUMERIC_LOC = "numeric_loc"
CH_CROSS_SCRIPT = "cross_script"
CH_MISSING_FIELD = "missing_field"
CH_SIBLING = "sibling"
CH_REVERSE = "reverse"

ALL_CHANNELS = [
    CH_EXACT_NAME,
    CH_EXACT_ADDR,
    CH_EXACT_BOTH,
    CH_RARE_TOKEN,
    CH_BM25_NAME,
    CH_BM25_ADDR,
    CH_NUMERIC_LOC,
    CH_CROSS_SCRIPT,
    CH_MISSING_FIELD,
    CH_SIBLING,
    CH_REVERSE,
]


def normalize_country_key(country: Any) -> str:
    """Standardize country code for partitioning while remaining open-set."""
    if country is None:
        return "unknown"
    c = str(country).strip().lower()
    return c if c else "unknown"


def get_condensed_alphanumeric(text: str) -> str:
    """Normalize web URLs, handles, and strip symbols to compact alphanumerics."""
    if not text:
        return ""
    s = re.sub(r"https?://(?:www\.)?", "", str(text).lower())
    s = re.sub(r"\.(com|org|net|in|co|io|biz|info|gov|fr)\b", "", s)
    s = re.sub(r"^[@#*]+", "", s)
    return re.sub(r"[\W_]+", "", s)


class RetrievalProvenance:
    """Tracks rich second-order provenance evidence for a retrieved candidate pair."""

    __slots__ = (
        "channels",
        "channel_scores",
        "channel_ranks",
        "best_rank",
        "second_best_rank",
        "rank_gap",
        "best_score",
        "second_best_score",
        "score_gap",
        "reciprocal_rank",
        "agreement_count",
        "forward_reverse_agreement",
    )

    def __init__(self):
        self.channels: Set[str] = set()
        self.channel_scores: Dict[str, float] = {}
        self.channel_ranks: Dict[str, int] = {}
        self.best_rank: int = 999
        self.second_best_rank: int = 999
        self.rank_gap: float = 0.0
        self.best_score: float = 0.0
        self.second_best_score: float = 0.0
        self.score_gap: float = 0.0
        self.reciprocal_rank: float = 0.0
        self.agreement_count: int = 0
        self.forward_reverse_agreement: float = 0.0

    def add_hit(self, channel: str, rank: int, score: float):
        self.channels.add(channel)
        if channel not in self.channel_ranks or rank < self.channel_ranks[channel]:
            self.channel_ranks[channel] = rank
            self.channel_scores[channel] = score

    def finalize(self):
        self.agreement_count = len(self.channels)
        ranks = sorted(self.channel_ranks.values()) if self.channel_ranks else [999]
        scores = sorted(self.channel_scores.values(), reverse=True) if self.channel_scores else [0.0]

        self.best_rank = ranks[0]
        self.second_best_rank = ranks[1] if len(ranks) > 1 else ranks[0]
        self.rank_gap = float(self.second_best_rank - self.best_rank)

        self.best_score = scores[0]
        self.second_best_score = scores[1] if len(scores) > 1 else 0.0
        self.score_gap = float(self.best_score - self.second_best_score)

        self.reciprocal_rank = float(1.0 / (1.0 + self.best_rank))

        # Check forward vs reverse agreement
        has_forward = any(ch != CH_REVERSE for ch in self.channels)
        has_reverse = CH_REVERSE in self.channels
        self.forward_reverse_agreement = 1.0 if (has_forward and has_reverse) else 0.0

    def to_feature_dict(self) -> Dict[str, float]:
        """Convert provenance into model feature dictionary."""
        feats = {
            f"retrieved_by_{ch}": (1.0 if ch in self.channels else 0.0)
            for ch in ALL_CHANNELS
        }
        feats["retrieval_agreement_count"] = float(self.agreement_count)
        feats["retrieval_best_rank"] = float(self.best_rank)
        feats["retrieval_second_best_rank"] = float(self.second_best_rank)
        feats["retrieval_rank_gap"] = float(self.rank_gap)
        feats["retrieval_best_score"] = float(self.best_score)
        feats["retrieval_second_best_score"] = float(self.second_best_score)
        feats["retrieval_score_gap"] = float(self.score_gap)
        feats["retrieval_reciprocal_rank"] = float(self.reciprocal_rank)
        feats["forward_reverse_agreement"] = float(self.forward_reverse_agreement)
        return feats


class MultiChannelBidirectionalRetriever:
    """Production Multi-Channel Bidirectional Retrieval Engine."""

    def __init__(
        self,
        max_bucket_size: int = 150,
        max_rare_token_freq: int = 50,
        bm25_top_k: int = 25,
        default_budget: int = 60,
        ambiguous_budget: int = 120,
        exact_budget: int = 25,
        enable_reverse: bool = True,
        enable_bm25: bool = True,
        enable_rare_tokens: bool = True,
        enable_transliteration: bool = True,
    ):
        self.max_bucket_size = max_bucket_size
        self.max_rare_token_freq = max_rare_token_freq
        self.bm25_top_k = bm25_top_k
        self.default_budget = default_budget
        self.ambiguous_budget = ambiguous_budget
        self.exact_budget = exact_budget
        self.enable_reverse = enable_reverse
        self.enable_bm25 = enable_bm25
        self.enable_rare_tokens = enable_rare_tokens
        self.enable_transliteration = enable_transliteration

        # Forward Inverted Indexes: (country, key) -> list of target entity_ids
        self.idx_exact_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_legal_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_sorted_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_compact_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_exact_addr: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_exact_both: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_rare_token: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_pin_num: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_num_sal: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_pin_sal: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_romanized: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_missing_name_addr: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Reverse Inverted Indexes: (country, key) -> list of S1 entity_ids
        self.idx_rev_exact_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_rev_exact_addr: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_rev_sorted_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_rev_pin_num: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.idx_rev_rare_token: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Corpus frequencies for IDF / rare tokens
        self.target_token_freq: Dict[Tuple[str, str], int] = Counter()
        self.target_doc_count: Dict[str, int] = Counter()

        # Inverted index for fast sparse BM25 scoring: (country, token) -> list of (doc_id, tf)
        self.bm25_name_index: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
        self.bm25_addr_index: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
        self.target_name_lengths: Dict[str, int] = {}
        self.target_addr_lengths: Dict[str, int] = {}
        self.target_avg_name_len: Dict[str, float] = defaultdict(lambda: 4.0)
        self.target_avg_addr_len: Dict[str, float] = defaultdict(lambda: 6.0)

        # Cached precomputed multi-views for targets: target_id -> precomputed dict
        self.target_precomputed: Dict[str, Dict[str, Any]] = {}
        self.s1_precomputed_cache: Dict[str, Dict[str, Any]] = {}

    def fit_targets(self, targets: List[Dict[str, Any]]) -> "MultiChannelBidirectionalRetriever":
        """Index all target entities (Source 2 and Source 3) into multi-channel forward indexes."""
        # 1. Precompute views and calculate corpus frequencies
        name_len_sums: Dict[str, int] = Counter()
        addr_len_sums: Dict[str, int] = Counter()

        for t in targets:
            tid = str(t["entity_id"])
            country = normalize_country_key(t.get("country"))
            self.target_doc_count[country] += 1

            nv = get_multi_view_name(t.get("business_name", ""))
            av = get_multi_view_address(t.get("business_address", ""))
            self.target_precomputed[tid] = {"name": nv, "addr": av, "country": country}

            # Count token frequencies for rare-token & BM25 indexing
            tokens = nv["tokens"]
            self.target_name_lengths[tid] = len(tokens)
            name_len_sums[country] += len(tokens)

            addr_tokens = av["tokens"]
            self.target_addr_lengths[tid] = len(addr_tokens)
            addr_len_sums[country] += len(addr_tokens)

            for token in set(tokens):
                self.target_token_freq[(country, token)] += 1

        # Calculate average lengths per country partition
        for country, count in self.target_doc_count.items():
            if count > 0:
                self.target_avg_name_len[country] = float(name_len_sums[country] / count)
                self.target_avg_addr_len[country] = float(addr_len_sums[country] / count)

        # 2. Populate forward multi-channel indexes
        for t in targets:
            tid = str(t["entity_id"])
            country = self.target_precomputed[tid]["country"]
            nv = self.target_precomputed[tid]["name"]
            av = self.target_precomputed[tid]["addr"]

            clean_name = nv["clean"]
            legal_name = nv["legal_stripped"]
            sorted_name = nv["sorted_tokens"]
            compact_name = nv["compact"]
            romanized = nv["romanized"]

            clean_addr = av["clean"]
            first_num = av["first_num"]
            second_num = av["second_num"]
            first_postal = av["first_postal"]
            first_salient = av["first_salient"]
            second_salient = av["second_salient"]

            # Channel 1: Exact Name
            if clean_name:
                self.idx_exact_name[(country, clean_name)].append(tid)
            if legal_name and legal_name != clean_name:
                self.idx_legal_name[(country, legal_name)].append(tid)
            if sorted_name:
                self.idx_sorted_name[(country, sorted_name)].append(tid)
            if compact_name and len(compact_name) >= 4:
                self.idx_compact_name[(country, compact_name)].append(tid)

            # Channel 2: Exact Address
            if clean_addr:
                self.idx_exact_addr[(country, clean_addr)].append(tid)

            # Channel 3: Exact Both
            if clean_name and clean_addr:
                self.idx_exact_both[(country, clean_name, clean_addr)].append(tid)

            # Channel 4: Rare Tokens
            if self.enable_rare_tokens:
                for token in nv["tokens"]:
                    if len(token) >= 3 and self.target_token_freq.get((country, token), 0) <= self.max_rare_token_freq:
                        self.idx_rare_token[(country, token)].append(tid)

            # Channel 5 & 6: BM25 / Sparse term indexes
            if self.enable_bm25:
                token_counts = Counter(nv["tokens"])
                for tok, count in token_counts.items():
                    if len(tok) >= 2:
                        self.bm25_name_index[(country, tok)].append((tid, count))

                addr_counts = Counter(av["tokens"])
                for tok, count in addr_counts.items():
                    if len(tok) >= 2 and tok not in av["salient_set"] and not tok.isdigit():
                        continue
                    self.bm25_addr_index[(country, tok)].append((tid, count))

            # Channel 7: Numeric / Postal / Locality Keys
            if first_postal and first_num:
                self.idx_pin_num[(country, first_postal, first_num)].append(tid)
            if first_postal and first_salient:
                self.idx_pin_sal[(country, first_postal, first_salient)].append(tid)
            if first_num and first_salient:
                self.idx_num_sal[(country, first_num, first_salient)].append(tid)

            # Channel 8: Cross-Script Romanization
            if self.enable_transliteration and romanized:
                self.idx_romanized[(country, romanized)].append(tid)

            # Channel 9: Missing Field Fallback (Postal + Salient or House number + Second number)
            if not clean_name and clean_addr and first_salient and first_num:
                self.idx_missing_name_addr[(country, f"{first_salient}_{first_num}")].append(tid)

        return self

    def fit_s1_reverse(self, s1_records: List[Dict[str, Any]]) -> "MultiChannelBidirectionalRetriever":
        """Index S1 entities to enable reverse retrieval (S2/S3 -> S1)."""
        if not self.enable_reverse:
            return self

        for s1 in s1_records:
            sid = str(s1["entity_id"])
            country = normalize_country_key(s1.get("country"))
            nv = get_multi_view_name(s1.get("business_name", ""))
            av = get_multi_view_address(s1.get("business_address", ""))
            self.s1_precomputed_cache[sid] = {"name": nv, "addr": av, "country": country}

            clean_name = nv["clean"]
            sorted_name = nv["sorted_tokens"]
            clean_addr = av["clean"]
            first_num = av["first_num"]
            first_postal = av["first_postal"]

            if clean_name:
                self.idx_rev_exact_name[(country, clean_name)].append(sid)
            if sorted_name:
                self.idx_rev_sorted_name[(country, sorted_name)].append(sid)
            if clean_addr:
                self.idx_rev_exact_addr[(country, clean_addr)].append(sid)
            if first_postal and first_num:
                self.idx_rev_pin_num[(country, first_postal, first_num)].append(sid)

            for token in nv["tokens"]:
                if len(token) >= 4:
                    self.idx_rev_rare_token[(country, token)].append(sid)

        return self

    def _score_bm25(
        self,
        query_tokens: List[str],
        country: str,
        index: Dict[Tuple[str, str], List[Tuple[str, int]]],
        doc_lengths: Dict[str, int],
        avg_doc_len: float,
        k1: float = 1.2,
        b: float = 0.75,
        top_k: int = 25,
    ) -> List[Tuple[str, float]]:
        """Compute sparse BM25 scores for query tokens against the index."""
        scores: Dict[str, float] = Counter()
        N = max(1, self.target_doc_count.get(country, 1))

        for tok in query_tokens:
            key = (country, tok)
            postings = index.get(key)
            if not postings or len(postings) > self.max_bucket_size * 2:
                continue

            n_q = len(postings)
            idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
            if idf <= 0:
                continue

            for doc_id, tf in postings:
                d_len = doc_lengths.get(doc_id, int(avg_doc_len))
                len_norm = 1.0 - b + b * (d_len / max(1.0, avg_doc_len))
                tf_component = (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
                scores[doc_id] += idf * tf_component

        if not scores:
            return []
        # Return top K sorted descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def query_entity(
        self,
        s1: Dict[str, Any],
        precomputed_s1: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, RetrievalProvenance]]:
        """Retrieve candidate targets for an S1 query across all channels with full provenance."""
        sid = str(s1["entity_id"])
        country = normalize_country_key(s1.get("country"))

        if precomputed_s1:
            nv = precomputed_s1["name"]
            av = precomputed_s1["addr"]
        else:
            nv = get_multi_view_name(s1.get("business_name", ""))
            av = get_multi_view_address(s1.get("business_address", ""))

        clean_name = nv["clean"]
        legal_name = nv["legal_stripped"]
        sorted_name = nv["sorted_tokens"]
        compact_name = nv["compact"]
        romanized = nv["romanized"]
        clean_addr = av["clean"]
        first_num = av["first_num"]
        first_postal = av["first_postal"]
        first_salient = av["first_salient"]

        # Determine adaptive candidate budget
        is_missing_name = (len(clean_name) == 0)
        is_missing_addr = (len(clean_addr) == 0)
        is_unseen_domain = (country == "fr" or country not in ("us", "in", "india"))

        if is_missing_name or is_missing_addr or is_unseen_domain:
            budget = self.ambiguous_budget
        else:
            budget = self.default_budget

        provenance_map: Dict[str, RetrievalProvenance] = defaultdict(RetrievalProvenance)

        # Helper to record hits into provenance
        def record_channel(cands: List[str], channel_name: str, base_score: float = 1.0):
            for rank, tid in enumerate(cands[: self.max_bucket_size]):
                sc = base_score / (1.0 + 0.05 * rank)
                provenance_map[tid].add_hit(channel_name, rank=rank, score=sc)

        # --- Forward Channel 1: Exact Name Keys ---
        if clean_name:
            exact_hits = self.idx_exact_name.get((country, clean_name), [])
            if exact_hits:
                record_channel(exact_hits, CH_EXACT_NAME, base_score=1.0)

        if legal_name and legal_name != clean_name:
            legal_hits = self.idx_legal_name.get((country, legal_name), [])
            if legal_hits:
                record_channel(legal_hits, CH_EXACT_NAME, base_score=0.95)

        if sorted_name:
            sorted_hits = self.idx_sorted_name.get((country, sorted_name), [])
            if sorted_hits:
                record_channel(sorted_hits, CH_EXACT_NAME, base_score=0.90)

        if compact_name and len(compact_name) >= 5:
            compact_hits = self.idx_compact_name.get((country, compact_name), [])
            if compact_hits:
                record_channel(compact_hits, CH_EXACT_NAME, base_score=0.88)

        # --- Forward Channel 2: Exact Address ---
        if clean_addr:
            addr_hits = self.idx_exact_addr.get((country, clean_addr), [])
            if addr_hits:
                record_channel(addr_hits, CH_EXACT_ADDR, base_score=0.95)

        # --- Forward Channel 3: Exact Both ---
        if clean_name and clean_addr:
            both_hits = self.idx_exact_both.get((country, clean_name, clean_addr), [])
            if both_hits:
                record_channel(both_hits, CH_EXACT_BOTH, base_score=1.0)
                budget = min(budget, self.exact_budget)

        # --- Forward Channel 4: Rare Tokens ---
        if self.enable_rare_tokens:
            for token in nv["tokens"]:
                if len(token) >= 3 and self.target_token_freq.get((country, token), 0) <= self.max_rare_token_freq:
                    rare_hits = self.idx_rare_token.get((country, token), [])
                    if rare_hits:
                        record_channel(rare_hits, CH_RARE_TOKEN, base_score=0.85)

        # --- Forward Channel 5 & 6: BM25 Name and Address ---
        if self.enable_bm25:
            if nv["tokens"]:
                bm25_name_hits = self._score_bm25(
                    nv["tokens"],
                    country,
                    self.bm25_name_index,
                    self.target_name_lengths,
                    self.target_avg_name_len[country],
                    top_k=self.bm25_top_k,
                )
                for rank, (tid, score) in enumerate(bm25_name_hits):
                    provenance_map[tid].add_hit(CH_BM25_NAME, rank=rank, score=float(score))

            if av["tokens"]:
                bm25_addr_hits = self._score_bm25(
                    av["tokens"],
                    country,
                    self.bm25_addr_index,
                    self.target_addr_lengths,
                    self.target_avg_addr_len[country],
                    top_k=self.bm25_top_k,
                )
                for rank, (tid, score) in enumerate(bm25_addr_hits):
                    provenance_map[tid].add_hit(CH_BM25_ADDR, rank=rank, score=float(score))

        # --- Forward Channel 7: Numeric / Postal / Locality ---
        if first_postal and first_num:
            pin_num_hits = self.idx_pin_num.get((country, first_postal, first_num), [])
            if pin_num_hits:
                record_channel(pin_num_hits, CH_NUMERIC_LOC, base_score=0.80)

        if first_postal and first_salient:
            pin_sal_hits = self.idx_pin_sal.get((country, first_postal, first_salient), [])
            if pin_sal_hits:
                record_channel(pin_sal_hits, CH_NUMERIC_LOC, base_score=0.75)

        if first_num and first_salient:
            num_sal_hits = self.idx_num_sal.get((country, first_num, first_salient), [])
            if num_sal_hits:
                record_channel(num_sal_hits, CH_NUMERIC_LOC, base_score=0.70)

        # --- Forward Channel 8: Cross-Script Romanization ---
        if self.enable_transliteration and romanized:
            rom_hits = self.idx_romanized.get((country, romanized), [])
            if rom_hits:
                record_channel(rom_hits, CH_CROSS_SCRIPT, base_score=0.85)

        # --- Forward Channel 9: Missing-Field Fallback ---
        if is_missing_name and clean_addr:
            if first_salient and first_num:
                mf_hits = self.idx_missing_name_addr.get((country, f"{first_salient}_{first_num}"), [])
                if mf_hits:
                    record_channel(mf_hits, CH_MISSING_FIELD, base_score=0.70)

        # --- Forward Channel 10: Sibling / Cluster Expansion ---
        # If top hits agree on postal and house number, expand to siblings sharing that location
        if first_postal and first_num:
            sibling_hits = self.idx_pin_num.get((country, first_postal, first_num), [])
            if sibling_hits and len(sibling_hits) <= 20:
                record_channel(sibling_hits, CH_SIBLING, base_score=0.65)

        # --- Reverse Channel: Reverse Retrieval (S2/S3 -> S1) ---
        if self.enable_reverse:
            # Check which targets in our candidate set link back to this S1
            for tid in list(provenance_map.keys()):
                tgt_info = self.target_precomputed.get(tid)
                if not tgt_info:
                    continue
                tnv = tgt_info["name"]
                tav = tgt_info["addr"]
                t_country = tgt_info["country"]

                # Reverse match checks
                rev_matched = False
                if tnv["clean"] and sid in self.idx_rev_exact_name.get((t_country, tnv["clean"]), []):
                    rev_matched = True
                elif tnv["sorted_tokens"] and sid in self.idx_rev_sorted_name.get((t_country, tnv["sorted_tokens"]), []):
                    rev_matched = True
                elif tav["clean"] and sid in self.idx_rev_exact_addr.get((t_country, tav["clean"]), []):
                    rev_matched = True
                elif tav["first_postal"] and tav["first_num"] and sid in self.idx_rev_pin_num.get((t_country, tav["first_postal"], tav["first_num"]), []):
                    rev_matched = True

                if rev_matched:
                    provenance_map[tid].add_hit(CH_REVERSE, rank=0, score=1.0)

        # Finalize provenance metrics
        for prov in provenance_map.values():
            prov.finalize()

        # Rank candidates by: agreement_count desc, best_score desc, best_rank asc
        ranked_candidates = sorted(
            provenance_map.keys(),
            key=lambda tid: (
                provenance_map[tid].agreement_count,
                provenance_map[tid].best_score,
                -provenance_map[tid].best_rank,
            ),
            reverse=True,
        )

        final_candidates = ranked_candidates[:budget]
        return final_candidates, {tid: provenance_map[tid] for tid in final_candidates}
