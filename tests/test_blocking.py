"""Unit tests for multi-pass and multi-channel bidirectional retrieval."""

import os
import pytest
from src.blocking.blocker import (
    MultiPassBlocker,
    evaluate_blocking,
    export_candidate_pairs_tsv,
)
from src.blocking.retrieval import (
    MultiChannelBidirectionalRetriever,
    CH_EXACT_NAME,
    CH_EXACT_ADDR,
    CH_REVERSE,
    CH_MISSING_FIELD,
)


def test_multi_pass_blocker_hits():
    """Verify that multi-pass blocker catches clean, legal-stripped, and address matches."""
    targets = [
        {"entity_id": "S2-101", "business_name": "Novent Owl LLC", "business_address": "9308 Home Ct, Des Plaines, IL", "country": "US"},
        {"entity_id": "S3-202", "business_name": "Novent Ówl", "business_address": "9308 Home Court, Des Plaines, IL", "country": "US"},
        {"entity_id": "S2-303", "business_name": "Unrelated Store", "business_address": "123 Main St, New York, NY", "country": "US"},
    ]
    s1 = {"entity_id": "S1-001", "business_name": "Novent Owl PLLC", "business_address": "Des Plaines, IL, 9308 Home Court", "country": "US"}

    blocker = MultiPassBlocker()
    blocker.fit_targets(targets)
    cands = blocker.query_entity(s1)

    assert "S2-101" in cands
    assert "S3-202" in cands
    assert "S2-303" not in cands


def test_country_isolation():
    """Candidates from different countries must NEVER be paired."""
    targets = [
        {"entity_id": "S2-999", "business_name": "Solar Solutions", "business_address": "8124 Vasant Kunj, Delhi", "country": "India"},
    ]
    s1_us = {"entity_id": "S1-US", "business_name": "Solar Solutions", "business_address": "8124 Vasant Kunj, Delhi", "country": "US"}

    blocker = MultiPassBlocker()
    blocker.fit_targets(targets)
    cands = blocker.query_entity(s1_us)

    assert len(cands) == 0


def test_candidate_export_tsv(tmp_path):
    """Verify candidate_pairs.tsv formatting."""
    cands_dict = {
        "S1-001": {"S2-10", "S3-20"},
        "S1-002": set(),
    }
    out_file = str(tmp_path / "candidate_pairs.tsv")
    export_candidate_pairs_tsv(cands_dict, out_file)

    with open(out_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert "S1-001\tS2-10,S3-20" in lines
    assert "S1-002" in lines[2]


def test_multi_channel_bidirectional_retriever():
    """Verify multi-channel retrieval, reverse retrieval, and provenance recording."""
    targets = [
        {"entity_id": "S2-10", "business_name": "Acme Widgets Corp", "business_address": "404 Main St, Austin, TX 78701", "country": "US"},
        {"entity_id": "S3-20", "business_name": "Boulangerie Parisienne", "business_address": "12 Rue de Rivoli, 75001 Paris", "country": "FR"},
        {"entity_id": "S2-30", "business_name": "", "business_address": "900 Market St, Austin, TX 78701", "country": "US"},
    ]
    s1_list = [
        {"entity_id": "S1-01", "business_name": "Acme Widgets", "business_address": "Austin, TX 78701, 404 Main Street", "country": "US"},
        {"entity_id": "S1-FR", "business_name": "Boulangerie Parisienne SAS", "business_address": "12 R de Rivoli, Paris 75001", "country": "FR"},
        {"entity_id": "S1-MISSING", "business_name": "", "business_address": "Austin, TX, 900 Market Street", "country": "US"},
    ]

    retriever = MultiChannelBidirectionalRetriever(enable_reverse=True, enable_bm25=True)
    retriever.fit_targets(targets)
    retriever.fit_s1_reverse(s1_list)

    # 1. Test standard S1 query
    cands_01, prov_01 = retriever.query_entity(s1_list[0])
    assert "S2-10" in cands_01
    prov = prov_01["S2-10"]
    assert prov.agreement_count >= 1
    assert prov.best_rank == 0
    assert prov.best_score > 0
    feat_dict = prov.to_feature_dict()
    assert "retrieval_agreement_count" in feat_dict
    assert feat_dict["forward_reverse_agreement"] in (0.0, 1.0)

    # 2. Test French entity open-set retrieval
    cands_fr, prov_fr = retriever.query_entity(s1_list[1])
    assert "S3-20" in cands_fr

    # 3. Test missing name fallback retrieval
    cands_miss, prov_miss = retriever.query_entity(s1_list[2])
    assert "S2-30" in cands_miss
