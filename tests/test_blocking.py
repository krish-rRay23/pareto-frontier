"""Unit tests for multi-pass candidate blocking."""

import os
import pytest
from src.blocking.blocker import (
    MultiPassBlocker,
    evaluate_blocking,
    export_candidate_pairs_tsv,
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
