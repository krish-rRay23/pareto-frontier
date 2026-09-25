"""Unit tests for pairwise feature extraction."""

import pytest
from src.features.pairwise_features import (
    extract_pair_features,
    precompute_record_views,
    jaccard_similarity,
    sequence_match_ratio,
)


def test_jaccard_similarity():
    assert jaccard_similarity({"apple", "banana"}, {"apple", "banana"}) == 1.0
    assert jaccard_similarity({"apple"}, {"orange"}) == 0.0
    assert jaccard_similarity(set(), set()) == 1.0


def test_pairwise_feature_extraction():
    s1 = {"entity_id": "S1-1", "business_name": "Novent Owl PLLC", "business_address": "9308 Home Court, Des Plaines, IL", "country": "US"}
    t1 = {"entity_id": "S2-1", "business_name": "Novent Owl LLC", "business_address": "9308 Home Ct, Des Plaines, IL", "country": "US"}

    s1_pv = precompute_record_views(s1)
    t1_pv = precompute_record_views(t1)

    feats = extract_pair_features(s1, t1, s1_pv, t1_pv)
    assert len(feats) == 24
    # exact_legal_name is index 1
    assert feats[1] == 1.0
    # addr_number_match is index 14
    assert feats[14] == 1.0
