"""Unit tests for pairwise and contextual feature extraction."""

import pytest
from src.features.pairwise_features import (
    FEATURE_COLUMNS,
    extract_pair_features,
    precompute_record_views,
    jaccard_similarity,
    sequence_match_ratio,
    jaro_winkler_similarity,
)
from src.features.context import compute_context_and_competition_features
from src.blocking.retrieval import RetrievalProvenance, CH_EXACT_NAME, CH_EXACT_ADDR


def test_jaccard_similarity():
    assert jaccard_similarity({"apple", "banana"}, {"apple", "banana"}) == 1.0
    assert jaccard_similarity({"apple"}, {"orange"}) == 0.0
    assert jaccard_similarity(set(), set()) == 1.0


def test_jaro_winkler_similarity():
    assert jaro_winkler_similarity("Novent Owl", "Novent Owl") == 1.0
    assert jaro_winkler_similarity("Dixon", "Dicksonx") > 0.8
    assert jaro_winkler_similarity("", "Test") == 0.0


def test_pairwise_feature_extraction():
    s1 = {"entity_id": "S1-1", "business_name": "Novent Owl PLLC", "business_address": "9308 Home Court, Des Plaines, IL 60016", "country": "US"}
    t1 = {"entity_id": "S2-1", "business_name": "Novent Owl LLC", "business_address": "9308 Home Ct, Des Plaines, IL 60016", "country": "US"}

    s1_pv = precompute_record_views(s1)
    t1_pv = precompute_record_views(t1)

    prov = RetrievalProvenance()
    prov.add_hit(CH_EXACT_NAME, rank=0, score=1.0)
    prov.add_hit(CH_EXACT_ADDR, rank=1, score=0.9)
    prov.finalize()

    feats = extract_pair_features(s1, t1, s1_pv, t1_pv, prov)
    assert len(feats) == len(FEATURE_COLUMNS)
    assert len(feats) == 86

    # exact_legal_name is index 2
    assert feats[2] == 1.0
    # postal_exact_match is index 33
    assert feats[33] == 1.0


def test_context_and_competition_features():
    pairs = [("S1-1", "S2-1"), ("S1-1", "S3-2"), ("S1-2", "S2-1")]
    s1 = {"entity_id": "S1-1", "business_name": "Test A", "business_address": "123 Main St", "country": "US"}
    s2 = {"entity_id": "S1-2", "business_name": "Test B", "business_address": "456 Market St", "country": "US"}
    t1 = {"entity_id": "S2-1", "business_name": "Test A Corp", "business_address": "123 Main St", "country": "US"}
    t2 = {"entity_id": "S3-2", "business_name": "Test A Ltd", "business_address": "123 Main St", "country": "US"}

    s1_pre = {"S1-1": precompute_record_views(s1), "S1-2": precompute_record_views(s2)}
    tgt_pre = {"S2-1": precompute_record_views(t1), "S3-2": precompute_record_views(t2)}

    ctx_df = compute_context_and_competition_features(pairs, s1_pre, tgt_pre)
    assert len(ctx_df) == 3
    assert "s1_candidate_density" in ctx_df.columns
    assert "target_competition_count" in ctx_df.columns
    # S2-1 was claimed by both S1-1 and S1-2
    assert ctx_df.loc[0, "target_competition_count"] == 2.0
