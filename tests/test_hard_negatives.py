"""Unit tests for HardNegativeMiner."""

import numpy as np
import pytest
from src.training.hard_negative_miner import HardNegativeMiner
from src.features.pairwise_features import precompute_record_views


def test_hard_negative_mining():
    s1_records = [
        {"entity_id": "S1-1", "business_name": "Acme Tools", "business_address": "100 Main St, Austin, TX 78701", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Beta Services", "business_address": "200 Market St, Dallas, TX 75001", "country": "US"},
    ]
    target_records = [
        {"entity_id": "S2-1", "business_name": "Acme Tools", "business_address": "100 Main St, Austin, TX 78701", "country": "US"},  # TRUE MATCH
        {"entity_id": "S2-2", "business_name": "Acme Tools", "business_address": "999 Other St, Seattle, WA 98101", "country": "US"}, # Same name / diff addr
        {"entity_id": "S3-3", "business_name": "Different Biz", "business_address": "100 Main St, Austin, TX 78701", "country": "US"}, # Same addr / diff name
        {"entity_id": "S2-4", "business_name": "Acme Store", "business_address": "105 Main St, Austin, TX 78701", "country": "US"},  # Nearby house num
    ]

    s1_pre = {r["entity_id"]: precompute_record_views(r) for r in s1_records}
    tgt_pre = {r["entity_id"]: precompute_record_views(r) for r in target_records}

    ground_truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    candidate_pairs = [
        ("S1-1", "S2-1"),
        ("S1-1", "S2-2"),
        ("S1-1", "S3-3"),
        ("S1-1", "S2-4"),
    ]

    miner = HardNegativeMiner(max_negatives_per_positive=5, random_state=42)
    hard_negs = miner.mine_hard_negatives(
        candidate_pairs,
        ground_truth,
        s1_pre,
        tgt_pre,
    )

    # True positive ("S1-1", "S2-1") must NOT be in hard negatives
    assert ("S1-1", "S2-1") not in hard_negs
    assert len(hard_negs) > 0
    # Must include difficult look-alikes
    assert any(tid in ("S2-2", "S3-3", "S2-4") for _, tid in hard_negs)
