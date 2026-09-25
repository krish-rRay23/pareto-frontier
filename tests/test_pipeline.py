"""End-to-end integration test for the Entity Resolution pipeline."""

import pytest
from src.inference.pipeline import EntityResolutionPipeline


def test_entity_resolution_pipeline_end_to_end(tmp_path):
    """Verify that fit_and_validate and predict_test_and_export execute cleanly."""
    s1_records = [
        {"entity_id": "S1-001", "business_name": "Novent Owl PLLC", "business_address": "9308 Home Ct, Des Plaines, IL", "country": "US"},
        {"entity_id": "S1-002", "business_name": "Prime Money Inc", "business_address": "17560 Ellis Road, Tahlequah, OK", "country": "US"},
        {"entity_id": "S1-003", "business_name": "Unique Singleton Company", "business_address": "999 Lone Road, Austin, TX", "country": "US"},
    ]
    target_records = [
        {"entity_id": "S2-101", "business_name": "Novent Owl LLC", "business_address": "9308 Home Court, Des Plaines, IL", "country": "US"},
        {"entity_id": "S3-102", "business_name": "Novent [Owl]", "business_address": "9308 Home Ct, Des Plaines, Illinois", "country": "US"},
        {"entity_id": "S2-201", "business_name": "Prime Money", "business_address": "17560 Ellis Rd, Tahlequah, OK", "country": "US"},
        {"entity_id": "S2-999", "business_name": "Irrelevant Shop", "business_address": "1 Main St, Dallas, TX", "country": "US"},
    ]
    ground_truth = {
        "S1-001": {"S2-101", "S3-102"},
        "S1-002": {"S2-201"},
        "S1-003": set(),  # Singleton
    }

    pipeline = EntityResolutionPipeline(
        max_candidates_per_entity=20,
        enable_consensus_features=True,
        lgb_params={"n_estimators": 10, "min_child_samples": 1},
    )

    metrics = pipeline.fit_and_validate(
        s1_records=s1_records,
        target_records=target_records,
        ground_truth=ground_truth,
        n_splits=2,
        optimize_policy=True,
        verbose=False,
    )

    assert "macro_f05" in metrics
    assert "candidate_recall" in metrics
    assert metrics["candidate_recall"] == 1.0  # Perfect blocker recall on this synthetic set

    # Test inference export
    out_dir = str(tmp_path / "output")
    matching_path, candidate_path = pipeline.predict_test_and_export(
        test_s1_records=s1_records,
        test_target_records=target_records,
        output_dir=out_dir,
        validate=False,
        verbose=False,
    )

    import os
    assert os.path.exists(matching_path)
    assert os.path.exists(candidate_path)
