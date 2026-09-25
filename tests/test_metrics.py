"""Unit tests for ML Challenge 2026 Macro-Averaged F0.5 metric."""

import pytest
from src.validation.metrics import (
    compute_f_beta,
    compute_entity_f05,
    compute_macro_f05,
)


def test_f05_official_example():
    """Verify official problem statement example:
    Predicted: [S2-00047, S2-00193, S3-00812]
    True:      [S2-00047, S3-00812]
    Precision = 2/3, Recall = 1.0 -> F0.5 = 0.7142857...
    """
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    res = compute_entity_f05(true_set, pred_set)

    assert res["precision"] == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert res["recall"] == pytest.approx(1.0, abs=1e-5)
    assert res["f05"] == pytest.approx(0.7142857, abs=1e-5)


def test_singleton_correct():
    """Singleton correctly predicted as empty list must score 1.0."""
    res = compute_entity_f05(set(), set())
    assert res["f05"] == 1.0
    assert res["fp"] == 0


def test_singleton_false_merge():
    """Singleton predicted with a false candidate must score 0.0."""
    res = compute_entity_f05(set(), {"S2-99999"})
    assert res["f05"] == 0.0
    assert res["fp"] == 1


def test_perfect_prediction():
    """Perfect match must score 1.0."""
    true_set = {"S2-1", "S3-2"}
    pred_set = {"S2-1", "S3-2"}
    res = compute_entity_f05(true_set, pred_set)
    assert res["f05"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_macro_f05_average():
    """Verify macro average across mixed entities (singletons and multi-matches)."""
    gt = {
        "S1-1": {"S2-10", "S3-20"},  # Perfect -> 1.0
        "S1-2": set(),               # Correct singleton -> 1.0
        "S1-3": set(),               # Failed singleton (FP) -> 0.0
        "S1-4": {"S2-30"},           # Missed match (FN) -> 0.0
    }
    preds = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),
        "S1-3": {"S2-99"},
        "S1-4": set(),
    }
    meta = {
        "S1-1": {"country": "US"},
        "S1-2": {"country": "US"},
        "S1-3": {"country": "India"},
        "S1-4": {"country": "India"},
    }
    summary = compute_macro_f05(gt, preds, metadata=meta)

    # Average: (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.50
    assert summary["macro_f05"] == pytest.approx(0.50, abs=1e-5)
    assert summary["macro_f05_us"] == pytest.approx(1.0, abs=1e-5)
    assert summary["macro_f05_india"] == pytest.approx(0.0, abs=1e-5)
    assert summary["singleton_accuracy"] == pytest.approx(0.50, abs=1e-5)
    assert summary["false_merges"] == 1
