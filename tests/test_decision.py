"""Unit tests for decision policies, Expected-F0.5 decoder, exclusivity, and calibration."""

import numpy as np
import pytest
from src.inference.calibration import ProbabilityCalibrator
from src.inference.decision_policy import (
    PrecisionDecisionPolicy,
    ExpectedF05DecisionDecoder,
    TargetExclusivityResolver,
    ContradictionChecker,
)


def test_probability_calibrator():
    probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    y_true = np.array([0, 0, 0, 1, 1, 1])

    calib = ProbabilityCalibrator(method="isotonic")
    calib.fit(probs, y_true)
    cal_probs = calib.predict(probs)

    assert len(cal_probs) == len(probs)
    assert np.all(cal_probs >= 0.0)
    assert np.all(cal_probs <= 1.0)


def test_expected_f05_decoder():
    pairs = [
        ("S1-1", "S2-10"),
        ("S1-1", "S3-20"),
        ("S1-2", "S2-99"),  # weak evidence
    ]
    scores = np.array([0.95, 0.88, 0.15])
    decoder = ExpectedF05DecisionDecoder(singleton_prior_threshold=0.40)
    preds = decoder.apply(pairs, scores, all_s1_ids=["S1-1", "S1-2"])

    assert "S2-10" in preds["S1-1"]
    assert "S3-20" in preds["S1-1"]
    # S1-2 should abstain (empty) because top score is 0.15 < 0.40
    assert len(preds["S1-2"]) == 0


def test_target_exclusivity_resolver():
    # Both S1-1 and S1-2 claim target S2-10, but S1-1 has higher confidence (0.95 vs 0.70)
    predictions = {
        "S1-1": {"S2-10"},
        "S1-2": {"S2-10"},
    }
    pairs = [("S1-1", "S2-10"), ("S1-2", "S2-10")]
    scores = np.array([0.95, 0.70])

    resolver = TargetExclusivityResolver()
    resolved = resolver.resolve(predictions, pairs, scores)

    assert "S2-10" in resolved["S1-1"]
    assert "S2-10" not in resolved["S1-2"]


def test_contradiction_checker():
    predictions = {
        "S1-1": {"S2-US", "S2-INDIA"},
    }
    s1_pre = {"S1-1": {"country": "us", "addr": {"first_num": "100", "first_salient": "main"}}}
    tgt_pre = {
        "S2-US": {"country": "us", "addr": {"first_num": "100", "first_salient": "main"}},
        "S2-INDIA": {"country": "in", "addr": {"first_num": "100", "first_salient": "main"}},
    }

    checker = ContradictionChecker(strict_country=True)
    filtered = checker.filter_predictions(predictions, s1_pre, tgt_pre)

    assert "S2-US" in filtered["S1-1"]
    assert "S2-INDIA" not in filtered["S1-1"]
