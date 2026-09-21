"""Unit tests for validation metrics and SMAPE edge cases."""

import numpy as np
import pytest

from src.validation.metrics import compute_all_metrics, compute_smape


def test_smape_identical():
    """SMAPE should be exactly 0% when predictions match ground truth."""
    y = np.array([10.0, 50.0, 100.0, 500.0])
    assert compute_smape(y, y) == pytest.approx(0.0, abs=1e-6)


def test_smape_zero_pairs():
    """When both ground truth and prediction are 0, error must be 0.0 without division by zero."""
    y_true = np.array([0.0, 0.0, 100.0])
    y_pred = np.array([0.0, 0.0, 100.0])
    assert compute_smape(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)


def test_smape_known_values():
    """Validate hand-calculated SMAPE values."""
    # y_true=100, y_pred=200 -> 2*|200-100|/(100+200) = 200/300 = 66.6667%
    y_t = [100.0]
    y_p = [200.0]
    expected = (200.0 / 300.0) * 100.0
    assert compute_smape(y_t, y_p) == pytest.approx(expected, rel=1e-4)


def test_smape_shape_mismatch():
    """Ensure shape mismatch raises ValueError."""
    with pytest.raises(ValueError):
        compute_smape([1.0, 2.0], [1.0])


def test_compute_all_metrics():
    """Ensure compute_all_metrics returns correct keys and valid numbers."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    metrics = compute_all_metrics(y_true, y_pred)

    assert "smape" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["mae"] == pytest.approx(4.0 / 3.0, abs=1e-4)
    assert metrics["rmse"] == pytest.approx(np.sqrt(8.0 / 3.0), abs=1e-4)
