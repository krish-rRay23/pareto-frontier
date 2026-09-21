"""Validation metrics module with robust SMAPE and standard regression metrics."""

from typing import Dict, Union

import numpy as np
import pandas as pd


def compute_smape(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    eps: float = 1e-8,
) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE) in percentage [0, 200].

    Formula:
        SMAPE = (100 / n) * sum( 2 * |y_true - y_pred| / (|y_true| + |y_pred|) )

    Edge Cases:
        - When both y_true and y_pred are 0 (or close to 0 within eps), error is defined as 0.0.
        - Negative values are protected via absolute values.
        - Returns a Python float.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if y_t.shape != y_p.shape:
        raise ValueError(
            f"Shape mismatch in SMAPE: y_true shape {y_t.shape} != y_pred shape {y_p.shape}"
        )

    if y_t.size == 0:
        return 0.0

    numerator = 2.0 * np.abs(y_p - y_t)
    denominator = np.abs(y_t) + np.abs(y_p)

    # Identically zero or both under epsilon threshold -> error is 0.0
    zero_mask = denominator < eps

    ratio = np.zeros_like(numerator)
    ratio[~zero_mask] = numerator[~zero_mask] / denominator[~zero_mask]

    smape_val = float(np.mean(ratio) * 100.0)
    return smape_val


def compute_mae(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Mean Absolute Error (MAE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def compute_rmse(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def compute_all_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """Compute all evaluation metrics and return summary dictionary."""
    return {
        "smape": compute_smape(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
    }
