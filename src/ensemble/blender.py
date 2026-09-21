"""Out-of-fold blending and ensembling algorithms."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.validation.metrics import compute_smape


class OptimalLinearBlender:
    """Optimizes non-negative blending weights to directly minimize CV SMAPE."""

    def __init__(self, metric: str = "smape"):
        self.metric = metric
        self.weights_: Optional[np.ndarray] = None

    def fit(
        self,
        oof_preds_list: List[Union[np.ndarray, pd.Series]],
        y_true: Union[np.ndarray, pd.Series],
    ) -> "OptimalLinearBlender":
        """Find optimal weights w >= 0 such that sum(w) = 1 minimizing validation SMAPE."""
        # Matrix shape: (n_samples, n_models)
        preds_mat = np.column_stack([np.asarray(p, dtype=float) for p in oof_preds_list])
        y = np.asarray(y_true, dtype=float)
        n_models = preds_mat.shape[1]

        def objective(weights: np.ndarray) -> float:
            blend = np.dot(preds_mat, weights)
            return compute_smape(y, blend)

        init_weights = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-6},
        )

        self.weights_ = res.x / np.sum(res.x)
        return self

    def predict(self, test_preds_list: List[Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """Blend test model predictions using learned weights."""
        if self.weights_ is None:
            raise RuntimeError("Blender must be fitted before predict.")

        preds_mat = np.column_stack([np.asarray(p, dtype=float) for p in test_preds_list])
        blend = np.dot(preds_mat, self.weights_)
        return blend


def simple_average_blend(preds_list: List[np.ndarray]) -> np.ndarray:
    """Simple arithmetic mean of multiple model predictions."""
    return np.mean(preds_list, axis=0)


def median_blend(preds_list: List[np.ndarray]) -> np.ndarray:
    """Median of multiple model predictions, robust to model outliers."""
    return np.median(preds_list, axis=0)
