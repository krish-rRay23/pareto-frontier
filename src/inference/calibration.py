"""Probability calibration modules: Isotonic Regression and Platt Scaling."""

from typing import Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Calibrates model probability outputs using Isotonic Regression or Platt Scaling."""

    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        elif self.method == "platt":
            self.model = LogisticRegression(solver="lbfgs", random_state=42)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        self.is_fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibrator on validation out-of-fold probabilities."""
        probs = np.asarray(probs, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        if len(np.unique(y_true)) < 2:
            self.is_fitted = False
            return self

        if self.method == "isotonic":
            self.model.fit(probs, y_true)
        elif self.method == "platt":
            self.model.fit(probs.reshape(-1, 1), y_true)

        self.is_fitted = True
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities into calibrated probabilities."""
        if not self.is_fitted:
            return np.asarray(probs, dtype=float)

        probs = np.asarray(probs, dtype=float)
        if self.method == "isotonic":
            return np.clip(self.model.predict(probs), 0.0, 1.0)
        elif self.method == "platt":
            return self.model.predict_proba(probs.reshape(-1, 1))[:, 1]
        return probs
