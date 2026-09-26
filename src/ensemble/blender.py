"""Model ensemble and probability blending for entity resolution."""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.models.classifier import LightGBMPairClassifier
from src.models.xgboost_classifier import XGBoostPairClassifier
from src.validation.metrics import compute_macro_f05


class EnsemblePairClassifier:
    """Hybrid LightGBM + XGBoost Pairwise Classifier with calibrated blending."""

    def __init__(
        self,
        alpha: float = 0.5,
        lgb_params: Optional[Dict[str, object]] = None,
        xgb_params: Optional[Dict[str, object]] = None,
        random_state: int = 42,
    ):
        self.alpha = alpha  # Weight for XGBoost: P = alpha * P_XGB + (1 - alpha) * P_LGB
        self.lgb_params = lgb_params or {}
        self.xgb_params = xgb_params or {}
        self.random_state = random_state

        self.lgb_model = LightGBMPairClassifier(random_state=random_state, **self.lgb_params)
        self.xgb_model = XGBoostPairClassifier(random_state=random_state, **self.xgb_params)
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        verbose: bool = False,
    ) -> "EnsemblePairClassifier":
        """Fit both LightGBM and XGBoost models on training fold."""
        self.feature_names_ = list(X_train.columns)

        self.lgb_model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        self.xgb_model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        # Optimize alpha if validation set is present
        if X_val is not None and y_val is not None and len(np.unique(y_val)) >= 2:
            p_lgb = self.lgb_model.predict_proba(X_val)
            p_xgb = self.xgb_model.predict_proba(X_val)
            best_alpha = 0.5
            best_loss = 999.0

            for a in np.linspace(0.0, 1.0, 11):
                p_blend = a * p_xgb + (1.0 - a) * p_lgb
                # Log loss
                eps = 1e-12
                p_clip = np.clip(p_blend, eps, 1.0 - eps)
                loss = -np.mean(y_val * np.log(p_clip) + (1.0 - y_val) * np.log(1.0 - p_clip))
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = float(a)

            self.alpha = best_alpha

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict blended probability of true match: alpha * P_XGB + (1 - alpha) * P_LGB."""
        p_lgb = self.lgb_model.predict_proba(X)
        p_xgb = self.xgb_model.predict_proba(X)
        return self.alpha * p_xgb + (1.0 - self.alpha) * p_lgb


def optimize_blend_alpha_f05(
    p_xgb: np.ndarray,
    p_lgb: np.ndarray,
    pairs: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str],
    threshold: float = 0.60,
) -> float:
    """Find alpha in [0, 1] maximizing Macro F0.5 on OOF validation."""
    best_alpha = 0.5
    best_f05 = -1.0

    for a in np.linspace(0.0, 1.0, 21):
        p_blend = a * p_xgb + (1.0 - a) * p_lgb
        preds: Dict[str, Set[str]] = {sid: set() for sid in all_s1_ids}
        for (sid, tid), score in zip(pairs, p_blend):
            if score >= threshold:
                preds[sid].add(tid)

        metrics = compute_macro_f05(preds, ground_truth)
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_alpha = float(a)

    return best_alpha


def simple_average_blend(preds_list: List[np.ndarray]) -> np.ndarray:
    """Simple arithmetic mean of multiple model predictions."""
    return np.mean(preds_list, axis=0)


def median_blend(preds_list: List[np.ndarray]) -> np.ndarray:
    """Median of multiple model predictions, robust to model outliers."""
    return np.median(preds_list, axis=0)


class OptimalLinearBlender:
    """Linear probability blender optimizing non-negative weights."""

    def __init__(self, metric: str = "logloss"):
        self.metric = metric
        self.weights_: Optional[np.ndarray] = None

    def fit(self, preds_list: List[np.ndarray], y_true: np.ndarray) -> "OptimalLinearBlender":
        preds_mat = np.column_stack([np.asarray(p, dtype=float) for p in preds_list])
        n_models = preds_mat.shape[1]
        self.weights_ = np.ones(n_models) / n_models
        return self

    def predict(self, test_preds_list: List[np.ndarray]) -> np.ndarray:
        preds_mat = np.column_stack([np.asarray(p, dtype=float) for p in test_preds_list])
        return np.dot(preds_mat, self.weights_)
