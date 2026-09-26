"""XGBoost Pairwise Classifier for Entity Resolution."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostPairClassifier:
    """XGBoost Binary Classifier for Entity Resolution Candidate Pairs."""

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        early_stopping_rounds: int = 30,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {
            "objective": kwargs.get("objective", "binary:logistic"),
            "eval_metric": kwargs.get("eval_metric", "logloss"),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": kwargs.get("tree_method", "hist"),
        }
        self.params.update(kwargs)
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names_: List[str] = []
        self.constant_pred_: Optional[float] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        verbose: bool = False,
    ) -> "XGBoostPairClassifier":
        """Fit the XGBoost model with optional early stopping."""
        self.feature_names_ = list(X_train.columns)
        unique_classes = np.unique(y_train)

        if len(unique_classes) < 2:
            self.model = None
            self.constant_pred_ = float(unique_classes[0]) if len(unique_classes) == 1 else 0.0
            return self

        self.constant_pred_ = None
        fit_params = {}
        if (
            X_val is not None
            and y_val is not None
            and len(np.unique(y_val)) >= 2
            and early_stopping_rounds > 0
        ):
            self.params["early_stopping_rounds"] = early_stopping_rounds
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = verbose

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities."""
        if self.constant_pred_ is not None:
            return np.full(len(X), self.constant_pred_, dtype=float)
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        return self.model.predict_proba(X)[:, 1]
