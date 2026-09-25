"""Pairwise matching classifiers: LightGBM GBDT and deterministic baseline."""

import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBMPairClassifier:
    """LightGBM Binary Classifier for Entity Resolution Candidate Pairs."""

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
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
            "objective": kwargs.get("objective", "binary"),
            "metric": kwargs.get("metric", "binary_logloss"),
            "boosting_type": kwargs.get("boosting_type", "gbdt"),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        self.params.update(kwargs)
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        verbose: bool = False,
    ) -> "LightGBMPairClassifier":
        """Fit the LightGBM model with optional early stopping on validation split."""
        self.feature_names_ = list(X_train.columns)
        unique_classes = np.unique(y_train)

        # Edge case: if training fold has only one class (e.g. in tiny synthetic unit tests)
        if len(unique_classes) < 2:
            self.model = None
            self.constant_pred_ = float(unique_classes[0]) if len(unique_classes) == 1 else 0.0
            return self

        self.constant_pred_ = None
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = []
        # Early stopping requires both classes to be present in validation set
        if (
            X_val is not None
            and y_val is not None
            and len(np.unique(y_val)) >= 2
            and early_stopping_rounds > 0
        ):
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of true match for candidate pairs."""
        if getattr(self, "constant_pred_", None) is not None:
            return np.full(len(X), self.constant_pred_, dtype=float)
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return mapping of feature names to their importance values."""
        if self.model is None:
            return {}
        return dict(zip(self.feature_names_, self.model.feature_importances_.astype(float)))

    def save(self, filepath: str) -> None:
        """Save model checkpoint to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names_, "params": self.params}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LightGBMPairClassifier":
        """Load model checkpoint from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.params = data["params"]
        instance.feature_names_ = data["feature_names"]
        instance.model = data["model"]
        return instance


class CatBoostPairClassifier:
    """CatBoost Binary Classifier for Entity Resolution Candidate Pairs."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        early_stopping_rounds: int = 30,
        verbose: int = 0,
        **kwargs,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": random_seed,
            "verbose": verbose,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
        }
        self.params.update(kwargs)
        self.model = None
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
    ) -> "CatBoostPairClassifier":
        """Fit CatBoost classifier on candidate pair features."""
        from catboost import CatBoostClassifier

        self.feature_names_ = list(X_train.columns)
        unique_classes = np.unique(y_train)

        if len(unique_classes) < 2:
            self.model = None
            self.constant_pred_ = float(unique_classes[0]) if len(unique_classes) == 1 else 0.0
            return self

        self.constant_pred_ = None
        params = dict(self.params)
        if verbose:
            params["verbose"] = 100
        else:
            params["verbose"] = 0

        self.model = CatBoostClassifier(**params)

        eval_set = None
        if (
            X_val is not None
            and y_val is not None
            and len(np.unique(y_val)) >= 2
            and early_stopping_rounds > 0
        ):
            eval_set = (X_val, y_val)
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=params["verbose"],
            )
        else:
            self.model.fit(X_train, y_train, verbose=params["verbose"])

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict matching probability for candidate pairs."""
        if self.constant_pred_ is not None:
            return np.full(len(X), self.constant_pred_, dtype=float)
        if self.model is None:
            raise RuntimeError("CatBoost model has not been fitted.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importances."""
        if self.model is None:
            return {}
        return dict(zip(self.feature_names_, self.model.get_feature_importance().astype(float)))

    def save(self, filepath: str) -> None:
        """Save CatBoost model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names_, "params": self.params}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CatBoostPairClassifier":
        """Load CatBoost model checkpoint."""
        data = joblib.load(filepath)
        instance = cls()
        instance.params = data["params"]
        instance.feature_names_ = data["feature_names"]
        instance.model = data["model"]
        return instance


class BaselineDeterministicMatcher:
    """E0 Deterministic Baseline: matches on exact name and address anchors."""

    def __init__(self, min_name_jaccard: float = 0.85, require_number_match: bool = True):
        self.min_name_jaccard = min_name_jaccard
        self.require_number_match = require_number_match

    def predict_pair_features(self, X: pd.DataFrame) -> np.ndarray:
        """Score candidate pairs deterministically using feature matrix rules."""
        is_exact_name = (X["exact_legal_name"] == 1.0) | (X["name_token_jaccard"] >= self.min_name_jaccard)
        if self.require_number_match and "addr_number_match" in X.columns:
            # Address number must match unless address is missing
            num_ok = (X["addr_number_match"] == 1.0) | (X["addr_missing_indicator"] == 1.0)
            preds = (is_exact_name & num_ok).astype(float)
        else:
            preds = is_exact_name.astype(float)
        return preds.values

