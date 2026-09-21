"""Gradient Boosted Decision Tree models (LightGBM, CatBoost, HistGradientBoosting)."""

from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.tabular_builder import TabularFeaturePipeline

# Optional imports with graceful fallback
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class TabularGBDTRegressor:
    """Unified GBDT regressor for catalog tabular and engineered features.

    Supports:
        - LightGBM (if installed)
        - CatBoost (if installed)
        - HistGradientBoostingRegressor (scikit-learn built-in, zero external dependencies)
    """

    def __init__(
        self,
        engine: str = "auto",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        target_log: bool = True,
        min_clip: float = 1e-3,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.engine = engine.lower()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.target_log = target_log
        self.min_clip = min_clip
        self.random_state = random_state
        self.extra_kwargs = kwargs

        self.feature_pipeline = TabularFeaturePipeline()
        self.model: Any = None
        self.active_engine_: str = ""
        self.is_fitted_ = False

    def _initialize_model(self) -> None:
        """Resolve model engine according to environment availability."""
        if self.engine == "lightgbm" or (self.engine == "auto" and HAS_LIGHTGBM):
            if not HAS_LIGHTGBM:
                raise ImportError("lightgbm is not installed. Install via pip install lightgbm.")
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
                **self.extra_kwargs,
            )
            self.active_engine_ = "lightgbm"

        elif self.engine == "catboost" or (
            self.engine == "auto" and HAS_CATBOOST and not HAS_LIGHTGBM
        ):
            if not HAS_CATBOOST:
                raise ImportError("catboost is not installed. Install via pip install catboost.")
            self.model = cb.CatBoostRegressor(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.max_depth,
                random_seed=self.random_state,
                verbose=False,
                **self.extra_kwargs,
            )
            self.active_engine_ = "catboost"

        else:
            # Universal scikit-learn HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.extra_kwargs,
            )
            self.active_engine_ = "hist_gbdt"

    def fit(self, df: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> "TabularGBDTRegressor":
        """Fit feature pipeline and tree regressor."""
        self._initialize_model()

        X = self.feature_pipeline.fit_transform(df)
        y_arr = np.asarray(y, dtype=float)
        y_train = np.log1p(np.maximum(y_arr, 0.0)) if self.target_log else y_arr

        self.model.fit(X, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features and predict targets."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict.")

        X = self.feature_pipeline.transform(df)
        raw_preds = self.model.predict(X)

        preds = np.expm1(raw_preds) if self.target_log else raw_preds
        clipped = np.clip(preds, self.min_clip, None)
        return np.asarray(clipped, dtype=float)
