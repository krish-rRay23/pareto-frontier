"""Fast text baseline using TF-IDF and Ridge Regression."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


class TfidfRidgeRegressor:
    """TF-IDF + Ridge regression model on combined catalog text fields.

    Features:
        - Multi-column text concatenation.
        - Sublinear TF scaling with word unigrams and bigrams.
        - Log1p target transformation to directly stabilize percentage metrics (e.g. SMAPE).
        - Non-negative clipping for physical targets (e.g. prices or quantities).
    """

    def __init__(
        self,
        text_cols: Optional[List[str]] = None,
        max_features: int = 25000,
        ngram_range: tuple = (1, 2),
        alpha: float = 1.0,
        target_log: bool = True,
        min_clip: float = 1e-3,
        random_state: int = 42,
    ):
        self.text_cols = text_cols or ["catalog_title", "bullet_points", "description"]
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.target_log = target_log
        self.min_clip = min_clip
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            min_df=2,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.is_fitted_ = False

    def _prepare_text(self, df: pd.DataFrame) -> List[str]:
        """Combine specified text columns row-by-row into a single clean string."""
        cols = [c for c in self.text_cols if c in df.columns]
        if not cols:
            # Fallback: search for any object/string columns
            cols = list(df.select_dtypes(include=["object", "string"]).columns)
            if not cols:
                return [""] * len(df)

        combined = df[cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
        return combined

    def fit(self, df: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> "TfidfRidgeRegressor":
        """Fit vectorizer and Ridge model on training catalog text."""
        texts = self._prepare_text(df)
        X = self.vectorizer.fit_transform(texts)

        y_arr = np.asarray(y, dtype=float)
        if self.target_log:
            y_train = np.log1p(np.maximum(y_arr, 0.0))
        else:
            y_train = y_arr

        self.model.fit(X, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict continuous targets from catalog text."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict.")

        texts = self._prepare_text(df)
        X = self.vectorizer.transform(texts)
        raw_preds = self.model.predict(X)

        if self.target_log:
            preds = np.expm1(raw_preds)
        else:
            preds = raw_preds

        # Enforce non-negative physical bounds
        clipped_preds = np.clip(preds, self.min_clip, None)
        return np.asarray(clipped_preds, dtype=float)
