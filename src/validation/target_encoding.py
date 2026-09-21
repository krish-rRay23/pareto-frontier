"""Leakage-safe Out-of-Fold (OOF) Target Encoding."""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


class LeakageSafeTargetEncoder:
    """Smoothed target encoder with strict out-of-fold calculation to prevent leakage.

    Uses m-estimate smoothing:
        encoded_value = (count * category_mean + smoothing * global_mean) / (count + smoothing)
    """

    def __init__(
        self, categorical_cols: List[str], smoothing: float = 10.0, target_log: bool = False
    ):
        self.categorical_cols = categorical_cols
        self.smoothing = smoothing
        self.target_log = target_log
        self.encodings_: Dict[str, Dict[Union[str, int], float]] = {}
        self.global_means_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> "LeakageSafeTargetEncoder":
        """Fit target encodings strictly on given training data."""
        y_vals = (
            np.log1p(np.asarray(y, dtype=float)) if self.target_log else np.asarray(y, dtype=float)
        )
        global_mean = float(np.mean(y_vals))

        self.encodings_ = {}
        self.global_means_ = {}

        for col in self.categorical_cols:
            self.global_means_[col] = global_mean
            col_series = df[col].astype(str)

            stats = (
                pd.DataFrame({"cat": col_series, "y": y_vals})
                .groupby("cat")["y"]
                .agg(["count", "mean"])
            )
            smoothed = (stats["count"] * stats["mean"] + self.smoothing * global_mean) / (
                stats["count"] + self.smoothing
            )

            self.encodings_[col] = smoothed.to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test or validation features using previously learned encodings."""
        df_out = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            mapping = self.encodings_.get(col, {})
            prior = self.global_means_.get(col, 0.0)
            col_series = df[col].astype(str)
            encoded = col_series.map(mapping).fillna(prior).astype(float)
            df_out[f"{col}_te"] = encoded
        return df_out

    def fit_transform_oof(
        self,
        df: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[pd.DataFrame, "LeakageSafeTargetEncoder"]:
        """Compute Out-of-Fold target encodings for training data without fold leakage.

        Returns:
            oof_df: DataFrame of shape (len(df), len(categorical_cols)) containing OOF encodings.
            final_fitted_encoder: Encoder fit on full training set (ready to transform test set).
        """
        oof_df = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            oof_df[f"{col}_te"] = np.nan

        # Fill validation folds strictly using models fit on training fold
        for train_idx, val_idx in folds:
            fold_tr_df = df.iloc[train_idx]
            fold_tr_y = np.asarray(y)[train_idx]

            fold_val_df = df.iloc[val_idx]

            fold_encoder = LeakageSafeTargetEncoder(
                categorical_cols=self.categorical_cols,
                smoothing=self.smoothing,
                target_log=self.target_log,
            )
            fold_encoder.fit(fold_tr_df, fold_tr_y)
            fold_encoded_val = fold_encoder.transform(fold_val_df)

            for col in self.categorical_cols:
                oof_df.loc[oof_df.index[val_idx], f"{col}_te"] = fold_encoded_val[f"{col}_te"]

        # Fit final encoder on all training data for future test transforms
        self.fit(df, y)
        return oof_df, self
