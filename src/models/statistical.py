"""Simple statistical baseline model (Median / Mean with optional category grouping)."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class StatisticalBaselineRegressor:
    """Predicts training target median or mean (overall or grouped by category).

    Serves as the absolute sanity check lower-bound baseline for any ML competition.
    """

    def __init__(self, aggregation: str = "median", group_col: Optional[str] = None):
        if aggregation not in ["median", "mean"]:
            raise ValueError(f"Unknown aggregation '{aggregation}'. Choose 'median' or 'mean'.")
        self.aggregation = aggregation
        self.group_col = group_col
        self.global_val_: float = 0.0
        self.group_vals_: Dict[str, float] = {}

    def fit(
        self, df: pd.DataFrame, y: Union[np.ndarray, pd.Series]
    ) -> "StatisticalBaselineRegressor":
        """Compute global and group-level aggregation on training set."""
        y_arr = np.asarray(y, dtype=float)
        self.global_val_ = float(
            np.median(y_arr) if self.aggregation == "median" else np.mean(y_arr)
        )

        self.group_vals_ = {}
        if self.group_col and self.group_col in df.columns:
            temp_df = pd.DataFrame(
                {self.group_col: df[self.group_col].astype(str), "target": y_arr}
            )
            grouped = temp_df.groupby(self.group_col)["target"]
            agg_series = grouped.median() if self.aggregation == "median" else grouped.mean()
            self.group_vals_ = agg_series.to_dict()

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict median/mean for input dataframe."""
        if not self.group_col or self.group_col not in df.columns:
            return np.full(len(df), self.global_val_, dtype=float)

        groups = df[self.group_col].astype(str)
        preds = groups.map(self.group_vals_).fillna(self.global_val_).to_numpy(dtype=float)
        return preds
