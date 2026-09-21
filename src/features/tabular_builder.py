"""Tabular dataset builder combining catalog extraction and encoding."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.features.text_extraction import extract_features_dataframe


class TabularFeaturePipeline:
    """End-to-end feature pipeline that transforms catalog dataframe to numeric matrix."""

    def __init__(
        self,
        title_col: str = "catalog_title",
        desc_col: str = "description",
        bullets_col: str = "bullet_points",
        categorical_cols: Optional[List[str]] = None,
    ):
        self.title_col = title_col
        self.desc_col = desc_col
        self.bullets_col = bullets_col
        self.categorical_cols = categorical_cols or [
            "category",
            "brand_candidate",
            "extracted_unit",
        ]
        self.numeric_cols_: List[str] = []
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.medians_: dict = {}
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame) -> "TabularFeaturePipeline":
        """Fit preprocessing transformations on training dataframe."""
        extracted = extract_features_dataframe(
            df,
            title_col=self.title_col,
            desc_col=self.desc_col,
            bullets_col=self.bullets_col,
        )

        full_df = pd.concat([df, extracted], axis=1)

        # Identify numeric features
        num_candidates = [
            c
            for c in extracted.columns
            if c not in self.categorical_cols and np.issubdtype(extracted[c].dtype, np.number)
        ]
        self.numeric_cols_ = num_candidates

        # Compute median imputation values
        for col in self.numeric_cols_:
            self.medians_[col] = (
                float(full_df[col].median(skipna=True)) if not full_df[col].dropna().empty else 0.0
            )

        # Fit ordinal encoding for categoricals
        cat_present = [c for c in self.categorical_cols if c in full_df.columns]
        if cat_present:
            cat_df = full_df[cat_present].fillna("MISSING").astype(str)
            self.encoder_.fit(cat_df)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform catalog dataframe into numeric feature matrix."""
        if not self.is_fitted_:
            raise RuntimeError("TabularFeaturePipeline must be fit before calling transform.")

        extracted = extract_features_dataframe(
            df,
            title_col=self.title_col,
            desc_col=self.desc_col,
            bullets_col=self.bullets_col,
        )
        full_df = pd.concat([df, extracted], axis=1)

        out = pd.DataFrame(index=df.index)

        # 1. Fill numeric features
        for col in self.numeric_cols_:
            val = (
                full_df[col]
                if col in full_df.columns
                else pd.Series(self.medians_.get(col, 0.0), index=df.index)
            )
            out[col] = (
                pd.to_numeric(val, errors="coerce")
                .fillna(self.medians_.get(col, 0.0))
                .astype(float)
            )

        # 2. Encode categoricals
        cat_present = [c for c in self.categorical_cols if c in full_df.columns]
        if cat_present:
            cat_df = full_df[cat_present].fillna("MISSING").astype(str)
            encoded_cats = self.encoder_.transform(cat_df)
            for i, col in enumerate(cat_present):
                out[f"{col}_cat"] = encoded_cats[:, i]

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
