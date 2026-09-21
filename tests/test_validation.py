"""Unit tests for schema validation and leakage-safe validation splitters."""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import validate_dataset_schema
from src.validation.splitters import get_folds_list
from src.validation.target_encoding import LeakageSafeTargetEncoder


def test_schema_validation_valid():
    df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "target_price": [10.0, 20.0, 30.0],
            "catalog_title": ["Item A", "Item B", "Item C"],
        }
    )
    report = validate_dataset_schema(
        df,
        required_columns=["sample_id", "target_price"],
        id_column="sample_id",
        target_column="target_price",
    )
    assert report.is_valid
    assert len(report.errors) == 0


def test_schema_validation_missing_columns():
    df = pd.DataFrame({"sample_id": ["S1", "S2"]})
    report = validate_dataset_schema(
        df,
        required_columns=["sample_id", "target_price"],
        target_column="target_price",
    )
    assert not report.is_valid
    assert "target_price" in report.missing_required_columns


def test_schema_validation_duplicate_ids():
    df = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2"],
            "target_price": [10.0, 20.0, 30.0],
        }
    )
    report = validate_dataset_schema(
        df,
        required_columns=["sample_id", "target_price"],
        id_column="sample_id",
        target_column="target_price",
    )
    assert not report.is_valid
    assert report.duplicate_id_count == 1


def test_schema_validation_invalid_targets():
    df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "target_price": [10.0, -5.0, np.nan, np.inf],
        }
    )
    report = validate_dataset_schema(
        df,
        required_columns=["sample_id", "target_price"],
        id_column="sample_id",
        target_column="target_price",
    )
    assert not report.is_valid
    assert report.invalid_target_count == 3


def test_cross_validation_splitters():
    y = np.linspace(10, 1000, 100)
    folds = get_folds_list(strategy="stratified", n_splits=5, seed=42, y=y)
    assert len(folds) == 5

    val_indices = []
    for train_idx, val_idx in folds:
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        val_indices.extend(val_idx)

    # All validation indices must cover entire dataset exactly once
    assert sorted(val_indices) == list(range(100))


def test_target_encoding_leakage_safety():
    df = pd.DataFrame(
        {
            "brand": ["Apple", "Apple", "Samsung", "Samsung", "Apple", "Samsung"],
            "target": [100.0, 120.0, 50.0, 60.0, 110.0, 55.0],
        }
    )
    folds = [(np.array([0, 1, 2, 3]), np.array([4, 5]))]

    encoder = LeakageSafeTargetEncoder(categorical_cols=["brand"], smoothing=1.0)
    oof_df, fitted_enc = encoder.fit_transform_oof(df, df["target"], folds)

    # Check that OOF target encoded values are populated
    assert not np.isnan(oof_df.loc[4, "brand_te"])
    assert not np.isnan(oof_df.loc[5, "brand_te"])

    # Test transform on unseen data works with learned global prior
    test_df = pd.DataFrame({"brand": ["UnknownBrand"]})
    test_enc = fitted_enc.transform(test_df)
    assert test_enc.loc[0, "brand_te"] == pytest.approx(fitted_enc.global_means_["brand"], abs=1e-4)
