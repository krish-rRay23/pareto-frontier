"""Unit tests for submission integrity verification."""

import numpy as np
import pandas as pd
import pytest

from src.inference.submission_validator import validate_submission_file


@pytest.fixture
def test_data(tmp_path):
    test_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5"],
            "catalog_title": ["A", "B", "C", "D", "E"],
        }
    )
    test_csv = tmp_path / "test.csv"
    test_df.to_csv(test_csv, index=False)
    return test_csv, test_df


def test_submission_valid(tmp_path, test_data):
    test_csv, test_df = test_data
    sub_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5"],
            "prediction": [10.5, 20.2, 30.1, 40.0, 50.9],
        }
    )
    sub_csv = tmp_path / "sub_valid.csv"
    sub_df.to_csv(sub_csv, index=False)

    report = validate_submission_file(sub_csv, test_df)
    assert report.is_valid
    assert len(report.errors) == 0


def test_submission_row_count_mismatch(tmp_path, test_data):
    test_csv, test_df = test_data
    sub_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "prediction": [10.5, 20.2],
        }
    )
    sub_csv = tmp_path / "sub_short.csv"
    sub_df.to_csv(sub_csv, index=False)

    report = validate_submission_file(sub_csv, test_df)
    assert not report.is_valid
    assert any("Row count mismatch" in e for e in report.errors)


def test_submission_nan_and_negative(tmp_path, test_data):
    test_csv, test_df = test_data
    sub_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5"],
            "prediction": [10.5, np.nan, -5.0, np.inf, 20.0],
        }
    )
    sub_csv = tmp_path / "sub_invalid_nums.csv"
    sub_df.to_csv(sub_csv, index=False)

    report = validate_submission_file(sub_csv, test_df)
    assert not report.is_valid
    assert report.nan_count == 1
    assert report.negative_count == 1
    assert report.inf_count == 1


def test_submission_id_order_warning(tmp_path, test_data):
    test_csv, test_df = test_data
    sub_df = pd.DataFrame(
        {
            "sample_id": ["S5", "S4", "S3", "S2", "S1"],
            "prediction": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    sub_csv = tmp_path / "sub_reordered.csv"
    sub_df.to_csv(sub_csv, index=False)

    report = validate_submission_file(sub_csv, test_df)
    assert report.is_valid  # Still valid set of IDs
    assert report.order_mismatches == 1  # But flagged with order warning
