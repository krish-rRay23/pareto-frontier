"""Comprehensive submission validator for competition deliverables."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd


@dataclass
class SubmissionValidationReport:
    """Diagnostic report from submission integrity verification."""

    is_valid: bool = True
    row_count: int = 0
    test_row_count: int = 0
    column_names: List[str] = field(default_factory=list)
    id_mismatches: int = 0
    order_mismatches: int = 0
    nan_count: int = 0
    inf_count: int = 0
    negative_count: int = 0
    zero_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Submission Validation Report [{status}] ===",
            f"Submission Rows: {self.row_count} (Expected: {self.test_row_count})",
            f"Columns: {self.column_names}",
            f"NaN: {self.nan_count}, Inf: {self.inf_count}, Negatives: {self.negative_count}, Zeros: {self.zero_count}",
        ]
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - [FAIL] {err}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - [WARN] {w}")
        return "\n".join(lines)


def validate_submission_file(
    submission_path: Union[str, Path],
    test_df_or_path: Union[str, Path, pd.DataFrame],
    id_col: str = "sample_id",
    target_col: str = "prediction",
    allow_zeros: bool = False,
) -> SubmissionValidationReport:
    """Validate submission CSV against test set for row count, ordering, and value validity."""
    report = SubmissionValidationReport()

    # Load submission
    sub_path = Path(submission_path)
    if not sub_path.exists():
        report.is_valid = False
        report.errors.append(f"Submission file not found: {sub_path}")
        return report

    try:
        sub_df = pd.read_csv(sub_path)
    except Exception as e:
        report.is_valid = False
        report.errors.append(f"Could not read submission CSV: {e}")
        return report

    report.row_count = len(sub_df)
    report.column_names = list(sub_df.columns)

    # Load test set for reference IDs
    if isinstance(test_df_or_path, pd.DataFrame):
        test_df = test_df_or_path
    else:
        test_path = Path(test_df_or_path)
        if not test_path.exists():
            report.is_valid = False
            report.errors.append(f"Test reference file not found: {test_path}")
            return report
        test_df = pd.read_csv(test_path)

    report.test_row_count = len(test_df)

    # 1. Check Row Count
    if report.row_count != report.test_row_count:
        report.is_valid = False
        report.errors.append(
            f"Row count mismatch: submission has {report.row_count} rows, test set has {report.test_row_count}."
        )

    # 2. Check Required Columns
    expected_cols = [id_col, target_col]
    missing_cols = [c for c in expected_cols if c not in sub_df.columns]
    if missing_cols:
        report.is_valid = False
        report.errors.append(f"Missing required submission columns: {missing_cols}")
        return report

    # 3. Check ID alignment and ordering
    sub_ids = sub_df[id_col].astype(str).tolist()
    test_ids = test_df[id_col].astype(str).tolist()

    if set(sub_ids) != set(test_ids):
        diff_missing = len(set(test_ids) - set(sub_ids))
        diff_extra = len(set(sub_ids) - set(test_ids))
        report.is_valid = False
        report.errors.append(
            f"ID set mismatch: {diff_missing} test IDs missing, {diff_extra} unexpected extra IDs in submission."
        )
    else:
        # Check exact sequence order
        if sub_ids != test_ids:
            report.order_mismatches = 1
            report.warnings.append(
                "Submission IDs match test set but are in DIFFERENT ROW ORDER. Sorting submission to match test order is advised."
            )

    # 4. Numerical Validity
    preds = pd.to_numeric(sub_df[target_col], errors="coerce").to_numpy(dtype=float)

    nan_count = int(np.isnan(preds).sum())
    inf_count = int(np.isinf(preds).sum())
    neg_count = int((preds < 0).sum())
    zero_count = int((preds == 0).sum())

    report.nan_count = nan_count
    report.inf_count = inf_count
    report.negative_count = neg_count
    report.zero_count = zero_count

    if nan_count > 0:
        report.is_valid = False
        report.errors.append(f"Submission contains {nan_count} NaN values.")

    if inf_count > 0:
        report.is_valid = False
        report.errors.append(f"Submission contains {inf_count} Infinite (Inf / -Inf) values.")

    if neg_count > 0:
        report.is_valid = False
        report.errors.append(f"Submission contains {neg_count} negative predictions.")

    if zero_count > 0 and not allow_zeros:
        report.warnings.append(
            f"Submission contains {zero_count} zero predictions. Ensure zero values are valid in this competition."
        )

    return report
