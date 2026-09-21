"""Data and schema validation utilities for competition datasets."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Detailed report produced by data schema validation."""

    is_valid: bool = True
    row_count: int = 0
    column_count: int = 0
    missing_required_columns: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_id_count: int = 0
    invalid_target_count: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Data Validation Report [{status}] ===",
            f"Rows: {self.row_count}, Columns: {self.column_count}",
        ]
        if self.missing_required_columns:
            lines.append(f"Missing Required Columns: {self.missing_required_columns}")
        if self.duplicate_id_count > 0:
            lines.append(f"Duplicate IDs: {self.duplicate_id_count}")
        if self.invalid_target_count > 0:
            lines.append(f"Invalid Targets (negative/inf/NaN): {self.invalid_target_count}")
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def validate_dataset_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    id_column: Optional[str] = None,
    target_column: Optional[str] = None,
    allow_zero_target: bool = False,
) -> ValidationReport:
    """Validate a raw or processed dataframe against competition requirements.

    Args:
        df: Input DataFrame.
        required_columns: List of columns that must exist.
        id_column: Name of sample identifier column (e.g. 'sample_id', 'id').
        target_column: Name of target regression column (if train dataset).
        allow_zero_target: Whether target values of 0 are permitted.

    Returns:
        ValidationReport with validation outcome and detailed diagnostics.
    """
    report = ValidationReport()
    report.row_count = len(df)
    report.column_count = len(df.columns)

    if report.row_count == 0:
        report.is_valid = False
        report.errors.append("Dataset contains 0 rows.")
        return report

    # 1. Required Columns
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        report.is_valid = False
        report.missing_required_columns = missing_cols
        report.errors.append(f"Missing required columns: {missing_cols}")

    # 2. Missing value diagnostics
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            report.null_counts[col] = n_null
            if col in required_columns:
                pct = (n_null / report.row_count) * 100
                report.warnings.append(f"Column '{col}' has {n_null} ({pct:.1f}%) missing values.")

    # 3. Duplicate IDs
    if id_column and id_column in df.columns:
        dup_count = int(df[id_column].duplicated().sum())
        report.duplicate_id_count = dup_count
        if dup_count > 0:
            report.is_valid = False
            report.errors.append(f"Found {dup_count} duplicate IDs in column '{id_column}'.")

    # 4. Target validity (for training set)
    if target_column and target_column in df.columns:
        target_series = pd.to_numeric(df[target_column], errors="coerce")
        n_nan = int(target_series.isna().sum())
        n_inf = int(np.isinf(target_series.fillna(0)).sum())
        n_neg = int((target_series < 0).sum())
        n_zero = int((target_series == 0).sum())

        invalid_targets = n_nan + n_inf + n_neg
        if not allow_zero_target:
            invalid_targets += n_zero

        report.invalid_target_count = invalid_targets

        if invalid_targets > 0:
            report.is_valid = False
            report.errors.append(
                f"Target column '{target_column}' contains invalid values: "
                f"NaN={n_nan}, Inf={n_inf}, Negative={n_neg}, Zero={n_zero}."
            )

    return report
