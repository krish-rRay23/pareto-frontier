"""Standalone CLI script to validate any submission CSV before uploading."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.submission_validator import validate_submission_file

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Submission CSV")
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to generated submission CSV",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test reference CSV",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="sample_id",
        help="Name of sample identifier column",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="prediction",
        help="Name of prediction column in submission",
    )
    parser.add_argument(
        "--allow_zeros",
        action="store_true",
        help="Allow zero values in prediction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sub_path = Path(args.submission)
    test_path = Path(args.test)

    report = validate_submission_file(
        submission_path=sub_path,
        test_df_or_path=test_path,
        id_col=args.id_col,
        target_col=args.target_col,
        allow_zeros=args.allow_zeros,
    )

    if report.is_valid:
        console.print(f"[bold green]{report.summary()}[/bold green]")
        sys.exit(0)
    else:
        console.print(f"[bold red]{report.summary()}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
