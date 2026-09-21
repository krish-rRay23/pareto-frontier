"""Master execution script for training, OOF generation, and submission."""

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import generate_synthetic_catalog_data, load_dataset
from src.data.validation import validate_dataset_schema
from src.inference.predict import generate_submission_file
from src.inference.submission_validator import validate_submission_file
from src.training.logger import ExperimentLogger
from src.training.trainer import CrossValidationTrainer

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML Competition Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "baseline_ridge.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run pipeline on synthetic catalog data for offline verification",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_id = config.get("experiment", {}).get("id", "experiment")
    seed = config.get("experiment", {}).get("seed", 42)
    notes = config.get("experiment", {}).get("notes", "")

    console.print(f"[bold cyan]=== Pareto-Frontier: Starting {exp_id} ===[/bold cyan]")

    data_cfg = config.get("data", {})
    id_col = data_cfg.get("id_col", "sample_id")
    target_col = data_cfg.get("target_col", "target_price")
    group_col = data_cfg.get("group_col", "category")
    allow_zeros = data_cfg.get("allow_zero_target", False)

    # 1. Load Data
    train_path = PROJECT_ROOT / data_cfg.get("train_path", "data/raw/train.csv")
    test_path = PROJECT_ROOT / data_cfg.get("test_path", "data/raw/test.csv")

    if args.synthetic or not train_path.exists() or not test_path.exists():
        console.print("[yellow]Using generated synthetic catalog data for verification.[/yellow]")
        train_df = generate_synthetic_catalog_data(n_samples=400, seed=seed, is_test=False)
        test_df = generate_synthetic_catalog_data(n_samples=100, seed=seed + 1, is_test=True)
    else:
        console.print(f"[green]Loading train data from {train_path}[/green]")
        train_df = load_dataset(train_path)
        console.print(f"[green]Loading test data from {test_path}[/green]")
        test_df = load_dataset(test_path)

    # 2. Validate Data Schema
    val_report = validate_dataset_schema(
        train_df,
        required_columns=[id_col, target_col],
        id_column=id_col,
        target_column=target_col,
        allow_zero_target=allow_zeros,
    )
    if not val_report.is_valid:
        console.print(f"[red]Train data schema validation failed:\n{val_report.summary()}[/red]")
        sys.exit(1)

    console.print(f"[dim]{val_report.summary()}[/dim]")

    # 3. Cross Validation Training
    trainer = CrossValidationTrainer(config=config, experiment_id=exp_id)
    console.print("[cyan]Running cross-validation folds...[/cyan]")
    cv_metrics = trainer.run_cv(
        train_df=train_df,
        target_col=target_col,
        id_col=id_col,
        group_col=group_col,
    )

    # Display Metrics Table
    table = Table(title=f"Cross Validation Results: {exp_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")
    table.add_row("SMAPE", f"{cv_metrics['smape']:.4f}%")
    table.add_row("MAE", f"{cv_metrics['mae']:.4f}")
    table.add_row("RMSE", f"{cv_metrics['rmse']:.4f}")
    console.print(table)

    # 4. Generate Test Predictions
    console.print("[cyan]Generating test set predictions...[/cyan]")
    test_preds = trainer.predict_test(test_df)

    out_cfg = config.get("output", {})
    sub_dir = PROJECT_ROOT / out_cfg.get("submission_dir", "artifacts/submissions")
    sub_file = out_cfg.get("submission_filename", f"{exp_id}_sub.csv")
    sub_target_col = out_cfg.get("target_name_in_sub", "prediction")
    sub_path = sub_dir / sub_file

    generate_submission_file(
        test_df=test_df,
        predictions=test_preds,
        output_path=sub_path,
        id_col=id_col,
        target_col=sub_target_col,
    )
    console.print(f"[green]Submission written to: {sub_path}[/green]")

    # 5. Validate Submission
    sub_report = validate_submission_file(
        submission_path=sub_path,
        test_df_or_path=test_df,
        id_col=id_col,
        target_col=sub_target_col,
        allow_zeros=allow_zeros,
    )
    if not sub_report.is_valid:
        console.print(f"[red]Submission validation FAILED:\n{sub_report.summary()}[/red]")
        sys.exit(1)
    console.print(f"[bold green]Submission Verified:\n{sub_report.summary()}[/bold green]")

    # 6. Log Experiment
    logger = ExperimentLogger()
    cv_strategy = config.get("validation", {}).get("strategy", "stratified")
    n_splits = config.get("validation", {}).get("n_splits", 5)
    logger.log(
        experiment_id=exp_id,
        config=config,
        metrics=cv_metrics,
        cv_method=f"{n_splits}-Fold {cv_strategy}",
        seed=seed,
        notes=notes,
        submission_file=str(sub_path),
    )
    console.print("[bold green]Pipeline finished successfully and experiment logged.[/bold green]")


if __name__ == "__main__":
    main()
