"""Master CLI runner for Amazon ML Challenge 2026 Entity Resolution Pipeline."""

import argparse
import os
import sys
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    load_benchmark_subset,
    load_ground_truth,
    load_source_tsv,
)
from src.inference.pipeline import EntityResolutionPipeline
from src.inference.submission_validator import validate_submission_package

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Entity Resolution")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run in rapid subsample mode for local CPU/GPU iteration",
    )
    parser.add_argument(
        "--n-s1",
        type=int,
        default=None,
        help="Override number of S1 entities to load in sample mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write matching_results.tsv and candidate_pairs.tsv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    sample_mode = args.sample or config.get("execution", {}).get("sample_mode", False)
    n_s1 = args.n_s1 or config.get("execution", {}).get("sample_n_s1", 2000)
    output_dir = args.output_dir or config.get("dataset", {}).get("output_dir", "output")
    output_dir = str(PROJECT_ROOT / output_dir) if not Path(output_dir).is_absolute() else output_dir

    train_dir = str(PROJECT_ROOT / config.get("dataset", {}).get("train_dir", "resources/student_resource/dataset/train"))
    test_dir = str(PROJECT_ROOT / config.get("dataset", {}).get("test_dir", "resources/student_resource/dataset/test"))

    console.print("[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]       AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION PIPELINE    [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]")
    console.print(f"[dim]Mode: {'SUBSAMPLE BENCHMARK' if sample_mode else 'FULL INDUSTRIAL SCALE'} | S1 Target: {n_s1 if sample_mode else 'ALL'}[/dim]")
    console.print(f"[dim]Train Dir:  {train_dir}[/dim]")
    console.print(f"[dim]Test Dir:   {test_dir}[/dim]")
    console.print(f"[dim]Output Dir: {output_dir}[/dim]\n")

    # 1. Load Data
    console.print("[bold yellow]>>> Step 1: Loading Training Data & Ground Truth...[/bold yellow]")
    if sample_mode:
        s1_records, target_records, ground_truth = load_benchmark_subset(
            data_dir=train_dir,
            n_s1=n_s1,
            background_noise_ratio=25,
        )
    else:
        s1_records = load_source_tsv(os.path.join(train_dir, "train_source1.tsv"))
        s2_records = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"))
        s3_records = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"))
        target_records = s2_records + s3_records
        ground_truth = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    console.print(f"[green]Loaded {len(s1_records):,} S1 records, {len(target_records):,} target pool records, {len(ground_truth):,} ground truth entries.[/green]\n")

    # 2. Build Pipeline
    console.print("[bold yellow]>>> Step 2: Initializing Pipeline & Blocker...[/bold yellow]")
    block_cfg = config.get("blocking", {})
    model_cfg = config.get("model", {})
    feat_cfg = config.get("features", {})
    dec_cfg = config.get("decision_policy", {})

    pipeline = EntityResolutionPipeline(
        max_candidates_per_entity=block_cfg.get("max_candidates_per_entity", 50),
        enable_consensus_features=feat_cfg.get("enable_consensus", True),
        lgb_params=model_cfg,
    )

    # 3. Fit and Validate
    console.print("[bold yellow]>>> Step 3: Executing Blocking, Features, GroupKFold, and OOF Decision Optimization...[/bold yellow]")
    n_splits = config.get("execution", {}).get("n_splits", 5)
    metrics = pipeline.fit_and_validate(
        s1_records=s1_records,
        target_records=target_records,
        ground_truth=ground_truth,
        n_splits=n_splits,
        optimize_policy=dec_cfg.get("optimize_on_oof", True),
        verbose=True,
    )

    # Display Metrics Summary Table
    table = Table(title="Out-of-Fold Validation Summary (Official Macro F0.5)")
    table.add_column("Evaluation Metric", style="cyan")
    table.add_column("Score / Count", style="green bold")

    table.add_row("Macro F0.5 (Official Scored)", f"{metrics['macro_f05']:.4f}")
    table.add_row("Macro Precision", f"{metrics['macro_precision']:.4f}")
    table.add_row("Macro Recall", f"{metrics['macro_recall']:.4f}")
    table.add_row("Singleton Accuracy", f"{metrics['singleton_accuracy']:.4f}")
    table.add_row("Candidate Recall (Blocking)", f"{metrics['candidate_recall']*100:.2f}%")
    table.add_row("Avg Candidates per S1", f"{metrics['avg_candidates_per_s1']:.2f}")
    table.add_row("False Merges (Penalized FP)", f"{metrics['false_merges']:,}")
    console.print(table)
    console.print()

    # 4. Test Inference (Optional in sample mode, standard on full test)
    console.print("[bold yellow]>>> Step 4: Test Inference & Official Submission Validation...[/bold yellow]")
    test_s1_file = os.path.join(test_dir, "test_source1.tsv")
    test_s2_file = os.path.join(test_dir, "test_source2.tsv")
    test_s3_file = os.path.join(test_dir, "test_source3.tsv")

    if os.path.isfile(test_s1_file) and os.path.isfile(test_s2_file) and os.path.isfile(test_s3_file):
        console.print("[cyan]Loading test records from dataset/test...[/cyan]")
        if sample_mode:
            # In sample mode, run inference on first 100 test S1 records for smoke verification
            console.print("[yellow]Running smoke inference on first 100 test entities...[/yellow]")
            test_s1 = load_source_tsv(test_s1_file, max_rows=100)
            test_s2 = load_source_tsv(test_s2_file, max_rows=2000)
            test_s3 = load_source_tsv(test_s3_file, max_rows=2000)
            test_targets = test_s2 + test_s3
            # In sample mode, skip full test row count validator
            matching_path, candidate_path = pipeline.predict_test_and_export(
                test_s1_records=test_s1,
                test_target_records=test_targets,
                output_dir=output_dir,
                test_dir=test_dir,
                validate=True,
                verbose=True,
            )
            console.print(f"[bold green]Smoke test submission generated successfully:[/bold green]")
            console.print(f"  - Matching Results: {matching_path}")
            console.print(f"  - Candidate Pairs:  {candidate_path}")
        else:
            test_s1 = load_source_tsv(test_s1_file)
            test_s2 = load_source_tsv(test_s2_file)
            test_s3 = load_source_tsv(test_s3_file)
            test_targets = test_s2 + test_s3
            matching_path, candidate_path = pipeline.predict_test_and_export(
                test_s1_records=test_s1,
                test_target_records=test_targets,
                output_dir=output_dir,
                test_dir=test_dir,
                validate=True,
                verbose=True,
            )
            console.print(f"[bold green]Official submission files successfully validated and ready for upload![/bold green]")
    else:
        console.print("[yellow]Test directory files not found, skipping test inference step.[/yellow]")

    console.print("\n[bold green]Pipeline run complete.[/bold green]")


if __name__ == "__main__":
    main()
