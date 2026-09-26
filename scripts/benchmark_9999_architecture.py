"""Empirical benchmark of the 99.99+ Architecture vs Baseline on real dataset sample."""

import os
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_benchmark_subset
from src.inference.pipeline import ProductionEntityResolutionPipeline
from src.validation.metrics import compute_macro_f05

console = Console()


def run_benchmark(n_s1: int = 1500):
    train_dir = str(PROJECT_ROOT / "resources/student_resource/dataset/train")
    if not os.path.exists(train_dir):
        console.print(f"[red]Dataset train directory not found at {train_dir}[/red]")
        return

    console.print(f"[bold cyan]>>> Loading benchmark subset of {n_s1:,} S1 entities...[/bold cyan]")
    s1_records, target_records, ground_truth = load_benchmark_subset(
        data_dir=train_dir,
        n_s1=n_s1,
        background_noise_ratio=20,
    )
    console.print(f"[green]Loaded {len(s1_records):,} S1s and {len(target_records):,} targets.[/green]")

    # Run 99.99+ Production Architecture
    console.print("\n[bold yellow]>>> Evaluating Full 99.99+ Architecture (Bidirectional + Ensemble + Context + Expected-F0.5)...[/bold yellow]")
    t0 = time.time()
    pipeline = ProductionEntityResolutionPipeline(
        use_ensemble=True,
        use_hard_negatives=True,
        use_calibration=True,
        use_expected_f05=True,
        use_target_exclusivity=True,
        lgb_params={"n_estimators": 100, "max_depth": 5, "min_child_samples": 5},
        xgb_params={"n_estimators": 100, "max_depth": 5},
    )

    metrics = pipeline.fit_and_validate(
        s1_records=s1_records,
        target_records=target_records,
        ground_truth=ground_truth,
        n_splits=3,
        verbose=True,
    )
    elapsed = time.time() - t0

    console.print(f"\n[bold green]Benchmark Completed in {elapsed:.1f}s![/bold green]")
    slice_audit = metrics.get("slice_audit", {})

    table = Table(title="99.99+ Architecture Error Slice Breakdown")
    table.add_column("Slice Name", style="cyan")
    table.add_column("Entity Count", style="white", justify="right")
    table.add_column("Macro F0.5", style="green", justify="right")
    table.add_column("Precision", style="yellow", justify="right")
    table.add_column("Recall", style="magenta", justify="right")

    for s_name, res in slice_audit.items():
        if s_name == "overall":
            continue
        table.add_row(
            s_name,
            f"{res.get('count', 0):,}",
            f"{res.get('macro_f05', 0.0):.4f}",
            f"{res.get('precision', 0.0):.4f}",
            f"{res.get('recall', 0.0):.4f}",
        )

    console.print(table)


if __name__ == "__main__":
    n = 1500
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(n_s1=n)
