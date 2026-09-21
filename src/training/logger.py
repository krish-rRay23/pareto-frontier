"""Experiment logging system to record configs, CV metrics, and notes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Logs competition experiments to both JSONL and formatted Markdown table."""

    def __init__(
        self,
        log_md_path: Optional[Path] = None,
        log_json_path: Optional[Path] = None,
    ):
        base_dir = Path(__file__).resolve().parents[2]
        self.log_md_path = log_md_path or (base_dir / "docs" / "experiments" / "experiment_log.md")
        self.log_json_path = log_json_path or (base_dir / "artifacts" / "experiments_history.jsonl")

        self.log_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_json_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        cv_method: str,
        seed: int,
        notes: str = "",
        submission_file: str = "-",
    ) -> None:
        """Record experiment run into JSONL and Markdown table."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "seed": seed,
            "cv_method": cv_method,
            "metrics": metrics,
            "notes": notes,
            "submission_file": submission_file,
            "config": config,
        }

        # 1. Append to JSONL
        with open(self.log_json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # 2. Append row to Markdown table
        smape_str = f"{metrics.get('smape', 0.0):.4f}%" if "smape" in metrics else "-"
        mae_str = f"{metrics.get('mae', 0.0):.4f}" if "mae" in metrics else "-"
        rmse_str = f"{metrics.get('rmse', 0.0):.4f}" if "rmse" in metrics else "-"
        sub_str = Path(submission_file).name if submission_file != "-" else "-"

        md_row = (
            f"| {experiment_id} | {timestamp} | {config.get('model', {}).get('type', 'custom')} | "
            f"{cv_method} (s={seed}) | {smape_str} | {mae_str} | {rmse_str} | "
            f"{notes.replace('|', '/')} | {sub_str} |\n"
        )

        with open(self.log_md_path, "a", encoding="utf-8") as f:
            f.write(md_row)
