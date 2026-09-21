"""Inference engine to generate predictions from saved fold checkpoints."""

from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd


def load_fold_models(models_dir: Path) -> List[object]:
    """Load all saved model fold checkpoints from a directory."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {models_dir}")

    model_files = sorted(models_dir.glob("model_fold_*.joblib"))
    if not model_files:
        raise ValueError(f"No fold checkpoints found in {models_dir}")

    models = [joblib.load(p) for p in model_files]
    return models


def predict_from_checkpoints(
    models_dir: Path,
    test_df: pd.DataFrame,
    min_clip: float = 1e-3,
) -> np.ndarray:
    """Load checkpoints and generate fold-averaged ensemble predictions."""
    models = load_fold_models(models_dir)
    all_preds = []
    for model in models:
        preds = model.predict(test_df)
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    return np.clip(avg_preds, min_clip, None)


def generate_submission_file(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Union[str, Path],
    id_col: str = "sample_id",
    target_col: str = "prediction",
    min_clip: float = 1e-3,
) -> Path:
    """Generate and write a sanitized submission file matching test order."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds = np.asarray(predictions, dtype=float)
    if len(preds) != len(test_df):
        raise ValueError(
            f"Prediction length ({len(preds)}) does not match test rows ({len(test_df)})"
        )

    preds = np.nan_to_num(preds, nan=min_clip, posinf=min_clip, neginf=min_clip)
    preds = np.clip(preds, min_clip, None)

    sub_df = pd.DataFrame(
        {
            id_col: test_df[id_col],
            target_col: preds,
        }
    )
    sub_df.to_csv(out_path, index=False)
    return out_path
