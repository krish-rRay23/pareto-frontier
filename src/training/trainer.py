"""Cross-validation training orchestration, OOF generation, and fold management."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.models.linear_text import TfidfRidgeRegressor
from src.models.statistical import StatisticalBaselineRegressor
from src.models.tree_models import TabularGBDTRegressor
from src.validation.metrics import compute_all_metrics
from src.validation.splitters import get_folds_list


def create_model_instance(model_config: Dict[str, Any]) -> Any:
    """Instantiate model based on YAML configuration dict."""
    model_type = model_config.get("type", "ridge").lower()
    params = model_config.get("params", {})

    if model_type in ["statistical", "baseline_stat"]:
        return StatisticalBaselineRegressor(**params)
    elif model_type in ["ridge", "tfidf_ridge"]:
        return TfidfRidgeRegressor(**params)
    elif model_type in ["gbdt", "lightgbm", "catboost", "hist_gbdt"]:
        return TabularGBDTRegressor(**params)
    else:
        raise ValueError(f"Unknown model type '{model_type}'.")


class CrossValidationTrainer:
    """Orchestrates leak-free K-Fold training, OOF collection, and checkpoint saving."""

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_id: str = "experiment",
        artifacts_dir: Optional[Path] = None,
    ):
        self.config = config
        self.experiment_id = experiment_id
        base_dir = Path(__file__).resolve().parents[2]
        self.artifacts_dir = artifacts_dir or (base_dir / "artifacts")
        self.models_dir = self.artifacts_dir / "models" / self.experiment_id
        self.oof_dir = self.artifacts_dir / "oof"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.oof_dir.mkdir(parents=True, exist_ok=True)

        self.fitted_models: List[Any] = []
        self.oof_preds: Optional[np.ndarray] = None
        self.cv_metrics: Dict[str, float] = {}

    def run_cv(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        id_col: str = "sample_id",
        group_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """Execute full K-Fold training loop without data leakage."""
        cv_cfg = self.config.get("validation", {})
        strategy = cv_cfg.get("strategy", "stratified")
        n_splits = cv_cfg.get("n_splits", 5)
        seed = cv_cfg.get("seed", 42)

        y = train_df[target_col].to_numpy(dtype=float)
        groups = (
            train_df[group_col].to_numpy() if group_col and group_col in train_df.columns else None
        )

        folds = get_folds_list(
            strategy=strategy,
            n_splits=n_splits,
            shuffle=True,
            seed=seed,
            y=y,
            groups=groups,
        )

        n_samples = len(train_df)
        oof_predictions = np.zeros(n_samples, dtype=float)
        self.fitted_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            fold_tr_df = train_df.iloc[train_idx].copy()
            fold_val_df = train_df.iloc[val_idx].copy()
            fold_tr_y = y[train_idx]

            # Clean fresh instance per fold to eliminate state leakage
            model = create_model_instance(self.config.get("model", {}))
            model.fit(fold_tr_df, fold_tr_y)

            val_preds = model.predict(fold_val_df)
            oof_predictions[val_idx] = val_preds
            self.fitted_models.append(model)

            # Checkpoint fold model
            model_path = self.models_dir / f"model_fold_{fold_idx}.joblib"
            joblib.dump(model, model_path)

        self.oof_preds = oof_predictions
        self.cv_metrics = compute_all_metrics(y, oof_predictions)

        # Save OOF CSV
        oof_df = pd.DataFrame(
            {
                id_col: train_df[id_col],
                "target_true": y,
                "target_pred": oof_predictions,
            }
        )
        oof_path = self.oof_dir / f"{self.experiment_id}_oof.csv"
        oof_df.to_csv(oof_path, index=False)

        return self.cv_metrics

    def predict_test(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate test predictions by averaging fold models."""
        if not self.fitted_models:
            raise RuntimeError("No fitted models available. Run run_cv first.")

        fold_preds = []
        for model in self.fitted_models:
            preds = model.predict(test_df)
            fold_preds.append(preds)

        avg_preds = np.mean(fold_preds, axis=0)
        return avg_preds
