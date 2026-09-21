"""End-to-end integration tests for baseline models and blending pipeline."""

import numpy as np
import pytest

from src.data.loader import generate_synthetic_catalog_data
from src.ensemble.blender import OptimalLinearBlender
from src.models.statistical import StatisticalBaselineRegressor
from src.training.trainer import CrossValidationTrainer
from src.validation.metrics import compute_smape


@pytest.fixture
def synthetic_data():
    train_df = generate_synthetic_catalog_data(n_samples=100, seed=42, is_test=False)
    test_df = generate_synthetic_catalog_data(n_samples=25, seed=43, is_test=True)
    return train_df, test_df


def test_statistical_baseline(synthetic_data):
    train_df, test_df = synthetic_data
    model = StatisticalBaselineRegressor(aggregation="median", group_col="category")
    model.fit(train_df, train_df["target_price"])
    preds = model.predict(test_df)

    assert len(preds) == len(test_df)
    assert np.all(preds > 0)


def test_tfidf_ridge_pipeline(synthetic_data, tmp_path):
    train_df, test_df = synthetic_data
    config = {
        "experiment": {"id": "test_ridge", "seed": 42},
        "validation": {"strategy": "stratified", "n_splits": 3, "seed": 42},
        "model": {
            "type": "ridge",
            "params": {"alpha": 1.0, "max_features": 1000, "target_log": True},
        },
    }

    trainer = CrossValidationTrainer(config, experiment_id="test_ridge", artifacts_dir=tmp_path)
    metrics = trainer.run_cv(train_df, target_col="target_price", id_col="sample_id")

    assert "smape" in metrics
    assert metrics["smape"] < 200.0  # Must produce sensible bounded score

    test_preds = trainer.predict_test(test_df)
    assert len(test_preds) == len(test_df)
    assert np.all(test_preds > 0)


def test_gbdt_tabular_pipeline(synthetic_data, tmp_path):
    train_df, test_df = synthetic_data
    config = {
        "experiment": {"id": "test_gbdt", "seed": 42},
        "validation": {"strategy": "stratified", "n_splits": 3, "seed": 42},
        "model": {
            "type": "gbdt",
            "params": {"engine": "auto", "n_estimators": 20, "max_depth": 3, "target_log": True},
        },
    }

    trainer = CrossValidationTrainer(config, experiment_id="test_gbdt", artifacts_dir=tmp_path)
    metrics = trainer.run_cv(train_df, target_col="target_price", id_col="sample_id")

    assert "smape" in metrics
    assert metrics["smape"] < 200.0

    test_preds = trainer.predict_test(test_df)
    assert len(test_preds) == len(test_df)
    assert np.all(test_preds > 0)


def test_optimal_linear_blender():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    # Model 1 underestimates, Model 2 overestimates
    m1 = np.array([90.0, 180.0, 270.0, 360.0])
    m2 = np.array([110.0, 220.0, 330.0, 440.0])

    blender = OptimalLinearBlender(metric="smape")
    blender.fit([m1, m2], y_true)

    assert blender.weights_ is not None
    assert len(blender.weights_) == 2
    assert np.sum(blender.weights_) == pytest.approx(1.0, abs=1e-4)

    blend_pred = blender.predict([m1, m2])
    blend_smape = compute_smape(y_true, blend_pred)
    # Blend should be at least as good as or better than individual models
    assert blend_smape <= compute_smape(y_true, m1)
