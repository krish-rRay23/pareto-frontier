"""Unit tests for LightGBM, XGBoost, and Ensemble classifiers."""

import numpy as np
import pandas as pd
import pytest
from src.models.classifier import LightGBMPairClassifier
from src.models.xgboost_classifier import XGBoostPairClassifier
from src.ensemble.blender import EnsemblePairClassifier


def _make_dummy_dataset(n_samples: int = 100):
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        {
            "f1": rng.randn(n_samples),
            "f2": rng.uniform(0, 1, n_samples),
            "f3": rng.choice([0.0, 1.0], size=n_samples),
        }
    )
    # Target correlated with f1 and f3
    prob = 1.0 / (1.0 + np.exp(-(X["f1"] + 2.0 * X["f3"] - 1.0)))
    y = (rng.uniform(0, 1, n_samples) < prob).astype(int)
    return X, y


def test_lightgbm_pair_classifier():
    X, y = _make_dummy_dataset(150)
    clf = LightGBMPairClassifier(n_estimators=20, max_depth=3)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert len(probs) == len(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_xgboost_pair_classifier():
    X, y = _make_dummy_dataset(150)
    clf = XGBoostPairClassifier(n_estimators=20, max_depth=3)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert len(probs) == len(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_ensemble_pair_classifier():
    X, y = _make_dummy_dataset(200)
    clf = EnsemblePairClassifier(
        lgb_params={"n_estimators": 20, "max_depth": 3},
        xgb_params={"n_estimators": 20, "max_depth": 3},
    )
    clf.fit(X[:120], y[:120], X_val=X[120:], y_val=y[120:])
    probs = clf.predict_proba(X[120:])

    assert len(probs) == 80
    assert 0.0 <= clf.alpha <= 1.0
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
