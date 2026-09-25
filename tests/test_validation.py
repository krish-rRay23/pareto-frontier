"""Unit tests for entity resolution cross-validation splitters."""

import numpy as np
import pytest
from src.validation.splitters import get_cv_splitter, get_entity_splits, get_folds_list


def test_entity_splits():
    """Verify that entity splits partition S1 IDs disjointly across folds."""
    s1_ids = [f"S1-{i:04d}" for i in range(100)]
    folds = get_entity_splits(s1_ids, n_splits=5, shuffle=True, seed=42)

    assert len(folds) == 5
    seen_val = []

    for train_idx, val_idx in folds:
        # Zero overlap between train and val
        assert len(set(train_idx) & set(val_idx)) == 0
        seen_val.extend(val_idx)

    # Every entity must be in validation exactly once
    assert sorted(seen_val) == list(range(100))


def test_group_cv_splitter():
    """Verify that GroupKFold prevents same S1 ID from leaking between train and val."""
    groups = np.array(["S1-A", "S1-A", "S1-B", "S1-B", "S1-C", "S1-D", "S1-E"])
    folds = get_folds_list(strategy="group", n_splits=3, groups=groups)

    assert len(folds) == 3
    for train_idx, val_idx in folds:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert len(train_groups & val_groups) == 0
