"""Reproducible and leakage-safe cross-validation splitters."""

from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


def create_target_bins(
    y: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> np.ndarray:
    """Bin continuous targets to allow StratifiedKFold splitting on continuous regression values.

    Args:
        y: Target array.
        n_bins: Number of stratification bins.
        strategy: 'quantile' (equal frequency) or 'uniform' (equal width).

    Returns:
        Array of integer bin labels.
    """
    y = np.asarray(y)
    if strategy == "quantile":
        # Handle duplicate quantiles gracefully
        try:
            bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
            return np.asarray(bins, dtype=int)
        except Exception:
            bins = pd.cut(y, bins=n_bins, labels=False)
            return np.asarray(bins, dtype=int)
    else:
        bins = pd.cut(y, bins=n_bins, labels=False)
        return np.asarray(bins, dtype=int)


def get_cv_splitter(
    strategy: str = "stratified",
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate train/val index splits for cross-validation.

    Strategies supported:
        - 'kfold': Standard K-Fold.
        - 'stratified': Target-binned Stratified K-Fold (balances regression target distribution).
        - 'group': GroupKFold (no group overlap between train and val).
        - 'stratified_group': StratifiedGroupKFold (balances target bins while keeping groups disjoint).

    Yields:
        (train_idx, val_idx)
    """
    if strategy == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
        dummy_X = np.zeros(len(y) if y is not None else 100)
        yield from kf.split(dummy_X)

    elif strategy == "stratified":
        if y is None:
            raise ValueError("Stratified split requires target array y.")
        y_bins = create_target_bins(y, n_bins=n_bins)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        dummy_X = np.zeros(len(y))
        yield from skf.split(dummy_X, y_bins)

    elif strategy == "group":
        if groups is None:
            raise ValueError("Group split requires groups array.")
        gkf = GroupKFold(n_splits=n_splits)
        dummy_X = np.zeros(len(groups))
        yield from gkf.split(dummy_X, groups=groups)

    elif strategy == "stratified_group":
        if y is None or groups is None:
            raise ValueError("Stratified group split requires both y and groups.")
        y_bins = create_target_bins(y, n_bins=n_bins)
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
        )
        dummy_X = np.zeros(len(y))
        yield from sgkf.split(dummy_X, y_bins, groups=groups)

    else:
        raise ValueError(
            f"Unknown split strategy '{strategy}'. Choose from: 'kfold', 'stratified', 'group', 'stratified_group'."
        )


def get_folds_list(
    strategy: str = "stratified",
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convenience function returning a static list of (train_idx, val_idx)."""
    return list(
        get_cv_splitter(
            strategy=strategy,
            n_splits=n_splits,
            shuffle=shuffle,
            seed=seed,
            y=y,
            groups=groups,
            n_bins=n_bins,
        )
    )
