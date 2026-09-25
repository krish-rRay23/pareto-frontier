"""Reproducible and leakage-safe cross-validation splitters for Entity Resolution."""

from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


def get_entity_splits(
    entity_ids: List[str],
    stratify_labels: Optional[List[str]] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate k-fold splits of unique S1 entity IDs.
    
    Args:
        entity_ids: List of unique Source 1 entity IDs.
        stratify_labels: Optional stratification labels (e.g. country or is_singleton).
        n_splits: Number of folds (default 5).
        shuffle: Whether to shuffle before splitting.
        seed: Random seed.
        
    Returns:
        List of (train_indices, val_indices) indexing into the provided entity_ids list.
    """
    n = len(entity_ids)
    indices = np.arange(n)
    
    if stratify_labels is not None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        return list(skf.split(indices, stratify_labels))
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
        return list(kf.split(indices))


def get_cv_splitter(
    strategy: str = "group",
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate pair-level train/val index splits for cross-validation.

    For Entity Resolution, 'group' strategy ensures that no source1_entity_id
    appears in both train and validation splits simultaneously.
    """
    if strategy == "group":
        if groups is None:
            raise ValueError("Group split requires groups array (source1_entity_id).")
        gkf = GroupKFold(n_splits=n_splits)
        dummy_X = np.zeros(len(groups))
        yield from gkf.split(dummy_X, groups=groups)

    elif strategy == "stratified_group":
        if y is None or groups is None:
            raise ValueError("Stratified group split requires both y and groups.")
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
        )
        dummy_X = np.zeros(len(y))
        yield from sgkf.split(dummy_X, y, groups=groups)

    elif strategy == "stratified":
        if y is None:
            raise ValueError("Stratified split requires target array y.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        dummy_X = np.zeros(len(y))
        yield from skf.split(dummy_X, y)

    elif strategy == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
        dummy_X = np.zeros(len(y) if y is not None else 100)
        yield from kf.split(dummy_X)

    else:
        raise ValueError(
            f"Unknown split strategy '{strategy}'. Choose from: 'group', 'stratified_group', 'stratified', 'kfold'."
        )


def get_folds_list(
    strategy: str = "group",
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
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
        )
    )
