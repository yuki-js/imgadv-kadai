"""Dataset loading and preprocessing.

- Loads `datasets/aoki.npy` and `datasets/itadakimasu.npy`
- Balances sample counts by downsampling the larger dataset
- Splits each class into (train, test)

Datasets are expected as:
    X: shape (N, D) where each row is a flattened image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from src.common import BalancedSplit, downsample_to_match, normalize_vectors, train_test_split_indices


@dataclass(frozen=True)
class DatasetConfig:
    aoki_path: str = "datasets/aoki.npy"
    itadakimasu_path: str = "datasets/itadakimasu.npy"
    test_ratio: float = 0.2
    seed: int = 0


def load_raw_datasets(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    aoki = np.load(cfg.aoki_path)
    itadakimasu = np.load(cfg.itadakimasu_path)

    aoki = normalize_vectors(aoki)
    itadakimasu = normalize_vectors(itadakimasu)

    if aoki.ndim != 2 or itadakimasu.ndim != 2:
        raise ValueError("Each dataset must have shape (N, D)")

    if aoki.shape[1] != itadakimasu.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: aoki D={aoki.shape[1]} vs itadakimasu D={itadakimasu.shape[1]}"
        )

    return aoki, itadakimasu


def load_balanced_split(cfg: DatasetConfig) -> BalancedSplit:
    """Load, balance, and split into train/test for each class."""

    rng = np.random.default_rng(cfg.seed)

    aoki, itadakimasu = load_raw_datasets(cfg)
    aoki, itadakimasu = downsample_to_match(aoki, itadakimasu, rng=rng)

    a_tr_idx, a_te_idx = train_test_split_indices(aoki.shape[0], cfg.test_ratio, rng=rng)
    b_tr_idx, b_te_idx = train_test_split_indices(itadakimasu.shape[0], cfg.test_ratio, rng=rng)

    return BalancedSplit(
        a_train=aoki[a_tr_idx],
        a_test=aoki[a_te_idx],
        b_train=itadakimasu[b_tr_idx],
        b_test=itadakimasu[b_te_idx],
    )
