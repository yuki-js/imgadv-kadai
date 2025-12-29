"""Common utilities for subspace-method assignments.

This project intentionally avoids directly using `subspace-tk` implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


Array = np.ndarray


def set_global_seed(seed: int) -> np.random.Generator:
    """Create and return a NumPy RNG for reproducible experiments."""

    return np.random.default_rng(seed)


def train_test_split_indices(
    n: int,
    test_ratio: float = 0.2,
    *,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    """Return train/test indices for `n` samples."""

    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    perm = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    n_test = max(1, min(n - 1, n_test))

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


@dataclass(frozen=True)
class BalancedSplit:
    """Balanced per-class train/test arrays."""

    a_train: Array
    a_test: Array
    b_train: Array
    b_test: Array


def downsample_to_match(
    a: Array,
    b: Array,
    *,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    """Downsample the larger dataset so both have the same number of samples."""

    na, nb = a.shape[0], b.shape[0]
    n = min(na, nb)

    a_idx = rng.choice(na, size=n, replace=False) if na != n else np.arange(na)
    b_idx = rng.choice(nb, size=n, replace=False) if nb != n else np.arange(nb)

    return a[a_idx], b[b_idx]


def normalize_vectors(x: Array) -> Array:
    """Normalize input to float64, keeping original scaling.

    Notes:
        This function does NOT rescale to [0, 1] or standardize. It only ensures
        downstream linear algebra is stable by using float64.
    """

    return np.asarray(x, dtype=np.float64)

