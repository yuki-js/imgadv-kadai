"""Dataset loading and preprocessing.

- Loads `datasets/aoki.npy` and `datasets/itadakimasu.npy`
- Resizes each image to 32x32x3 (and flattens)
- Balances sample counts by downsampling the larger dataset
- Splits each class into (train, test)

Datasets are expected as:
    X: shape (N, D) where each row is a flattened image.

Resize policy (assignment spec):
    Always downscale to 32x32x3 before any subspace computation.

Implementation note:
    To avoid extra deps, we use block-mean downsampling when the source images
    are square RGB and the side length is an integer multiple of 32 (e.g. 256).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.common import BalancedSplit, downsample_to_match, normalize_vectors, train_test_split_indices


@dataclass(frozen=True)
class DatasetConfig:
    aoki_path: str = "datasets/aoki.npy"
    itadakimasu_path: str = "datasets/itadakimasu.npy"
    test_ratio: float = 0.2
    seed: int = 0


def _downscale_flat_rgb_to_32(x: np.ndarray) -> np.ndarray:
    """Downscale a batch of flattened RGB images to 32x32x3 and flatten.

    Expected input:
        x: (N, D) with D = H*W*3, square H=W, and H multiple of 32.

    Returns:
        (N, 32*32*3) float64
    """

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected (N, D); got shape={x.shape}")

    n, d = x.shape
    if d % 3 != 0:
        raise ValueError(f"Expected RGB flattened vectors (D divisible by 3); got D={d}")

    hw = d // 3
    h = int(round(hw**0.5))
    if h * h != hw:
        raise ValueError(f"Expected square RGB images; got D={d} => hw={hw} is not a square")

    if h % 32 != 0:
        raise ValueError(f"Expected side length multiple of 32; got H=W={h}")

    s = h // 32
    if s <= 0:
        raise ValueError(f"Invalid scale factor: H={h} => s={s}")

    img = x.reshape(n, h, h, 3).astype(np.float64, copy=False)

    # Block-mean downsampling: (h,h) -> (32,32) by averaging sxs blocks.
    img = img.reshape(n, 32, s, 32, s, 3).mean(axis=(2, 4))

    return img.reshape(n, 32 * 32 * 3)


def load_raw_datasets(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    aoki = np.load(cfg.aoki_path)
    itadakimasu = np.load(cfg.itadakimasu_path)

    aoki = normalize_vectors(aoki)
    itadakimasu = normalize_vectors(itadakimasu)

    if aoki.ndim != 2 or itadakimasu.ndim != 2:
        raise ValueError("Each dataset must have shape (N, D)")

    # Mandatory resize to 32x32x3 as specified by the assignment.
    aoki = _downscale_flat_rgb_to_32(aoki)
    itadakimasu = _downscale_flat_rgb_to_32(itadakimasu)

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
