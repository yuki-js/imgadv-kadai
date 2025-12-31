"""Dataset loading and preprocessing.

This assignment uses two face datasets stored as `.npy` files:
- `datasets/aoki.npy`
- `datasets/itadakimasu.npy`

Each dataset is represented as an array `X` of shape (N, D) where each row is a
flattened image vector.

Assignment spec (updated):
- Preprocess every image with:
  1) Bilinear downscale to 48x48
  2) Grayscale conversion
  3) (Optional) mask composition at preprocessing time
- The final feature dimension must be 48*48*1 = 2304.
- Balance dataset sizes by downsampling the larger class.
- Provide per-class train/test split.

Mask spec:
- Mask is provided as a grayscale PNG.
- White(255) => visible / keep, Black(0) => hidden / drop.
- Default filename: `mask.png`
- Can be overridden via environment variable: `IMGADV_MASK_FILE`.

Run-time note:
- If the stored `.npy` vectors are already 48x48x1 flattened (D=2304), we skip
  resize/grayscale and only apply the mask.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from src.common import (
    BalancedSplit,
    downsample_to_match,
    normalize_vectors,
    train_test_split_indices,
)

Array = np.ndarray


@dataclass(frozen=True)
class DatasetConfig:
    aoki_path: str = "datasets/aoki.npy"
    itadakimasu_path: str = "datasets/itadakimasu.npy"
    test_ratio: float = 0.2
    seed: int = 0

    out_h: int = 48
    out_w: int = 48
    mask_env: str = "IMGADV_MASK_FILE"
    mask_default: str = "mask.png"


def _infer_square_rgb_side(d: int) -> int:
    """Infer H=W from D=H*W*3. Raises if not square RGB."""

    if d % 3 != 0:
        raise ValueError(f"Expected RGB flattened vectors (D divisible by 3); got D={d}")

    hw = d // 3
    h = int(round(hw**0.5))
    if h * h != hw:
        raise ValueError(f"Expected square RGB images; got D={d} => hw={hw} is not a square")

    return h


def _load_mask_48(cfg: DatasetConfig) -> Array:
    """Load mask PNG and return float64 array (48,48,1) in [0,1]."""

    mask_path = Path(os.environ.get(cfg.mask_env, cfg.mask_default))
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_path} (set env {cfg.mask_env} to override)"
        )

    with Image.open(mask_path) as im:
        mask = im.convert("L").resize((cfg.out_w, cfg.out_h), resample=Image.BILINEAR)

    m = np.asarray(mask, dtype=np.float64) / 255.0
    return m[..., None]


def _apply_mask_gray(x_flat: Array, *, mask: Array, h: int, w: int) -> Array:
    """Apply grayscale mask (H,W,1) to grayscale flattened vectors (N,H*W)."""

    x_flat = np.asarray(x_flat, dtype=np.float64)
    if x_flat.ndim != 2 or x_flat.shape[1] != h * w:
        raise ValueError(f"Expected (N,{h*w}); got shape={x_flat.shape}")

    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != (h, w, 1):
        raise ValueError(f"mask must be ({h},{w},1); got shape={mask.shape}")

    n = x_flat.shape[0]
    img = x_flat.reshape(n, h, w, 1)
    img = img * mask
    return img.reshape(n, h * w)


def _rgb_flat_to_gray48(x_rgb_flat: Array, *, out_h: int, out_w: int) -> Array:
    """Convert flattened RGB images to 48x48 grayscale (bilinear) and flatten.

    Expected input:
        x_rgb_flat: (N, D) with D = H*W*3 (square H=W)

    Returns:
        (N, out_h*out_w) float64
    """

    x = np.asarray(x_rgb_flat)
    if x.ndim != 2:
        raise ValueError(f"Expected (N, D); got shape={x.shape}")

    n, d = x.shape
    side = _infer_square_rgb_side(d)

    # PIL expects uint8 for images. Our source is typically uint8; if not,
    # clip/convert safely.
    if x.dtype != np.uint8:
        x_uint8 = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x_uint8 = x

    out = np.empty((n, out_h * out_w), dtype=np.float64)

    for i in range(n):
        img = x_uint8[i].reshape(side, side, 3)
        im = Image.fromarray(img, mode="RGB")
        im = im.resize((out_w, out_h), resample=Image.BILINEAR).convert("L")
        out[i, :] = np.asarray(im, dtype=np.float64).reshape(-1)

    return out


def _ensure_gray48(x: Array, *, cfg: DatasetConfig) -> Array:
    """Ensure the feature vectors are grayscale 48x48 flattened."""

    x = normalize_vectors(x)

    if x.ndim != 2:
        raise ValueError(f"Each dataset must have shape (N, D); got {x.shape}")

    d = x.shape[1]
    if d == cfg.out_h * cfg.out_w:
        # already grayscale 48x48 (flatten)
        return x

    # otherwise, treat as flattened RGB and preprocess
    return _rgb_flat_to_gray48(x, out_h=cfg.out_h, out_w=cfg.out_w)


def load_raw_datasets(cfg: DatasetConfig) -> Tuple[Array, Array]:
    """Load datasets, preprocess to 48x48 grayscale, and apply mask."""

    aoki = np.load(cfg.aoki_path)
    itadakimasu = np.load(cfg.itadakimasu_path)

    aoki = _ensure_gray48(aoki, cfg=cfg)
    itadakimasu = _ensure_gray48(itadakimasu, cfg=cfg)

    if aoki.shape[1] != itadakimasu.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: aoki D={aoki.shape[1]} vs itadakimasu D={itadakimasu.shape[1]}"
        )

    mask = _load_mask_48(cfg)
    aoki = _apply_mask_gray(aoki, mask=mask, h=cfg.out_h, w=cfg.out_w)
    itadakimasu = _apply_mask_gray(itadakimasu, mask=mask, h=cfg.out_h, w=cfg.out_w)

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
