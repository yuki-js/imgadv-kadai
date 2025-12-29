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


def _read_bmp_mask_32(path: str | Path) -> np.ndarray:
    """Read a 32x32 BMP (24-bit, uncompressed) and return mask (32,32,1) in [0,1].

    Conventions:
        - White (255) => keep (1.0)
        - Black (0)  => drop (0.0)

    This is intentionally dependency-free so the mask can be edited in MS Paint.
    """

    p = Path(path)
    data = p.read_bytes()

    if data[:2] != b"BM":
        raise ValueError(f"Not a BMP file: {p}")

    pixel_offset = int.from_bytes(data[10:14], "little", signed=False)
    dib_size = int.from_bytes(data[14:18], "little", signed=False)
    if dib_size < 40:
        raise ValueError(f"Unsupported BMP DIB header size={dib_size} in {p}")

    w = int.from_bytes(data[18:22], "little", signed=True)
    h = int.from_bytes(data[22:26], "little", signed=True)
    planes = int.from_bytes(data[26:28], "little", signed=False)
    bpp = int.from_bytes(data[28:30], "little", signed=False)
    compression = int.from_bytes(data[30:34], "little", signed=False)

    if planes != 1:
        raise ValueError(f"Unsupported BMP planes={planes} in {p}")
    if (w, abs(h)) != (32, 32):
        raise ValueError(f"Mask BMP must be 32x32; got {w}x{abs(h)} in {p}")
    if bpp != 24:
        raise ValueError(f"Mask BMP must be 24-bit; got bpp={bpp} in {p}")
    if compression != 0:
        raise ValueError(f"Mask BMP must be uncompressed (BI_RGB); got compression={compression} in {p}")

    height = abs(h)
    # each row padded to 4-byte boundary
    row_bytes = ((w * 3 + 3) // 4) * 4

    mask = np.zeros((height, w), dtype=np.float64)

    # BMP stores rows bottom-up if h > 0, top-down if h < 0
    bottom_up = h > 0

    for row in range(height):
        src_row = (height - 1 - row) if bottom_up else row
        start = pixel_offset + src_row * row_bytes
        end = start + w * 3
        row_data = data[start:end]
        # BGR triplets; use luminance-like mean
        rgb = np.frombuffer(row_data, dtype=np.uint8).reshape(w, 3)[:, ::-1]
        v = rgb.mean(axis=1).astype(np.float64) / 255.0
        mask[row, :] = v

    return mask[..., None]


def _apply_face_mask_32(x32_flat: np.ndarray, *, mask32: np.ndarray) -> np.ndarray:
    """Apply a (32,32,1) mask to flattened 32x32x3 images."""

    x32_flat = np.asarray(x32_flat, dtype=np.float64)
    if x32_flat.ndim != 2 or x32_flat.shape[1] != 32 * 32 * 3:
        raise ValueError(f"Expected (N, 3072); got shape={x32_flat.shape}")

    mask32 = np.asarray(mask32, dtype=np.float64)
    if mask32.shape != (32, 32, 1):
        raise ValueError(f"mask32 must be (32,32,1); got shape={mask32.shape}")

    n = x32_flat.shape[0]
    img = x32_flat.reshape(n, 32, 32, 3)
    img = img * mask32  # broadcast over channels
    return img.reshape(n, 32 * 32 * 3)


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

    # Apply user-provided mask template (editable in MS Paint).
    # White (255) => keep, Black (0) => drop
    mask_path = Path("src") / "mask_template_32.bmp"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path} (expected 32x32 24-bit BMP)")

    mask32 = _read_bmp_mask_32(mask_path)
    aoki = _apply_face_mask_32(aoki, mask32=mask32)
    itadakimasu = _apply_face_mask_32(itadakimasu, mask32=mask32)

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
