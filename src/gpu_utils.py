"""GPU utilities (PyTorch) for Query6/7.

Design goals (per further-discussion-prompt.md):
- Move datasets to GPU once.
- Keep heavy linear algebra and masking on GPU.
- Minimize CPU<->GPU transfers.

This module is intentionally independent from Query1-4 (which remain NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class GpuBalancedSplit:
    """Balanced per-class train/test tensors on GPU."""

    a_train: Tensor
    a_test: Tensor
    b_train: Tensor
    b_test: Tensor


def get_default_device() -> torch.device:
    """Return CUDA device if available; else CPU."""

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def torch_rng(seed: int, *, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def load_npy_to_device(path: str | Path, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Load .npy (NumPy) then transfer to GPU once."""

    x_np = np.load(str(path))
    x = torch.as_tensor(x_np)
    if x.dtype not in (torch.float16, torch.float32, torch.float64):
        x = x.to(torch.float32)
    x = x.to(dtype=dtype, device=device, non_blocking=False)
    return x


def downscale_flat_rgb_to_32(x_flat: Tensor) -> Tensor:
    """Downscale flattened RGB images to 32x32x3 and flatten (GPU).

    Expected:
        x_flat: (N, D) with D = H*W*3, square H=W, and H multiple of 32.

    Returns:
        (N, 32*32*3) float tensor on same device.
    """

    if x_flat.ndim != 2:
        raise ValueError(f"Expected (N, D); got shape={tuple(x_flat.shape)}")

    n, d = int(x_flat.shape[0]), int(x_flat.shape[1])
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

    img = x_flat.reshape(n, h, h, 3)
    img = img.reshape(n, 32, s, 32, s, 3).mean(dim=(2, 4))
    return img.reshape(n, 32 * 32 * 3)


def apply_mask_32(x32_flat: Tensor, *, mask32: Tensor) -> Tensor:
    """Apply a (32,32,1) mask to flattened 32x32x3 images (GPU)."""

    if x32_flat.ndim != 2 or int(x32_flat.shape[1]) != 32 * 32 * 3:
        raise ValueError(f"Expected (N, 3072); got shape={tuple(x32_flat.shape)}")

    if tuple(mask32.shape) != (32, 32, 1):
        raise ValueError(f"mask32 must be (32,32,1); got shape={tuple(mask32.shape)}")

    n = int(x32_flat.shape[0])
    img = x32_flat.reshape(n, 32, 32, 3)
    img = img * mask32  # broadcast channels
    return img.reshape(n, 32 * 32 * 3)


def train_test_split_indices(n: int, test_ratio: float, *, g: torch.Generator, device: torch.device) -> Tuple[Tensor, Tensor]:
    if not (0.0 < float(test_ratio) < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    perm = torch.randperm(int(n), generator=g, device=device)
    n_test = int(round(n * float(test_ratio)))
    n_test = max(1, min(n - 1, n_test))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


def downsample_to_match(a: Tensor, b: Tensor, *, g: torch.Generator) -> Tuple[Tensor, Tensor]:
    na, nb = int(a.shape[0]), int(b.shape[0])
    n = min(na, nb)

    if na != n:
        a_idx = torch.randperm(na, generator=g, device=a.device)[:n]
        a = a.index_select(0, a_idx)
    if nb != n:
        b_idx = torch.randperm(nb, generator=g, device=b.device)[:n]
        b = b.index_select(0, b_idx)

    return a, b


def make_mask_from_squares(
    *,
    square_size: int = 12,
    shift_x: int = 0,
    shift_y: int = 0,
    gap: int = 0,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate a mask (32,32,1) with four square "keep" regions.

    Layout: 2x2 squares in a block:
      (0,0) (0,1)
      (1,0) (1,1)

    The entire block is shifted by (shift_x, shift_y). Out-of-bounds are clipped.

    Returns:
        mask32: (32,32,1) in {0,1} on `device`.
    """

    s = int(square_size)
    if s <= 0 or s > 32:
        raise ValueError("square_size must be in [1,32]")

    # block width/height: 2 squares + optional gap between them
    bw = 2 * s + int(gap)
    bh = 2 * s + int(gap)

    x0 = int(shift_x)
    y0 = int(shift_y)

    mask = torch.zeros((32, 32), device=device, dtype=dtype)

    def _fill_square(x: int, y: int) -> None:
        x1 = max(0, min(32, x))
        y1 = max(0, min(32, y))
        x2 = max(0, min(32, x + s))
        y2 = max(0, min(32, y + s))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

    # top-left of 2x2 block
    _fill_square(x0 + 0, y0 + 0)
    _fill_square(x0 + s + int(gap), y0 + 0)
    _fill_square(x0 + 0, y0 + s + int(gap))
    _fill_square(x0 + s + int(gap), y0 + s + int(gap))

    # (32,32,1)
    return mask.unsqueeze(-1)


def load_balanced_split_gpu(
    *,
    aoki_path: str = "datasets/aoki.npy",
    itadakimasu_path: str = "datasets/itadakimasu.npy",
    test_ratio: float = 0.2,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    mask32: Tensor | None = None,
) -> GpuBalancedSplit:
    """Load, downscale to 32x32, apply mask (optional), balance, and split (GPU)."""

    device = get_default_device() if device is None else device
    g = torch_rng(seed, device=device)

    a = load_npy_to_device(aoki_path, device=device, dtype=dtype)
    b = load_npy_to_device(itadakimasu_path, device=device, dtype=dtype)

    a = downscale_flat_rgb_to_32(a)
    b = downscale_flat_rgb_to_32(b)

    if mask32 is not None:
        mask32 = mask32.to(device=device, dtype=dtype)
        a = apply_mask_32(a, mask32=mask32)
        b = apply_mask_32(b, mask32=mask32)

    a, b = downsample_to_match(a, b, g=g)

    a_tr, a_te = train_test_split_indices(int(a.shape[0]), test_ratio, g=g, device=device)
    b_tr, b_te = train_test_split_indices(int(b.shape[0]), test_ratio, g=g, device=device)

    return GpuBalancedSplit(
        a_train=a.index_select(0, a_tr),
        a_test=a.index_select(0, a_te),
        b_train=b.index_select(0, b_tr),
        b_test=b.index_select(0, b_te),
    )
