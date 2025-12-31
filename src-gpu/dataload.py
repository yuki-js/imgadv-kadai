"""GPU dataset loading and preprocessing.

Differences vs [`src/dataload.py`](src/dataload.py:1):
- Loads `.npy` on CPU then moves tensors to GPU once.
- Preprocessing (RGB->grayscale + bilinear resize to 48x48) is done on GPU.
- Query7 mask application is done on GPU.
- Provides balanced per-class train/test split.

Mask for Query7:
- Query7 explores parametric masks; so this module does NOT load PNG masks.
- It returns grayscale 48x48 flattened vectors on GPU.

Import note:
- `src-gpu/` is not a Python package name (hyphen), so we add this directory to
  `sys.path` to import sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Allow `import common` from this directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common  # noqa: E402

Tensor = torch.Tensor
BalancedSplit = common.BalancedSplit


@dataclass(frozen=True)
class DatasetConfig:
    aoki_path: str = "datasets/aoki.npy"
    itadakimasu_path: str = "datasets/itadakimasu.npy"
    test_ratio: float = 0.2
    seed: int = 0

    out_h: int = 48
    out_w: int = 48


def _load_npy_to_device(path: str, *, device_: torch.device, dtype: torch.dtype) -> Tensor:
    x = np.load(path)
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected (N,D) array in {path}; got shape={x.shape}")

    # Most datasets are uint8 or float; keep numeric stability with float32/float64 on GPU.
    t = torch.from_numpy(x)
    if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        t = t.to(dtype=dtype)
    else:
        t = t.to(dtype=dtype)

    return t.to(device_)


def _infer_square_rgb_side(d: int) -> int:
    if d % 3 != 0:
        raise ValueError(f"Expected RGB flattened vectors (D divisible by 3); got D={d}")
    hw = d // 3
    side = int(round(hw**0.5))
    if side * side != hw:
        raise ValueError(f"Expected square RGB images; got D={d} => hw={hw} is not a square")
    return side


def _ensure_gray48(x: Tensor, *, out_h: int, out_w: int) -> Tensor:
    """Ensure x is grayscale 48x48 flattened on GPU.

    Accepts:
    - (N,2304) already grayscale 48x48
    - (N,H*W*3) square RGB flattened (e.g. 256*256*3)

    Returns:
    - (N,2304) float tensor
    """

    if x.ndim != 2:
        raise ValueError(f"Expected (N,D); got {tuple(x.shape)}")

    n, d = int(x.shape[0]), int(x.shape[1])
    target_d = int(out_h * out_w)
    if d == target_d:
        return x

    side = _infer_square_rgb_side(d)

    # (N, D) -> (N, 3, H, W)
    x_img = x.reshape(n, side, side, 3).permute(0, 3, 1, 2)

    # RGB -> grayscale (luma)
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x_gray = torch.sum(x_img * w, dim=1, keepdim=True)

    # Bilinear resize to 48x48
    x_resized = F.interpolate(x_gray, size=(out_h, out_w), mode="bilinear", align_corners=False)

    return x_resized.reshape(n, target_d)


def load_raw_datasets(
    cfg: DatasetConfig,
    *,
    device_: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    device_ = device_ or common.device()

    a = _load_npy_to_device(cfg.aoki_path, device_=device_, dtype=dtype)
    b = _load_npy_to_device(cfg.itadakimasu_path, device_=device_, dtype=dtype)

    a = _ensure_gray48(a, out_h=cfg.out_h, out_w=cfg.out_w)
    b = _ensure_gray48(b, out_h=cfg.out_h, out_w=cfg.out_w)

    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Feature dimension mismatch: D_a={a.shape[1]} vs D_b={b.shape[1]}")

    return a, b


def load_balanced_split(cfg: DatasetConfig, *, device_: torch.device | None = None, dtype: torch.dtype = torch.float32) -> BalancedSplit:
    device_ = device_ or common.device()

    common.set_global_seed(cfg.seed)
    gen = torch.Generator(device=device_)
    gen.manual_seed(int(cfg.seed))

    a, b = load_raw_datasets(cfg, device_=device_, dtype=dtype)
    a, b = common.downsample_to_match(a, b, generator=gen)

    a_tr, a_te = common.train_test_split_indices(
        a.shape[0], cfg.test_ratio, generator=gen, device_=device_
    )
    b_tr, b_te = common.train_test_split_indices(
        b.shape[0], cfg.test_ratio, generator=gen, device_=device_
    )

    return BalancedSplit(
        a_train=a.index_select(0, a_tr),
        a_test=a.index_select(0, a_te),
        b_train=b.index_select(0, b_tr),
        b_test=b.index_select(0, b_te),
    )
