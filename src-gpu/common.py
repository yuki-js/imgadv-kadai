"""GPU-side common utilities.

This directory is intentionally separated from `src/` (CPU version).

Design goals:
- Keep data on GPU (minimal CPU<->GPU transfer).
- Use a single tensor backend consistently: PyTorch.

Notes:
- Filenames follow the assignment spec, but Python imports require valid module names.
  Call these modules via path execution (e.g. `python src-gpu/query7.py`) or by adding
  `src-gpu` to `PYTHONPATH`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


Tensor = torch.Tensor


def device() -> torch.device:
    """Return the preferred device (CUDA if available)."""

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train_test_split_indices(
    n: int,
    test_ratio: float,
    *,
    generator: torch.Generator,
    device_: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Return train/test indices for `n` samples as GPU tensors."""

    if not (0.0 < float(test_ratio) < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    n = int(n)
    perm = torch.randperm(n, generator=generator, device=device_)

    n_test = int(round(n * float(test_ratio)))
    n_test = max(1, min(n - 1, n_test))

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


def downsample_to_match(
    a: Tensor,
    b: Tensor,
    *,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    """Downsample the larger tensor so both have the same number of rows."""

    na, nb = int(a.shape[0]), int(b.shape[0])
    n = min(na, nb)

    if na != n:
        idx = torch.randperm(na, generator=generator, device=a.device)[:n]
        a = a.index_select(0, idx)

    if nb != n:
        idx = torch.randperm(nb, generator=generator, device=b.device)[:n]
        b = b.index_select(0, idx)

    return a, b


@dataclass(frozen=True)
class BalancedSplit:
    a_train: Tensor
    a_test: Tensor
    b_train: Tensor
    b_test: Tensor


def quadratic_features_3d(x3: Tensor) -> Tensor:
    """Return quadratic feature map for 3D inputs.

    Input:
        x3: (N,3)

    Output:
        phi: (N,10) for
          [x1,x2,x3, x1^2,x2^2,x3^2, x1x2,x2x3,x3x1, 1]
    """

    if x3.ndim != 2 or x3.shape[1] != 3:
        raise ValueError(f"Expected (N,3); got {tuple(x3.shape)}")

    x1, x2, x3c = x3[:, 0], x3[:, 1], x3[:, 2]
    ones = torch.ones_like(x1)

    phi = torch.stack(
        [
            x1,
            x2,
            x3c,
            x1 * x1,
            x2 * x2,
            x3c * x3c,
            x1 * x2,
            x2 * x3c,
            x3c * x1,
            ones,
        ],
        dim=1,
    )
    return phi
