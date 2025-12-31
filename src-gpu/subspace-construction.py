"""GPU subspace construction and difference subspace.

This is a PyTorch (GPU) counterpart of [`src/subspace_construction.py`](src/subspace_construction.py:1).

Core operations:
- Fit k-dim principal subspace using SVD (on GPU).
- Build difference subspace via eigen-decomposition of A = P1 - P2.
- Project samples to r-dim coordinates.

All tensors are expected to be on the same device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

# Allow `import common` from this directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common  # noqa: E402

Tensor = torch.Tensor


@dataclass(frozen=True)
class Subspace:
    basis: Tensor  # (D,k)
    mean: Tensor | None = None  # (D,)

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def ambient_dim(self) -> int:
        return int(self.basis.shape[0])

    def projection(self) -> Tensor:
        u = self.basis
        return u @ u.transpose(0, 1)


class SubspaceConstructor:
    def __init__(self, *, k: int, center: bool = True) -> None:
        if k <= 0:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.center = bool(center)

    def fit(self, x: Tensor) -> Subspace:
        if x.ndim != 2:
            raise ValueError(f"Expected (N,D); got {tuple(x.shape)}")

        if self.center:
            mean = x.mean(dim=0)
            xc = x - mean
        else:
            mean = None
            xc = x

        # xc: (N,D). Vh: (min(N,D), D)
        # Principal subspace in feature space uses top-k right singular vectors.
        _u, _s, vh = torch.linalg.svd(xc, full_matrices=False)
        v = vh.transpose(0, 1)

        k_eff = min(self.k, v.shape[1])
        basis = v[:, :k_eff]

        # QR orthonormalize for numerical stability
        q, _r = torch.linalg.qr(basis, mode="reduced")
        return Subspace(basis=q, mean=mean)


def canonical_angles(s1: Subspace, s2: Subspace) -> tuple[Tensor, Tensor]:
    u1 = s1.basis
    u2 = s2.basis
    if u1.shape[0] != u2.shape[0]:
        raise ValueError(f"Ambient dims differ: {u1.shape[0]} vs {u2.shape[0]}")

    m = u1.transpose(0, 1) @ u2
    sv = torch.linalg.svdvals(m)
    sv = torch.clamp(sv, -1.0, 1.0)
    angles = torch.acos(sv)
    return angles, sv


def difference_subspace(s1: Subspace, s2: Subspace, *, r: int = 3) -> tuple[Subspace, Tensor]:
    if r <= 0:
        raise ValueError("r must be >= 1")

    p1 = s1.projection()
    p2 = s2.projection()
    a = p1 - p2

    evals, evecs = torch.linalg.eigh(a)  # asc
    order = torch.argsort(torch.abs(evals), descending=True)
    order = order[: min(r, evecs.shape[1])]

    sel_vecs = evecs.index_select(1, order)
    sel_vals = evals.index_select(0, order)

    q, _r = torch.linalg.qr(sel_vecs, mode="reduced")
    return Subspace(basis=q, mean=None), sel_vals


def project(x: Tensor, *, basis: Tensor, mean: Tensor) -> Tensor:
    """Project (N,D) samples to (N,r) coordinates via (x-mean) @ basis."""

    if x.ndim != 2:
        raise ValueError(f"Expected (N,D); got {tuple(x.shape)}")
    if mean.ndim != 1:
        mean = mean.reshape(-1)

    return (x - mean.unsqueeze(0)) @ basis


def fisher_ratio_score(x3: Tensor, y01: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Compute a simple Fisher ratio on 3D points.

    J = ||mu0-mu1||^2 / (tr(Sw) + eps)

    Returns:
        scalar tensor
    """

    if x3.ndim != 2 or x3.shape[1] != 3:
        raise ValueError(f"Expected (N,3); got {tuple(x3.shape)}")
    if y01.ndim != 1 or y01.shape[0] != x3.shape[0]:
        raise ValueError(f"Expected y shape (N,); got {tuple(y01.shape)}")

    y01 = y01.to(dtype=torch.long)
    x0 = x3[y01 == 0]
    x1 = x3[y01 == 1]
    if x0.shape[0] < 2 or x1.shape[0] < 2:
        raise ValueError("Need at least 2 samples per class")

    mu0 = x0.mean(dim=0)
    mu1 = x1.mean(dim=0)
    num = torch.sum((mu0 - mu1) ** 2)

    c0 = x0 - mu0
    c1 = x1 - mu1
    sw_trace = torch.sum(c0**2) + torch.sum(c1**2)

    return num / (sw_trace + float(eps))
