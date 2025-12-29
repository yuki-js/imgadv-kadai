"""GPU subspace construction and difference subspace (PyTorch) for Query6/7.

We mirror the NumPy implementation in src/subspace_construction.py, but operate
on torch tensors (ideally on CUDA) to keep data resident on GPU.

Notes:
- Uses torch.linalg.svd and torch.linalg.eigh.
- Intended ambient dimension D=3072, k<=3, r=3 (small).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class GpuSubspace:
    basis: Tensor  # (D,k) orthonormal columns
    mean: Tensor  # (D,)

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def ambient_dim(self) -> int:
        return int(self.basis.shape[0])


def _as_2d(x: Tensor) -> Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor (N,D); got shape={tuple(x.shape)}")
    return x


def _orthonormalize_columns(u: Tensor) -> Tensor:
    q, _r = torch.linalg.qr(u, mode="reduced")
    return q


def fit_subspace(x: Tensor, *, k: int, center: bool = True) -> GpuSubspace:
    """Fit PCA subspace via SVD.

    Args:
        x: (N,D)
        k: subspace dimension
        center: subtract mean
    """

    x = _as_2d(x)
    n, d = int(x.shape[0]), int(x.shape[1])
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    if k > min(n, d):
        raise ValueError(f"k must be <= min(N,D) (= {min(n, d)}); got k={k}")

    mean = x.mean(dim=0) if center else torch.zeros((d,), device=x.device, dtype=x.dtype)
    xc = x - mean

    # torch.linalg.svd returns U,S,Vh
    _u, _s, vh = torch.linalg.svd(xc, full_matrices=False)
    basis = vh[:k].transpose(0, 1)  # (D,k)
    basis = _orthonormalize_columns(basis)

    return GpuSubspace(basis=basis, mean=mean)


def difference_subspace(s1: GpuSubspace, s2: GpuSubspace, *, r: int = 3) -> tuple[GpuSubspace, Tensor]:
    """Compute r-dim difference subspace from projection matrix difference A=P1-P2."""

    u1 = s1.basis
    u2 = s2.basis
    if u1.ndim != 2 or u2.ndim != 2:
        raise ValueError("Subspace bases must be 2D")
    if int(u1.shape[0]) != int(u2.shape[0]):
        raise ValueError(f"Ambient dimension mismatch: {int(u1.shape[0])} vs {int(u2.shape[0])}")

    d = int(u1.shape[0])
    r = int(r)
    if r <= 0:
        raise ValueError("r must be positive")
    if r > d:
        raise ValueError(f"r must be <= ambient dimension D (= {d}); got r={r}")

    p1 = u1 @ u1.transpose(0, 1)
    p2 = u2 @ u2.transpose(0, 1)
    a = p1 - p2

    evals, evecs = torch.linalg.eigh(a)  # ascending
    order = torch.argsort(torch.abs(evals), descending=True)[:r]

    basis = evecs.index_select(1, order)
    basis = _orthonormalize_columns(basis)

    ds = GpuSubspace(basis=basis, mean=torch.zeros((d,), device=basis.device, dtype=basis.dtype))
    return ds, evals.index_select(0, order)


def project(x: Tensor, *, basis: Tensor, mean: Tensor) -> Tensor:
    """Project samples (N,D) -> (N,r) via (x-mean) @ basis."""

    x = _as_2d(x)
    mean = mean.reshape(1, -1)
    return (x - mean) @ basis
