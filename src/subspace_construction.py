"""Subspace construction and core linear-algebra routines.

This module intentionally implements the required algorithms from scratch (no direct
use of `subspace-tk`).

Terminology:
- Given centered data X (N,D), we build an orthonormal basis U (D,k) spanning the
  principal subspace (top-k left singular vectors of X).
- Projection matrix: P = U U^T.

Difference subspace:
- For two subspaces with projections P1, P2, define A = P1 - P2 (symmetric).
- The "difference subspace" is spanned by eigenvectors of A corresponding to the
  largest |eigenvalue|.

References (conceptual): MSM / DCSM lecture notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Subspace:
    """Orthonormal basis representation of a linear subspace."""

    basis: Array  # (D,k)
    mean: Array | None = None  # (D,) if centering was used when fitting

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def ambient_dim(self) -> int:
        return int(self.basis.shape[0])

    def projection(self) -> Array:
        """Return projection matrix P = U U^T (D,D)."""

        u = np.asarray(self.basis, dtype=np.float64)
        return u @ u.T


class SubspaceConstructor:
    """Fit a k-dimensional subspace from data via SVD."""

    def __init__(
        self,
        *,
        k: int,
        center: bool = True,
        svd: Literal["full", "econ"] = "econ",
    ) -> None:
        if k <= 0:
            raise ValueError("k must be >= 1")
        if svd not in ("full", "econ"):
            raise ValueError("svd must be 'full' or 'econ'")
        self.k = int(k)
        self.center = bool(center)
        self.svd = svd

    def fit(self, x: Array) -> Subspace:
        """Fit subspace from samples x with shape (N,D)."""

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected (N,D); got shape={x.shape}")

        mean: Array | None
        if self.center:
            mean = x.mean(axis=0)
            xc = x - mean
        else:
            mean = None
            xc = x

        # SVD on (N,D). Right singular vectors Vt are (D,D) (or (min,D)).
        # The principal subspace in feature space is spanned by the first k
        # right singular vectors (rows of Vt), i.e., columns of V (D,k).
        full_matrices = self.svd == "full"
        _u, _s, vt = np.linalg.svd(xc, full_matrices=full_matrices)
        v = vt.T

        k_eff = min(self.k, v.shape[1])
        basis = v[:, :k_eff].copy()
        basis = _orthonormalize(basis)
        return Subspace(basis=basis, mean=mean)


def _orthonormalize(basis: Array) -> Array:
    """Return an orthonormal basis spanning the same columns (QR-based)."""

    q, _r = np.linalg.qr(np.asarray(basis, dtype=np.float64), mode="reduced")
    return q


def radians_to_degrees(rad: Array) -> Array:
    return np.asarray(rad, dtype=np.float64) * (180.0 / np.pi)


def canonical_angles(s1: Subspace, s2: Subspace) -> Tuple[Array, Array]:
    """Compute canonical angles between two subspaces.

    Returns:
        angles_rad: (k,) in [0, pi/2]
        singular_values: (k,) singular values of (U1^T U2)
    """

    u1 = np.asarray(s1.basis, dtype=np.float64)
    u2 = np.asarray(s2.basis, dtype=np.float64)

    if u1.shape[0] != u2.shape[0]:
        raise ValueError(f"Ambient dims differ: {u1.shape[0]} vs {u2.shape[0]}")

    m = u1.T @ u2
    sv = np.linalg.svd(m, compute_uv=False)

    # numerical guard
    sv = np.clip(sv, -1.0, 1.0)
    angles = np.arccos(sv)
    return angles, sv


def difference_subspace(s1: Subspace, s2: Subspace, *, r: int = 3) -> Tuple[Subspace, Array]:
    """Compute r-dim difference subspace from two subspaces.

    Steps:
        A = P1 - P2 (symmetric)
        eigen-decompose: A w = lambda w
        choose r eigenvectors with largest |lambda|

    Returns:
        ds: Subspace(basis=(D,r))
        evals: selected eigenvalues aligned with basis columns
    """

    if r <= 0:
        raise ValueError("r must be >= 1")

    p1 = s1.projection()
    p2 = s2.projection()
    a = p1 - p2

    # symmetric eigendecomposition
    evals, evecs = np.linalg.eigh(a)  # evals asc

    order = np.argsort(np.abs(evals))[::-1]
    order = order[: min(r, evecs.shape[1])]

    sel_vecs = evecs[:, order].copy()
    sel_vals = evals[order].copy()

    sel_vecs = _orthonormalize(sel_vecs)
    return Subspace(basis=sel_vecs, mean=None), sel_vals
