"""Subspace construction utilities.

This file intentionally avoids directly using `subspace-tk` implementations.

Terminology:
    Given a dataset X \in R^{N x D} (each row is a sample), we construct a
    k-dimensional subspace S spanned by an orthonormal basis U \in R^{D x k}.

Main operations:
    - Fit a subspace by PCA/SVD
    - Compute canonical angles between two subspaces
    - Compute a *difference subspace* between two subspaces

Notes on the difference subspace:
    We use a standard symmetric formulation based on projection matrices.

    Let P1 = U1 U1^T and P2 = U2 U2^T. Define A = P1 - P2 (symmetric).
    Eigenvectors of A associated with large |eigenvalues| capture directions
    where the two subspaces differ the most.

    - positive eigenvalue: direction more aligned with S1 than S2
    - negative eigenvalue: direction more aligned with S2 than S1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Subspace:
    """An orthonormal-basis representation of a linear subspace."""

    basis: Array  # (D, k) with orthonormal columns
    mean: Array  # (D,) mean used for centering (zeros if not centered)

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def ambient_dim(self) -> int:
        return int(self.basis.shape[0])


def _as_2d(x: Array) -> Array:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (N, D); got shape={x.shape}")
    return x


def _orthonormalize_columns(u: Array) -> Array:
    """Return an orthonormal basis spanning the same column-space as `u`."""

    # QR is stable and fast; we only need Q.
    q, _r = np.linalg.qr(u, mode="reduced")
    return q


class SubspaceConstructor:
    """Construct a k-dimensional subspace from samples using PCA (SVD).

    Args:
        k: Target subspace dimension.
        center: If True, subtract the sample mean before SVD.
        svd: Which SVD backend to use.
    """

    def __init__(
        self,
        *,
        k: int,
        center: bool = True,
        svd: Literal["full", "randomized"] = "full",
        seed: int = 0,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.center = bool(center)
        self.svd = svd
        self.seed = int(seed)

    def fit(self, x: Array) -> Subspace:
        """Fit a subspace from samples X of shape (N, D)."""

        x = _as_2d(x)
        n, d = x.shape
        if self.k > min(n, d):
            raise ValueError(f"k must be <= min(N, D) (= {min(n, d)}); got k={self.k}")

        mean = x.mean(axis=0) if self.center else np.zeros((d,), dtype=x.dtype)
        xc = x - mean

        # PCA basis: right singular vectors Vt (k rows) correspond to top principal directions.
        if self.svd == "full":
            _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
            basis = vt[: self.k].T  # (D, k)
        elif self.svd == "randomized":
            basis = _randomized_pca_basis(xc, k=self.k, seed=self.seed)
        else:
            raise ValueError("svd must be one of: 'full', 'randomized'")

        basis = _orthonormalize_columns(basis)
        return Subspace(basis=basis, mean=mean)


def _randomized_pca_basis(xc: Array, *, k: int, seed: int = 0, n_oversamples: int = 10, n_iter: int = 2) -> Array:
    """A small randomized PCA routine (no external deps).

    Notes:
        This is included for speed on very large D. For this assignment,
        'full' SVD is typically fine.
    """

    xc = _as_2d(xc)
    n, d = xc.shape
    l = min(d, k + int(n_oversamples))

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(size=(d, l))
    y = xc @ omega  # (N, l)

    for _ in range(int(n_iter)):
        y = xc @ (xc.T @ y)

    q, _r = np.linalg.qr(y, mode="reduced")  # (N, l)
    b = q.T @ xc  # (l, D)
    _ub, _sb, vt = np.linalg.svd(b, full_matrices=False)
    basis = vt[:k].T
    return basis


def canonical_angles(s1: Subspace, s2: Subspace) -> Tuple[Array, Array]:
    """Compute canonical angles between two subspaces.

    Returns:
        angles_rad: (m,) angles in radians, where m = min(dim(S1), dim(S2)).
        singular_values: (m,) singular values of U1^T U2.

    Details:
        If U1 and U2 have orthonormal columns, singular values of U1^T U2 are
        cosines of canonical angles.
    """

    u1 = np.asarray(s1.basis)
    u2 = np.asarray(s2.basis)

    if u1.ndim != 2 or u2.ndim != 2:
        raise ValueError("Subspace bases must be 2D")
    if u1.shape[0] != u2.shape[0]:
        raise ValueError(f"Ambient dimension mismatch: {u1.shape[0]} vs {u2.shape[0]}")

    m = min(u1.shape[1], u2.shape[1])
    # SVD of the cross-Gram matrix.
    c = u1.T @ u2
    _uc, s, _vt = np.linalg.svd(c, full_matrices=False)
    s = s[:m]

    # Numerical clipping: s in [0, 1]
    s_clipped = np.clip(s, 0.0, 1.0)
    angles = np.arccos(s_clipped)
    return angles, s


def difference_subspace(
    s1: Subspace,
    s2: Subspace,
    *,
    r: int = 3,
) -> Tuple[Subspace, Array]:
    """Compute an r-dimensional difference subspace between S1 and S2.

    We use a symmetric formulation based on projection matrices:
        P1 = U1 U1^T, P2 = U2 U2^T, A = P1 - P2.

    Eigenvectors of A with largest |eigenvalue| form the difference subspace.

    Returns:
        ds: Difference subspace with basis shape (D, r)
        evals: Selected eigenvalues (r,)
    """

    u1 = np.asarray(s1.basis)
    u2 = np.asarray(s2.basis)

    if u1.ndim != 2 or u2.ndim != 2:
        raise ValueError("Subspace bases must be 2D")
    if u1.shape[0] != u2.shape[0]:
        raise ValueError(f"Ambient dimension mismatch: {u1.shape[0]} vs {u2.shape[0]}")

    d = int(u1.shape[0])
    r = int(r)
    if r <= 0:
        raise ValueError("r must be positive")
    if r > d:
        raise ValueError(f"r must be <= ambient dimension D (= {d}); got r={r}")

    p1 = u1 @ u1.T
    p2 = u2 @ u2.T
    a = p1 - p2

    evals, evecs = np.linalg.eigh(a)  # ascending
    order = np.argsort(np.abs(evals))[::-1][:r]

    basis = _orthonormalize_columns(evecs[:, order])
    ds = Subspace(basis=basis, mean=np.zeros((d,), dtype=basis.dtype))
    return ds, evals[order]


def radians_to_degrees(rad: Array) -> Array:
    return np.asarray(rad) * (180.0 / np.pi)
