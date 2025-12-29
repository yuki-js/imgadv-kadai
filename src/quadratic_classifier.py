"""Quadratic (second-order) classifier in 3D.

We fit a quadratic decision function:
  f(z) = c + w^T z + z^T Q z
where z in R^3. Decision: sign(f(z)) -> class.

Training uses ridge-regularized least squares over polynomial features:
  phi(z) = [1, x, y, z, x^2, y^2, z^2, xy, xz, yz]
so f(z)=theta^T phi(z).

This matches the "decision boundary is a quadric" requirement.

This module operates on torch tensors (CPU or GPU).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class QuadraticModel:
    theta: Tensor  # (10,)

    def predict_logits(self, z: Tensor) -> Tensor:
        phi = quadratic_features(z)
        return phi @ self.theta

    def predict(self, z: Tensor) -> Tensor:
        """Return labels in {-1, +1}."""

        logits = self.predict_logits(z)
        return torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))


def quadratic_features(z: Tensor) -> Tensor:
    """(N,3) -> (N,10) feature map."""

    if z.ndim != 2 or int(z.shape[1]) != 3:
        raise ValueError(f"Expected (N,3); got shape={tuple(z.shape)}")

    x = z[:, 0]
    y = z[:, 1]
    w = z[:, 2]

    ones = torch.ones_like(x)
    feats = torch.stack(
        [
            ones,
            x,
            y,
            w,
            x * x,
            y * y,
            w * w,
            x * y,
            x * w,
            y * w,
        ],
        dim=1,
    )
    return feats


def fit_quadratic_ridge(z: Tensor, y: Tensor, *, l2: float = 1e-2) -> QuadraticModel:
    """Fit ridge regression on quadratic features.

    Args:
        z: (N,3)
        y: (N,) labels in {-1,+1}
        l2: ridge coefficient
    """

    if y.ndim != 1:
        raise ValueError(f"Expected y as (N,); got shape={tuple(y.shape)}")
    if int(z.shape[0]) != int(y.shape[0]):
        raise ValueError("z and y must have same N")

    phi = quadratic_features(z)  # (N,10)

    # Solve (Phi^T Phi + l2 I) theta = Phi^T y
    a = phi.transpose(0, 1) @ phi
    b = phi.transpose(0, 1) @ y

    reg = float(l2) * torch.eye(int(a.shape[0]), device=a.device, dtype=a.dtype)
    theta = torch.linalg.solve(a + reg, b)

    return QuadraticModel(theta=theta)


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true/y_pred must be 1D")
    if int(y_true.shape[0]) != int(y_pred.shape[0]):
        raise ValueError("y_true and y_pred must have same length")
    return float((y_true == y_pred).float().mean().item())
