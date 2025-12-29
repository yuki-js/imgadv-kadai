"""Query 6 (GPU)

Task:
- Build diff-subspace (3D) using subspace method on train.
- Train a classifier from train data (in the 3D coordinates).
- Visualize the classifier decision boundary as a quadric surface in 3D.
- Evaluate accuracy on test data.

Requirements (from further-discussion-prompt.md):
- Prepare GPU acceleration; keep data and heavy computations on GPU.
- Decision boundary must be a quadratic surface.
- Visualize in 3D with interactive rotation (we output Plotly HTML).

Run:
    python -m src.query6

Outputs:
    src/artifacts/query6_decision_boundary_3d.html
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from src.gpu_subspace import difference_subspace, fit_subspace, project
from src.gpu_utils import get_default_device, load_balanced_split_gpu
from src.quadratic_classifier import accuracy, fit_quadratic_ridge


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").numpy()


def _make_grid(bounds: tuple[float, float], n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lo, hi = float(bounds[0]), float(bounds[1])
    return torch.linspace(lo, hi, int(n), device=device, dtype=dtype)


def main() -> None:
    device = get_default_device()
    dtype = torch.float32

    # Load/downscale/balance/split on GPU (no mask sweep in Query6)
    split = load_balanced_split_gpu(test_ratio=0.2, seed=0, device=device, dtype=dtype)

    # Construct per-class subspaces and difference subspace (all on GPU)
    n_train = min(int(split.a_train.shape[0]), int(split.b_train.shape[0]))
    d = int(split.a_train.shape[1])
    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples to construct a subspace")

    s1 = fit_subspace(split.a_train, k=k, center=True)
    s2 = fit_subspace(split.b_train, k=k, center=True)
    ds, evals = difference_subspace(s1, s2, r=3)

    global_mean = torch.cat([split.a_train, split.b_train], dim=0).mean(dim=0)

    a_tr = project(split.a_train, basis=ds.basis, mean=global_mean)
    b_tr = project(split.b_train, basis=ds.basis, mean=global_mean)
    a_te = project(split.a_test, basis=ds.basis, mean=global_mean)
    b_te = project(split.b_test, basis=ds.basis, mean=global_mean)

    # Train quadratic classifier in 3D: labels {-1,+1}
    z_tr = torch.cat([a_tr, b_tr], dim=0)
    y_tr = torch.cat([
        -torch.ones((int(a_tr.shape[0]),), device=device, dtype=dtype),
        torch.ones((int(b_tr.shape[0]),), device=device, dtype=dtype),
    ])

    model = fit_quadratic_ridge(z_tr, y_tr, l2=1e-2)

    # Test accuracy
    z_te = torch.cat([a_te, b_te], dim=0)
    y_te = torch.cat([
        -torch.ones((int(a_te.shape[0]),), device=device, dtype=dtype),
        torch.ones((int(b_te.shape[0]),), device=device, dtype=dtype),
    ])
    y_hat = model.predict(z_te)
    acc = accuracy(y_te, y_hat)

    # --- Decision boundary visualization ---
    # We visualize implicit surface f(x,y,z)=0 using a volumetric grid.
    # Plotly isosurface supports interactive rotation.

    # Determine bounds from train+test points (on GPU, then small transfer)
    all_pts = torch.cat([a_tr, b_tr, a_te, b_te], dim=0)
    mins = all_pts.min(dim=0).values
    maxs = all_pts.max(dim=0).values
    pad = 0.15 * (maxs - mins + 1e-6)
    mins = mins - pad
    maxs = maxs + pad

    nx = 40  # keep reasonable (grid^3)
    xs = _make_grid((float(mins[0]), float(maxs[0])), nx, device=device, dtype=dtype)
    ys = _make_grid((float(mins[1]), float(maxs[1])), nx, device=device, dtype=dtype)
    zs = _make_grid((float(mins[2]), float(maxs[2])), nx, device=device, dtype=dtype)

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)  # (M,3)
    vals = model.predict_logits(grid).reshape(nx, nx, nx)

    # Transfer only what plotly needs
    Xn, Yn, Zn = _to_numpy(X), _to_numpy(Y), _to_numpy(Z)
    Vn = _to_numpy(vals)

    fig = go.Figure()

    # Scatter points
    a_tr_n = _to_numpy(a_tr)
    b_tr_n = _to_numpy(b_tr)
    a_te_n = _to_numpy(a_te)
    b_te_n = _to_numpy(b_te)

    fig.add_trace(
        go.Scatter3d(x=a_tr_n[:, 0], y=a_tr_n[:, 1], z=a_tr_n[:, 2], mode="markers", name="A train", marker=dict(size=3, opacity=0.7))
    )
    fig.add_trace(
        go.Scatter3d(x=b_tr_n[:, 0], y=b_tr_n[:, 1], z=b_tr_n[:, 2], mode="markers", name="B train", marker=dict(size=3, opacity=0.7))
    )
    fig.add_trace(
        go.Scatter3d(
            x=a_te_n[:, 0],
            y=a_te_n[:, 1],
            z=a_te_n[:, 2],
            mode="markers",
            name="A test",
            marker=dict(size=5, opacity=0.9, symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=b_te_n[:, 0],
            y=b_te_n[:, 1],
            z=b_te_n[:, 2],
            mode="markers",
            name="B test",
            marker=dict(size=5, opacity=0.9, symbol="diamond"),
        )
    )

    # Isosurface (decision boundary)
    # Use a single isosurface at value=0
    fig.add_trace(
        go.Isosurface(
            x=Xn.reshape(-1),
            y=Yn.reshape(-1),
            z=Zn.reshape(-1),
            value=Vn.reshape(-1),
            isomin=0.0,
            isomax=0.0,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.35,
            showscale=False,
            name="f(x,y,z)=0",
        )
    )

    fig.update_layout(
        title=f"Query6: Quadratic decision boundary in diff-subspace (acc={acc:.3f}, eig={np.array2string(_to_numpy(evals), precision=4)})",
        scene=dict(xaxis_title="ds-1", yaxis_title="ds-2", zaxis_title="ds-3"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "query6_decision_boundary_3d.html"
    fig.write_html(out_path, include_plotlyjs="cdn")

    print("=== Query 6 ===")
    print(f"Device: {device}")
    print(f"Ambient D={d}, S1 dim k={s1.dim}, S2 dim k={s2.dim}, diffsubspace r={ds.dim}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
