"""Query 6 (GPU) with Query7 best mask applied.

- Loads the best mask produced by Query7/Query7_opt:
    src/artifacts/query7_best_mask.bmp
- Applies the mask on GPU, then runs the same Query6 pipeline:
  diff-subspace (3D) -> quadratic classifier -> decision boundary visualization.

Run:
    python -m src.query6_bestmask

Outputs:
    src/artifacts/query6_bestmask_decision_boundary_3d.html
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image

from src.gpu_subspace import difference_subspace, fit_subspace, project
from src.gpu_utils import (
    apply_mask_32,
    downsample_to_match,
    downscale_flat_rgb_to_32,
    get_default_device,
    load_npy_to_device,
    torch_rng,
    train_test_split_indices,
)
from src.quadratic_classifier import accuracy, fit_quadratic_ridge


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").numpy()


def _load_best_mask32(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Load src/artifacts/query7_best_mask.bmp into (32,32,1) float mask on GPU."""

    p = Path("src") / "artifacts" / "query7_best_mask.bmp"
    if not p.exists():
        raise FileNotFoundError(f"Best mask not found: {p} (run Query7/Query7_opt first)")

    img = Image.open(p).convert("L")
    if img.size != (32, 32):
        raise ValueError(f"Best mask must be 32x32; got {img.size} in {p}")

    m = (np.array(img, dtype=np.float32) / 255.0).reshape(32, 32, 1)
    mask32 = torch.as_tensor(m, device=device, dtype=dtype)
    return mask32


def main() -> None:
    device = get_default_device()
    dtype = torch.float32

    # Load/downscale once (GPU)
    a = load_npy_to_device("datasets/aoki.npy", device=device, dtype=dtype)
    b = load_npy_to_device("datasets/itadakimasu.npy", device=device, dtype=dtype)
    a = downscale_flat_rgb_to_32(a)
    b = downscale_flat_rgb_to_32(b)

    # Apply best mask (GPU)
    mask32 = _load_best_mask32(device=device, dtype=dtype)
    a = apply_mask_32(a, mask32=mask32)
    b = apply_mask_32(b, mask32=mask32)

    # Balance + split (GPU)
    g = torch_rng(0, device=device)
    a, b = downsample_to_match(a, b, g=g)

    a_tr_idx, a_te_idx = train_test_split_indices(int(a.shape[0]), 0.2, g=g, device=device)
    b_tr_idx, b_te_idx = train_test_split_indices(int(b.shape[0]), 0.2, g=g, device=device)

    a_train = a.index_select(0, a_tr_idx)
    a_test = a.index_select(0, a_te_idx)
    b_train = b.index_select(0, b_tr_idx)
    b_test = b.index_select(0, b_te_idx)

    # Subspaces + diff-subspace
    n_train = min(int(a_train.shape[0]), int(b_train.shape[0]))
    d = int(a_train.shape[1])
    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples to construct a subspace")

    s1 = fit_subspace(a_train, k=k, center=True)
    s2 = fit_subspace(b_train, k=k, center=True)
    ds, evals = difference_subspace(s1, s2, r=3)

    global_mean = torch.cat([a_train, b_train], dim=0).mean(dim=0)

    a_tr = project(a_train, basis=ds.basis, mean=global_mean)
    b_tr = project(b_train, basis=ds.basis, mean=global_mean)
    a_te = project(a_test, basis=ds.basis, mean=global_mean)
    b_te = project(b_test, basis=ds.basis, mean=global_mean)

    # Train quadratic classifier in 3D
    z_tr = torch.cat([a_tr, b_tr], dim=0)
    y_tr = torch.cat(
        [
            -torch.ones((int(a_tr.shape[0]),), device=device, dtype=dtype),
            torch.ones((int(b_tr.shape[0]),), device=device, dtype=dtype),
        ]
    )
    model = fit_quadratic_ridge(z_tr, y_tr, l2=1e-2)

    # Test accuracy
    z_te = torch.cat([a_te, b_te], dim=0)
    y_te = torch.cat(
        [
            -torch.ones((int(a_te.shape[0]),), device=device, dtype=dtype),
            torch.ones((int(b_te.shape[0]),), device=device, dtype=dtype),
        ]
    )
    y_hat = model.predict(z_te)
    acc = accuracy(y_te, y_hat)

    # Decision boundary via grid + isosurface
    all_pts = torch.cat([a_tr, b_tr, a_te, b_te], dim=0)
    mins = all_pts.min(dim=0).values
    maxs = all_pts.max(dim=0).values
    pad = 0.15 * (maxs - mins + 1e-6)
    mins = mins - pad
    maxs = maxs + pad

    nx = 40
    xs = torch.linspace(float(mins[0]), float(maxs[0]), nx, device=device, dtype=dtype)
    ys = torch.linspace(float(mins[1]), float(maxs[1]), nx, device=device, dtype=dtype)
    zs = torch.linspace(float(mins[2]), float(maxs[2]), nx, device=device, dtype=dtype)

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    vals = model.predict_logits(grid).reshape(nx, nx, nx)

    Xn, Yn, Zn = _to_numpy(X), _to_numpy(Y), _to_numpy(Z)
    Vn = _to_numpy(vals)

    fig = go.Figure()

    a_tr_n = _to_numpy(a_tr)
    b_tr_n = _to_numpy(b_tr)
    a_te_n = _to_numpy(a_te)
    b_te_n = _to_numpy(b_te)

    fig.add_trace(go.Scatter3d(x=a_tr_n[:, 0], y=a_tr_n[:, 1], z=a_tr_n[:, 2], mode="markers", name="A train", marker=dict(size=3, opacity=0.7)))
    fig.add_trace(go.Scatter3d(x=b_tr_n[:, 0], y=b_tr_n[:, 1], z=b_tr_n[:, 2], mode="markers", name="B train", marker=dict(size=3, opacity=0.7)))
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
        title=(
            "Query6 (bestmask): Quadratic decision boundary in diff-subspace "
            f"(acc={acc:.3f}, eig={np.array2string(_to_numpy(evals), precision=4)})"
        ),
        scene=dict(xaxis_title="ds-1", yaxis_title="ds-2", zaxis_title="ds-3"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "query6_bestmask_decision_boundary_3d.html"
    fig.write_html(out_path, include_plotlyjs="cdn")

    print("=== Query 6 (bestmask) ===")
    print(f"Device: {device}")
    print(f"Mask: {Path('src') / 'artifacts' / 'query7_best_mask.bmp'}")
    print(f"Ambient D={d}, k={k}, diffsubspace r=3")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
