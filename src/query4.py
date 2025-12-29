"""Query 4

各クラスのデータを差分部分空間へ射影して3次元プロットする（Plotly）。

Run:
    python -m src.query4

Outputs:
    src/artifacts/query4_diffsubspace_3d.html

Notes:
    - 差分部分空間は Query2 と同じく A = P1 - P2 の固有ベクトル（|固有値|降順）で構成。
    - 2クラスを同一座標系で比較しやすいよう、射影の前に「両クラスtrainを結合した全体平均」で中心化する。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.dataload import DatasetConfig, load_balanced_split
from src.subspace_construction import SubspaceConstructor, difference_subspace


def _project(x: np.ndarray, *, basis: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project samples (N,D) to coordinates (N,r) via (x-mean) @ basis."""

    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    return (x - mean) @ basis


def main() -> None:
    cfg = DatasetConfig(test_ratio=0.2, seed=0)
    split = load_balanced_split(cfg)

    # Build 3D subspaces S1, S2 from train sets
    n_train = min(split.a_train.shape[0], split.b_train.shape[0])
    d = split.a_train.shape[1]

    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples to construct a subspace")

    ctor = SubspaceConstructor(k=k, center=True, svd="full")
    s1 = ctor.fit(split.a_train)
    s2 = ctor.fit(split.b_train)

    # Difference subspace (3D)
    r = 3
    ds, evals = difference_subspace(s1, s2, r=r)

    # Use a global mean so both classes are compared in the same coordinate system.
    global_mean = np.concatenate([split.a_train, split.b_train], axis=0).mean(axis=0)

    a_tr = _project(split.a_train, basis=ds.basis, mean=global_mean)
    b_tr = _project(split.b_train, basis=ds.basis, mean=global_mean)
    a_te = _project(split.a_test, basis=ds.basis, mean=global_mean)
    b_te = _project(split.b_test, basis=ds.basis, mean=global_mean)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=a_tr[:, 0],
            y=a_tr[:, 1],
            z=a_tr[:, 2],
            mode="markers",
            name="A train",
            marker=dict(size=3, opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=b_tr[:, 0],
            y=b_tr[:, 1],
            z=b_tr[:, 2],
            mode="markers",
            name="B train",
            marker=dict(size=3, opacity=0.7),
        )
    )

    # Test points: slightly larger, with symbol difference
    fig.add_trace(
        go.Scatter3d(
            x=a_te[:, 0],
            y=a_te[:, 1],
            z=a_te[:, 2],
            mode="markers",
            name="A test",
            marker=dict(size=5, opacity=0.9, symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=b_te[:, 0],
            y=b_te[:, 1],
            z=b_te[:, 2],
            mode="markers",
            name="B test",
            marker=dict(size=5, opacity=0.9, symbol="diamond"),
        )
    )

    fig.update_layout(
        title=f"Query4: Projection onto difference subspace (eig={np.array2string(evals, precision=4)})",
        scene=dict(
            xaxis_title="ds-1",
            yaxis_title="ds-2",
            zaxis_title="ds-3",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "query4_diffsubspace_3d.html"
    fig.write_html(out_path, include_plotlyjs="cdn")

    print("=== Query 4 ===")
    print(f"Ambient D={d}, S1 dim k={s1.dim}, S2 dim k={s2.dim}, diffsubspace r={ds.dim}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
