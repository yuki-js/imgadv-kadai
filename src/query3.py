"""Query 3

2で得られた差分部分空間の基底ベクトルを画像として可視化する。

- 差分部分空間: A = P1 - P2 の固有ベクトル (|固有値|の大きい順)
- 基底ベクトルは 32x32x3 の画像にreshapeして表示・保存する。

Run:
    python -m src.query3

Outputs:
    src/artifacts/query3_basis_grid.png
    src/artifacts/query3_basis_0.png ...
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.dataload import DatasetConfig, load_balanced_split
from src.subspace_construction import SubspaceConstructor, difference_subspace


def _vec_to_rgb_image(v: np.ndarray, *, h: int = 32, w: int = 32) -> np.ndarray:
    """Convert a basis vector (D,) into an RGB image (H,W,3) in [0,1].

    Notes:
        - Basis vectors can have negative values; we min-max normalize per vector.
        - If the vector is (almost) constant, returns zeros.
    """

    v = np.asarray(v, dtype=np.float64).reshape(h, w, 3)
    vmin = float(v.min())
    vmax = float(v.max())
    if np.isclose(vmax, vmin):
        return np.zeros((h, w, 3), dtype=np.float64)
    return (v - vmin) / (vmax - vmin)


def main() -> None:
    cfg = DatasetConfig(test_ratio=0.2, seed=0)
    split = load_balanced_split(cfg)

    n_train = min(split.a_train.shape[0], split.b_train.shape[0])
    d = split.a_train.shape[1]

    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples to construct a subspace")

    ctor = SubspaceConstructor(k=k, center=True, svd="full")
    s1 = ctor.fit(split.a_train)
    s2 = ctor.fit(split.b_train)

    r = 3
    ds, evals = difference_subspace(s1, s2, r=r)

    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)

    # Lazy import so that query1/query2 can run even if matplotlib isn't installed.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    imgs = []
    for i in range(ds.basis.shape[1]):
        v = ds.basis[:, i]
        img = _vec_to_rgb_image(v)
        imgs.append(img)

        p = out_dir / f"query3_basis_{i}.png"
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"basis[{i}]  eig={evals[i]:+.4f}")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

    # Grid image
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.imshow(imgs[i])
        ax.axis("off")
        ax.set_title(f"eig={evals[i]:+.4f}")
    fig.suptitle("Query3: Difference-subspace basis vectors (min-max normalized)")
    fig.tight_layout()
    grid_path = out_dir / "query3_basis_grid.png"
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)

    print("=== Query 3 ===")
    print(f"Saved: {grid_path}")
    for i in range(n):
        print(f"Saved: {out_dir / f'query3_basis_{i}.png'}")


if __name__ == "__main__":
    main()
