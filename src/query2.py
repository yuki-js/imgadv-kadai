"""Query 2

S1とS2の差分部分空間を求める。

Run:
    python -m src.query2
"""

from __future__ import annotations

import numpy as np

from src.dataload import DatasetConfig, load_balanced_split
from src.subspace_construction import SubspaceConstructor, difference_subspace


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

    # Compute 3D difference subspace
    r = 3
    ds, evals = difference_subspace(s1, s2, r=r)

    print("=== Query 2 ===")
    print(f"A_train: {split.a_train.shape}, B_train: {split.b_train.shape}")
    print(f"Ambient D={d}, S1 dim k={s1.dim}, S2 dim k={s2.dim}")
    print(f"Difference subspace dim r={ds.dim}")
    print("Selected eigenvalues of (P1 - P2) (sorted by |eig| desc):")
    print(np.array2string(evals, precision=6, separator=", "))

    # Sanity: basis orthonormality
    gram = ds.basis.T @ ds.basis
    print("Orthonormality check (Ds^T Ds):")
    print(np.array2string(gram, precision=6, separator=", "))


if __name__ == "__main__":
    main()
