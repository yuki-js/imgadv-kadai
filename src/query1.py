"""Query 1

２クラスの高次元ベクトルデータセットから部分空間S1とS2を構築し，
両者の成す正準角セットを求める。

Run:
    python -m src.query1
"""

from __future__ import annotations

import numpy as np

from src.dataload import DatasetConfig, load_balanced_split
from src.subspace_construction import SubspaceConstructor, canonical_angles, radians_to_degrees


def main() -> None:
    cfg = DatasetConfig(test_ratio=0.2, seed=0)
    split = load_balanced_split(cfg)

    # Build 3D subspaces S1, S2 from train sets.
    n_train = min(split.a_train.shape[0], split.b_train.shape[0])
    d = split.a_train.shape[1]
    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples to construct a subspace")

    ctor = SubspaceConstructor(k=k, center=True, svd="full")
    s1 = ctor.fit(split.a_train)
    s2 = ctor.fit(split.b_train)

    angles_rad, sv = canonical_angles(s1, s2)
    angles_deg = radians_to_degrees(angles_rad)

    print("=== Query 1 ===")
    print(f"A_train: {split.a_train.shape}, B_train: {split.b_train.shape}")
    print(f"Ambient D={d}, subspace dim k={k}")
    print("Canonical angles (degrees):")
    print(np.array2string(angles_deg, precision=4, separator=", "))
    print("Singular values (=cos angles):")
    print(np.array2string(sv, precision=6, separator=", "))


if __name__ == "__main__":
    main()
