"""Query 4.1

差分部分空間(3D)へ射影した特徴に対して、単純な回帰(最小二乗)で
2クラスを分割する「平面(2次元平面=決定境界)」を学習し、test accuracyを出す。

モデル:
  y_hat = w^T x + b   (x in R^3)
  trainでは y in {0,1} に対して最小二乗で (w,b) を推定
  推論では threshold=0.5 で 0/1 に丸める

Run:
  python -m src.query4_1

Mask:
  [`src/dataload.py`](src/dataload.py:1) と同様に環境変数 IMGADV_MASK_FILE で
  マスクPNGを指定できる。

Outputs:
  - 標準出力に train/test 指標
"""

from __future__ import annotations

import numpy as np

from src.dataload import DatasetConfig, load_balanced_split
from src.subspace_construction import SubspaceConstructor, difference_subspace


def _project(x: np.ndarray, *, basis: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project samples (N,D) to coordinates (N,3) via (x-mean) @ basis."""

    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    return (x - mean) @ basis


def _fit_linear_regression_plane(x3: np.ndarray, y01: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit y ≈ w^T x + b by least squares.

    Returns:
      w: (3,)
      b: scalar
    """

    x3 = np.asarray(x3, dtype=np.float64)
    y01 = np.asarray(y01, dtype=np.float64).reshape(-1)

    if x3.ndim != 2 or x3.shape[1] != 3:
        raise ValueError(f"Expected (N,3); got {x3.shape}")
    if y01.shape[0] != x3.shape[0]:
        raise ValueError(f"y length mismatch: {y01.shape[0]} vs {x3.shape[0]}")

    # Design matrix with bias
    x_aug = np.concatenate([x3, np.ones((x3.shape[0], 1), dtype=np.float64)], axis=1)  # (N,4)

    # Solve min ||Xb - y||_2
    beta, *_rest = np.linalg.lstsq(x_aug, y01, rcond=None)
    w = beta[:3].copy()
    b = float(beta[3])
    return w, b


def _predict(w: np.ndarray, b: float, x3: np.ndarray, *, thr: float = 0.5) -> np.ndarray:
    x3 = np.asarray(x3, dtype=np.float64)
    scores = x3 @ np.asarray(w, dtype=np.float64).reshape(3) + float(b)
    return (scores >= float(thr)).astype(np.int64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")

    acc = float(np.mean(y_true == y_pred))

    # class-conditional recalls
    a = y_true == 0
    b = y_true == 1
    rec_a = float(np.mean(y_pred[a] == 0)) if int(np.sum(a)) > 0 else float("nan")
    rec_b = float(np.mean(y_pred[b] == 1)) if int(np.sum(b)) > 0 else float("nan")

    bal = float(0.5 * (rec_a + rec_b)) if (not np.isnan(rec_a) and not np.isnan(rec_b)) else float("nan")

    return {"acc": acc, "recall_a": rec_a, "recall_b": rec_b, "bal_acc": bal}


def main() -> None:
    cfg = DatasetConfig(test_ratio=0.2, seed=0)
    split = load_balanced_split(cfg)

    # Fit per-class subspaces (k=3) using train
    n_train = min(split.a_train.shape[0], split.b_train.shape[0])
    d = split.a_train.shape[1]

    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples")

    ctor = SubspaceConstructor(k=k, center=True, svd="full")
    s1 = ctor.fit(split.a_train)
    s2 = ctor.fit(split.b_train)

    ds, evals = difference_subspace(s1, s2, r=3)

    # Global mean for a shared coordinate system
    global_mean = np.concatenate([split.a_train, split.b_train], axis=0).mean(axis=0)

    a_tr3 = _project(split.a_train, basis=ds.basis, mean=global_mean)
    b_tr3 = _project(split.b_train, basis=ds.basis, mean=global_mean)
    a_te3 = _project(split.a_test, basis=ds.basis, mean=global_mean)
    b_te3 = _project(split.b_test, basis=ds.basis, mean=global_mean)

    x_tr = np.concatenate([a_tr3, b_tr3], axis=0)
    y_tr = np.concatenate(
        [np.zeros((a_tr3.shape[0],), dtype=np.int64), np.ones((b_tr3.shape[0],), dtype=np.int64)],
        axis=0,
    )

    x_te = np.concatenate([a_te3, b_te3], axis=0)
    y_te = np.concatenate(
        [np.zeros((a_te3.shape[0],), dtype=np.int64), np.ones((b_te3.shape[0],), dtype=np.int64)],
        axis=0,
    )

    w, b = _fit_linear_regression_plane(x_tr, y_tr)

    pred_tr = _predict(w, b, x_tr)
    pred_te = _predict(w, b, x_te)

    m_tr = _metrics(y_tr, pred_tr)
    m_te = _metrics(y_te, pred_te)

    print("=== Query 4.1 ===")
    print(f"Ambient D={d}, diffsubspace r=3, eig={np.array2string(evals, precision=6)}")
    print(f"Plane params: w={np.array2string(w, precision=6)}, b={b:.6g}")
    print("Train metrics:")
    print(f"  acc={m_tr['acc']:.6g}  bal_acc={m_tr['bal_acc']:.6g}  recA={m_tr['recall_a']:.6g}  recB={m_tr['recall_b']:.6g}")
    print("Test metrics:")
    print(f"  acc={m_te['acc']:.6g}  bal_acc={m_te['bal_acc']:.6g}  recA={m_te['recall_a']:.6g}  recB={m_te['recall_b']:.6g}")


if __name__ == "__main__":
    main()
