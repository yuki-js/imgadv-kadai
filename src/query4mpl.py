"""Query 4 (Matplotlib版)

差分部分空間へ射影した3次元点群を Matplotlib で可視化し、
マウスオーバーした点に対応する 32x32x3 画像を右側パネルに表示する。

Run:
    python -m src.query4mpl

Notes:
    - Matplotlib のGUIバックエンドが必要です（例: Windowsなら通常OK）。
    - 3Dなので、表示回転/ズームのたびに「3D点→画面座標」を更新して最近傍を拾います。
"""

from __future__ import annotations

import numpy as np

from src.dataload import DatasetConfig, load_balanced_split
from src.subspace_construction import SubspaceConstructor, difference_subspace


def _project(x: np.ndarray, *, basis: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project samples (N,D) to coordinates (N,r) via (x-mean) @ basis."""

    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    return (x - mean) @ basis


def _vec_to_rgb_image(v: np.ndarray, *, h: int = 32, w: int = 32) -> np.ndarray:
    """(D,) -> (H,W,3) in [0,1] (min-max per image)."""

    img = np.asarray(v, dtype=np.float64).reshape(h, w, 3)
    vmin = float(img.min())
    vmax = float(img.max())
    if np.isclose(vmax, vmin):
        return np.zeros((h, w, 3), dtype=np.float64)
    return (img - vmin) / (vmax - vmin)


def main() -> None:
    # Lazy imports (so query1/2 can run without mpl installed).
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d import proj3d  # noqa: PLC0415

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

    # Same coordinate system for both classes
    global_mean = np.concatenate([split.a_train, split.b_train], axis=0).mean(axis=0)

    a_tr = _project(split.a_train, basis=ds.basis, mean=global_mean)
    b_tr = _project(split.b_train, basis=ds.basis, mean=global_mean)
    a_te = _project(split.a_test, basis=ds.basis, mean=global_mean)
    b_te = _project(split.b_test, basis=ds.basis, mean=global_mean)

    # Prepare hover payload
    coords = np.concatenate([a_tr, b_tr, a_te, b_te], axis=0)  # (N,3)
    imgs = np.concatenate(
        [
            split.a_train,
            split.b_train,
            split.a_test,
            split.b_test,
        ],
        axis=0,
    )
    imgs = np.stack([_vec_to_rgb_image(v) for v in imgs], axis=0)  # (N,32,32,3)

    meta: list[str] = (
        [f"A train #{i}" for i in range(a_tr.shape[0])]
        + [f"B train #{i}" for i in range(b_tr.shape[0])]
        + [f"A test  #{i}" for i in range(a_te.shape[0])]
        + [f"B test  #{i}" for i in range(b_te.shape[0])]
    )

    # Figure layout: left 3D scatter, right image panel
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax_img = fig.add_subplot(1, 2, 2)

    ax.set_title(f"Query4 (mpl): Projection onto difference subspace\n eig={np.array2string(evals, precision=4)}")
    ax.set_xlabel("ds-1")
    ax.set_ylabel("ds-2")
    ax.set_zlabel("ds-3")

    # Plot groups
    ax.scatter(a_tr[:, 0], a_tr[:, 1], a_tr[:, 2], s=10, alpha=0.7, label="A train")
    ax.scatter(b_tr[:, 0], b_tr[:, 1], b_tr[:, 2], s=10, alpha=0.7, label="B train")
    ax.scatter(a_te[:, 0], a_te[:, 1], a_te[:, 2], s=30, alpha=0.9, marker="D", label="A test")
    ax.scatter(b_te[:, 0], b_te[:, 1], b_te[:, 2], s=30, alpha=0.9, marker="D", label="B test")
    ax.legend(loc="upper left")

    # Image panel init
    ax_img.set_title("hover a point")
    im_artist = ax_img.imshow(np.zeros((32, 32, 3), dtype=np.float64))
    ax_img.axis("off")

    # --- Hover logic ---
    # We maintain per-point screen coords (pixels) updated on draw.
    screen_xy: np.ndarray | None = None

    def _recompute_screen_coords() -> None:
        nonlocal screen_xy
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        x2, y2, _z2 = proj3d.proj_transform(xs, ys, zs, ax.get_proj())
        pts = np.column_stack([x2, y2])
        screen_xy = ax.transData.transform(pts)  # to display pixels

    def _on_draw(_evt) -> None:
        _recompute_screen_coords()

    def _on_move(evt) -> None:
        if evt.inaxes is None:
            return
        if evt.inaxes != ax:
            return
        if screen_xy is None:
            _recompute_screen_coords()
        if screen_xy is None:
            return

        mx, my = float(evt.x), float(evt.y)
        d2 = (screen_xy[:, 0] - mx) ** 2 + (screen_xy[:, 1] - my) ** 2
        i = int(np.argmin(d2))
        if float(d2[i]) > 12.0**2:  # threshold in pixels
            return

        im_artist.set_data(imgs[i])
        ax_img.set_title(meta[i])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("draw_event", _on_draw)
    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    # Initial compute
    _recompute_screen_coords()

    print("=== Query4mpl ===")
    print(f"Ambient D={d}, S1 dim k={s1.dim}, S2 dim k={s2.dim}, diffsubspace r={ds.dim}")
    print("Move the mouse over points to preview images.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
