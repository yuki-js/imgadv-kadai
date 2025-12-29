"""Query 6 (Matplotlib GUI版)

- Query6 と同じく、差分部分空間(3D)に射影した座標で二次分類器を学習。
- 3D散布図をMatplotlibで表示し、マウスオーバーで対応する 32x32x3 画像を右側にプレビュー。
- 決定境界(二次曲面)は Plotly の isosurface の方が描画が簡単なので、
  本GUI版は「hoverで画像確認」用途に絞る。

Run:
    python -m src.query6mpl
"""

from __future__ import annotations

import numpy as np
import torch

from src.gpu_subspace import difference_subspace, fit_subspace, project
from src.gpu_utils import get_default_device, load_balanced_split_gpu
from src.quadratic_classifier import accuracy, fit_quadratic_ridge


def _vec_to_rgb_image(v: np.ndarray, *, h: int = 32, w: int = 32) -> np.ndarray:
    """(D,) -> (H,W,3) in [0,1] (min-max per image)."""

    img = np.asarray(v, dtype=np.float64).reshape(h, w, 3)
    vmin = float(img.min())
    vmax = float(img.max())
    if np.isclose(vmax, vmin):
        return np.zeros((h, w, 3), dtype=np.float64)
    return (img - vmin) / (vmax - vmin)


def main() -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d import proj3d  # noqa: PLC0415

    device = get_default_device()
    dtype = torch.float32

    split = load_balanced_split_gpu(test_ratio=0.2, seed=0, device=device, dtype=dtype)

    n_train = min(int(split.a_train.shape[0]), int(split.b_train.shape[0]))
    d = int(split.a_train.shape[1])
    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples to construct a subspace")

    s1 = fit_subspace(split.a_train, k=k, center=True)
    s2 = fit_subspace(split.b_train, k=k, center=True)
    ds, _evals = difference_subspace(s1, s2, r=3)

    global_mean = torch.cat([split.a_train, split.b_train], dim=0).mean(dim=0)

    a_tr = project(split.a_train, basis=ds.basis, mean=global_mean)
    b_tr = project(split.b_train, basis=ds.basis, mean=global_mean)
    a_te = project(split.a_test, basis=ds.basis, mean=global_mean)
    b_te = project(split.b_test, basis=ds.basis, mean=global_mean)

    # Train + test acc (for display)
    z_tr = torch.cat([a_tr, b_tr], dim=0)
    y_tr = torch.cat([
        -torch.ones((int(a_tr.shape[0]),), device=device, dtype=dtype),
        torch.ones((int(b_tr.shape[0]),), device=device, dtype=dtype),
    ])
    model = fit_quadratic_ridge(z_tr, y_tr, l2=1e-2)

    z_te = torch.cat([a_te, b_te], dim=0)
    y_te = torch.cat([
        -torch.ones((int(a_te.shape[0]),), device=device, dtype=dtype),
        torch.ones((int(b_te.shape[0]),), device=device, dtype=dtype),
    ])
    y_hat = model.predict(z_te)
    acc = accuracy(y_te, y_hat)

    # Prepare hover payload (CPU side)
    coords = torch.cat([a_tr, b_tr, a_te, b_te], dim=0).detach().to("cpu").numpy()
    imgs_flat = torch.cat([split.a_train, split.b_train, split.a_test, split.b_test], dim=0).detach().to("cpu").numpy()
    imgs = np.stack([_vec_to_rgb_image(v) for v in imgs_flat], axis=0)

    meta: list[str] = (
        [f"A train #{i}" for i in range(int(a_tr.shape[0]))]
        + [f"B train #{i}" for i in range(int(b_tr.shape[0]))]
        + [f"A test  #{i}" for i in range(int(a_te.shape[0]))]
        + [f"B test  #{i}" for i in range(int(b_te.shape[0]))]
    )

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax_img = fig.add_subplot(1, 2, 2)

    ax.set_title(f"Query6mpl: diff-subspace (3D) + hover preview\n test acc={acc:.3f}")
    ax.set_xlabel("ds-1")
    ax.set_ylabel("ds-2")
    ax.set_zlabel("ds-3")

    # groups
    a_tr_n = a_tr.detach().to("cpu").numpy()
    b_tr_n = b_tr.detach().to("cpu").numpy()
    a_te_n = a_te.detach().to("cpu").numpy()
    b_te_n = b_te.detach().to("cpu").numpy()

    ax.scatter(a_tr_n[:, 0], a_tr_n[:, 1], a_tr_n[:, 2], s=10, alpha=0.7, label="A train")
    ax.scatter(b_tr_n[:, 0], b_tr_n[:, 1], b_tr_n[:, 2], s=10, alpha=0.7, label="B train")
    ax.scatter(a_te_n[:, 0], a_te_n[:, 1], a_te_n[:, 2], s=30, alpha=0.9, marker="D", label="A test")
    ax.scatter(b_te_n[:, 0], b_te_n[:, 1], b_te_n[:, 2], s=30, alpha=0.9, marker="D", label="B test")
    ax.legend(loc="upper left")

    # image panel
    ax_img.set_title("hover a point")
    im_artist = ax_img.imshow(np.zeros((32, 32, 3), dtype=np.float64))
    ax_img.axis("off")

    screen_xy: np.ndarray | None = None

    def _recompute_screen_coords() -> None:
        nonlocal screen_xy
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        x2, y2, _z2 = proj3d.proj_transform(xs, ys, zs, ax.get_proj())
        pts = np.column_stack([x2, y2])
        screen_xy = ax.transData.transform(pts)

    def _on_draw(_evt) -> None:
        _recompute_screen_coords()

    def _on_move(evt) -> None:
        if evt.inaxes is None or evt.inaxes != ax:
            return
        if screen_xy is None:
            _recompute_screen_coords()
        if screen_xy is None:
            return

        mx, my = float(evt.x), float(evt.y)
        d2 = (screen_xy[:, 0] - mx) ** 2 + (screen_xy[:, 1] - my) ** 2
        i = int(np.argmin(d2))
        if float(d2[i]) > 12.0**2:
            return

        im_artist.set_data(imgs[i])
        ax_img.set_title(meta[i])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("draw_event", _on_draw)
    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    _recompute_screen_coords()

    print("=== Query6mpl ===")
    print(f"Device: {device}")
    print(f"Ambient D={d}, k={k}, test acc={acc:.4f}")
    print("Move the mouse over points to preview images.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
