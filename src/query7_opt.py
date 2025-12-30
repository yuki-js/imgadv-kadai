"""Query 7 OPT (GPU) - optimize mask for *Query6 metric*.

User request:
- Align Query7_opt conditions to Query6.
- Query6 pipeline is fixed.

Therefore this optimizer searches mask parameters while evaluating *exactly the same*
train/test protocol and the same model family as Query6:
- balance -> split (fixed seed)
- build per-class subspaces on train
- build 3D difference-subspace
- train quadratic (2nd-order) classifier in 3D on train
- evaluate accuracy on test

Search space (user-approved extensions):
- Many squares (default: 12) of fixed size.
- Each square can move individually (local/random search).
- Mask generation and application are done on GPU.

Run:
    python -m src.query7_opt

Outputs:
    src/artifacts/query7_best_mask.bmp
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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

# --- Search config (tweak here) ---
N_SQUARES = 24
SQUARE_SIZE = 11
BASE_STEP = 4
SEED_SHIFT_STEP = 4
ITERS = 1600
JITTER_RANGE = 10  # per-update move in [-JITTER_RANGE, +JITTER_RANGE]

# Anti-edge-hack constraint (centroid of squares must stay away from borders).
CENTROID_MARGIN = 8.0

TEST_RATIO = 0.2
SEED = 0


def _clip_xy(xy: torch.Tensor, *, square_size: int) -> torch.Tensor:
    """Clip top-left coords so partially visible squares are allowed."""

    s = int(square_size)
    return torch.clamp(xy, min=-(s - 1), max=31)


def _centroid_ok(xy: torch.Tensor, *, square_size: int, margin: float) -> bool:
    """Return True if centroid of square centers stays away from borders.

    We compute centers assuming each square is fully inside the 32x32 frame
    by clamping top-left to [0, 32-square_size].
    """

    s = int(square_size)
    # (N,2) float
    xy_in = torch.clamp(xy.to(dtype=torch.float32), min=0.0, max=float(32 - s))
    centers = xy_in + (float(s) / 2.0)
    c = centers.mean(dim=0)  # (2,)
    m = float(margin)
    return bool((c[0] >= m) and (c[0] <= 32.0 - m) and (c[1] >= m) and (c[1] <= 32.0 - m))


def rasterize_squares(
    xy: torch.Tensor,
    *,
    square_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rasterize squares into a (32,32,1) mask on GPU."""

    s = int(square_size)
    if xy.ndim != 2 or int(xy.shape[1]) != 2:
        raise ValueError(f"xy must be (N,2); got {tuple(xy.shape)}")

    xy = xy.to(device=device)
    m = torch.zeros((32, 32), device=device, dtype=dtype)

    n = int(xy.shape[0])
    for i in range(n):
        x = int(xy[i, 0].item())
        y = int(xy[i, 1].item())
        x1 = max(0, min(32, x))
        y1 = max(0, min(32, y))
        x2 = max(0, min(32, x + s))
        y2 = max(0, min(32, y + s))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1.0

    return m.unsqueeze(-1)


def make_base_layout_grid(
    *,
    origin_top_left: tuple[int, int] = (6, 6),
    grid_w: int = 4,
    grid_h: int = 3,
    step: int = BASE_STEP,
) -> torch.Tensor:
    """Make a 4x3 grid (12 squares) base layout as top-left coords."""

    ox, oy = int(origin_top_left[0]), int(origin_top_left[1])
    step = int(step)

    pts: list[tuple[int, int]] = []
    for j in range(int(grid_h)):
        for i in range(int(grid_w)):
            pts.append((ox + i * step, oy + j * step))

    return torch.tensor(pts, dtype=torch.int64)


def eval_mask_query6_metric(
    *,
    a32: torch.Tensor,
    b32: torch.Tensor,
    mask32: torch.Tensor,
    g: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Return Query6-style test accuracy under given mask.

    IMPORTANT:
    This must match the fixed Query6(+bestmask) pipeline order:
      mask -> balance -> split -> fit -> test-eval
    """

    a = apply_mask_32(a32, mask32=mask32)
    b = apply_mask_32(b32, mask32=mask32)

    # Match Query6_bestmask: balance AFTER masking (and this consumes RNG state).
    a, b = downsample_to_match(a, b, g=g)

    a_tr_idx, a_te_idx = train_test_split_indices(int(a.shape[0]), TEST_RATIO, g=g, device=device)
    b_tr_idx, b_te_idx = train_test_split_indices(int(b.shape[0]), TEST_RATIO, g=g, device=device)

    a_train = a.index_select(0, a_tr_idx)
    a_test = a.index_select(0, a_te_idx)
    b_train = b.index_select(0, b_tr_idx)
    b_test = b.index_select(0, b_te_idx)

    n_train = min(int(a_train.shape[0]), int(b_train.shape[0]))
    d = int(a_train.shape[1])
    k = min(3, n_train - 1, d)
    if k <= 0:
        return 0.0

    s1 = fit_subspace(a_train, k=k, center=True)
    s2 = fit_subspace(b_train, k=k, center=True)
    ds, _evals = difference_subspace(s1, s2, r=3)

    global_mean = torch.cat([a_train, b_train], dim=0).mean(dim=0)

    a_tr = project(a_train, basis=ds.basis, mean=global_mean)
    b_tr = project(b_train, basis=ds.basis, mean=global_mean)
    a_te = project(a_test, basis=ds.basis, mean=global_mean)
    b_te = project(b_test, basis=ds.basis, mean=global_mean)

    z_tr = torch.cat([a_tr, b_tr], dim=0)
    y_tr = torch.cat(
        [
            -torch.ones((int(a_tr.shape[0]),), device=device, dtype=dtype),
            torch.ones((int(b_tr.shape[0]),), device=device, dtype=dtype),
        ]
    )

    model = fit_quadratic_ridge(z_tr, y_tr, l2=1e-2)

    z_te = torch.cat([a_te, b_te], dim=0)
    y_te = torch.cat(
        [
            -torch.ones((int(a_te.shape[0]),), device=device, dtype=dtype),
            torch.ones((int(b_te.shape[0]),), device=device, dtype=dtype),
        ]
    )

    y_hat = model.predict(z_te)
    return accuracy(y_te, y_hat)


def main() -> None:
    device = get_default_device()
    dtype = torch.float32

    # Load/downscale/balance once (GPU)
    a = load_npy_to_device("datasets/aoki.npy", device=device, dtype=dtype)
    b = load_npy_to_device("datasets/itadakimasu.npy", device=device, dtype=dtype)
    a = downscale_flat_rgb_to_32(a)
    b = downscale_flat_rgb_to_32(b)

    # Keep unbalanced tensors here; eval() will do balance after masking to match Query6_bestmask.
    g = torch_rng(SEED, device=device)

    # Base layout: grid. Default is 4 columns; rows derived from N_SQUARES.
    if N_SQUARES % 4 != 0:
        raise RuntimeError(f"N_SQUARES must be divisible by 4 for grid_w=4; got N_SQUARES={N_SQUARES}")
    grid_w = 4
    grid_h = N_SQUARES // grid_w

    base_xy = make_base_layout_grid(origin_top_left=(6, 6), grid_w=grid_w, grid_h=grid_h, step=BASE_STEP)
    if int(base_xy.shape[0]) != N_SQUARES:
        raise RuntimeError(f"base layout has {int(base_xy.shape[0])} squares but N_SQUARES={N_SQUARES}")

    best_xy = base_xy.clone()
    best_acc = -1.0

    jitter_choices = torch.arange(-JITTER_RANGE, JITTER_RANGE + 1, dtype=torch.int64)

    # Seed scan (coarse global shifts)
    for sy in range(0, 32, SEED_SHIFT_STEP):
        for sx in range(0, 32, SEED_SHIFT_STEP):
            xy = base_xy + torch.tensor([sx, sy], dtype=torch.int64)
            if not _centroid_ok(xy, square_size=SQUARE_SIZE, margin=CENTROID_MARGIN):
                continue
            mask32 = rasterize_squares(
                _clip_xy(xy, square_size=SQUARE_SIZE),
                square_size=SQUARE_SIZE,
                device=device,
                dtype=dtype,
            )
            # IMPORTANT: use deterministic g per candidate => clone generator state by re-seeding.
            g_cand = torch_rng(SEED, device=device)
            acc = eval_mask_query6_metric(a32=a, b32=b, mask32=mask32, g=g_cand, device=device, dtype=dtype)
            if acc > best_acc:
                best_acc = acc
                best_xy = xy.clone()

    print(
        "seed best Query6-test-acc="
        f"{best_acc:.4f}  squares={N_SQUARES}  square_size={SQUARE_SIZE}  jitter=±{JITTER_RANGE}"
    )

    # Local/random search: perturb one square at a time
    for t in range(int(ITERS)):
        xy = best_xy.clone()

        i = int(torch.randint(0, N_SQUARES, (1,), generator=g, device=device).item())
        dx = int(
            jitter_choices[
                int(torch.randint(0, len(jitter_choices), (1,), generator=g, device=device).item())
            ].item()
        )
        dy = int(
            jitter_choices[
                int(torch.randint(0, len(jitter_choices), (1,), generator=g, device=device).item())
            ].item()
        )
        xy[i, 0] += dx
        xy[i, 1] += dy

        if not _centroid_ok(xy, square_size=SQUARE_SIZE, margin=CENTROID_MARGIN):
            continue

        mask32 = rasterize_squares(
            _clip_xy(xy, square_size=SQUARE_SIZE),
            square_size=SQUARE_SIZE,
            device=device,
            dtype=dtype,
        )

        g_cand = torch_rng(SEED, device=device)
        acc = eval_mask_query6_metric(a32=a, b32=b, mask32=mask32, g=g_cand, device=device, dtype=dtype)

        if acc >= best_acc:
            best_acc = acc
            best_xy = xy

        if (t + 1) % 50 == 0:
            print(f"iter {t+1:04d}/{ITERS}  best Query6-test-acc={best_acc:.4f}")

    # Save best mask as BMP
    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)

    best_mask32 = rasterize_squares(
        _clip_xy(best_xy, square_size=SQUARE_SIZE),
        square_size=SQUARE_SIZE,
        device=device,
        dtype=dtype,
    )
    m = (best_mask32.squeeze(-1).detach().to("cpu").numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(m, mode="L")
    bmp_path = out_dir / "query7_best_mask.bmp"
    img.save(bmp_path)

    print("=== Query 7 OPT (Query6-metric) ===")
    print(f"Device: {device}")
    print(f"Best Query6-test-acc={best_acc:.4f}")
    print(f"square_size={SQUARE_SIZE}")
    print(f"n_squares={N_SQUARES}")
    print(f"jitter_range=±{JITTER_RANGE}")
    print(f"iters={ITERS}")
    print(f"Best xy={best_xy.tolist()}")
    print(f"Saved: {bmp_path}")


if __name__ == "__main__":
    main()
