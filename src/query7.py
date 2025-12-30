"""Query 7 (GPU mask sweep; updated spec)

Task (from further-discussion-prompt.md):
- Classification accuracy changes by mask pattern.
- Use a mask based on 7 squares of size 10x10.
- Shift the mask by 6 pixels and search for good accuracy.
- Build a classifier using the subspace method and evaluate accuracy on
  the *combined* data (train + test).
- Mask generation and application must be on GPU.

Mask layout (per user instruction):
- 7 squares: center + up/down/left/right + up-left + down-right.
- The "base" square top-left is derived from a center anchor at (11,11) in 32x32.
- The whole pattern is then shifted by (shift_x, shift_y) in multiples of 6 px.

Run:
    python -m src.query7

Outputs:
    src/artifacts/query7_best_mask.bmp
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
)
from src.quadratic_classifier import accuracy, fit_quadratic_ridge


@dataclass(frozen=True)
class SweepResult:
    shift_x: int
    shift_y: int
    acc: float


def _fill_square(mask: torch.Tensor, *, x: int, y: int, s: int) -> None:
    """In-place fill of a square keep-region into mask (32,32) on GPU."""

    x1 = max(0, min(32, int(x)))
    y1 = max(0, min(32, int(y)))
    x2 = max(0, min(32, int(x) + int(s)))
    y2 = max(0, min(32, int(y) + int(s)))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0


def make_mask_7_squares_10(
    *,
    shift_x: int,
    shift_y: int,
    center_anchor_xy: tuple[int, int] = (11, 11),
    square_size: int = 10,
    step: int = 10,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate (32,32,1) mask of 7 squares.

    Pattern (relative to center square top-left):
      - center
      - up/down/left/right (offset by step)
      - up-left (offset by (-step,-step))
      - down-right (offset by (+step,+step))

    Then apply global (shift_x, shift_y).
    """

    cx, cy = int(center_anchor_xy[0]), int(center_anchor_xy[1])
    s = int(square_size)
    step = int(step)

    # Treat (cx,cy) as top-left of the center square.
    x0 = cx + int(shift_x)
    y0 = cy + int(shift_y)

    m = torch.zeros((32, 32), device=device, dtype=dtype)

    offsets = [
        (0, 0),
        (0, -step),
        (0, +step),
        (-step, 0),
        (+step, 0),
        (-step, -step),
        (+step, +step),
    ]

    for dx, dy in offsets:
        _fill_square(m, x=x0 + dx, y=y0 + dy, s=s)

    return m.unsqueeze(-1)


def main() -> None:
    device = get_default_device()
    dtype = torch.float32

    # Load & downscale once (GPU) to minimize CPU<->GPU transfers.
    a = load_npy_to_device("datasets/aoki.npy", device=device, dtype=dtype)
    b = load_npy_to_device("datasets/itadakimasu.npy", device=device, dtype=dtype)
    a = downscale_flat_rgb_to_32(a)
    b = downscale_flat_rgb_to_32(b)

    # Balance once
    g = torch_rng(0, device=device)
    a, b = downsample_to_match(a, b, g=g)

    # "train+test" combined evaluation: we use all balanced samples.
    # (No train/test split here.)

    # Near-exhaustive scan (全数検査に近い): scan every pixel shift.
    # We clip squares at the image boundary, so shifts outside the valid area still behave deterministically.
    shifts = list(range(0, 32))  # 0..31

    results: list[SweepResult] = []

    for sy in shifts:
        for sx in shifts:
            mask32 = make_mask_7_squares_10(
                shift_x=sx,
                shift_y=sy,
                center_anchor_xy=(11, 11),
                square_size=10,
                step=10,
                device=device,
                dtype=dtype,
            )

            a_m = apply_mask_32(a, mask32=mask32)
            b_m = apply_mask_32(b, mask32=mask32)

            n_train = min(int(a_m.shape[0]), int(b_m.shape[0]))
            d = int(a_m.shape[1])

            # To "maximize" accuracy (on the combined set), tune subspace dim k and ridge l2.
            # Note: evaluation is on the same data used for fitting, per updated spec.
            k_candidates = [3, 5, 8, 12]
            k_candidates = [kk for kk in k_candidates if 1 <= kk <= min(n_train - 1, d)]
            if not k_candidates:
                raise RuntimeError("Not enough samples to construct a subspace")

            # Keep l2 > 0 to avoid singular normal equations.
            l2_candidates = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]

            best_acc_local = -1.0

            for k in k_candidates:
                s1 = fit_subspace(a_m, k=k, center=True)
                s2 = fit_subspace(b_m, k=k, center=True)
                ds, _evals = difference_subspace(s1, s2, r=3)

                global_mean = torch.cat([a_m, b_m], dim=0).mean(dim=0)

                a_z = project(a_m, basis=ds.basis, mean=global_mean)
                b_z = project(b_m, basis=ds.basis, mean=global_mean)

                # Fit and evaluate on the same combined set (per updated spec)
                z_all = torch.cat([a_z, b_z], dim=0)
                y_all = torch.cat(
                    [
                        -torch.ones((int(a_z.shape[0]),), device=device, dtype=dtype),
                        torch.ones((int(b_z.shape[0]),), device=device, dtype=dtype),
                    ]
                )

                for l2 in l2_candidates:
                    try:
                        model = fit_quadratic_ridge(z_all, y_all, l2=float(l2))
                    except torch._C._LinAlgError:
                        continue
                    y_hat = model.predict(z_all)
                    acc = accuracy(y_all, y_hat)
                    if acc > best_acc_local:
                        best_acc_local = acc

            acc = float(best_acc_local)

            results.append(SweepResult(shift_x=sx, shift_y=sy, acc=acc))
            # Logging (avoid flooding): once per row
            if sx == 0:
                best_so_far = max(r.acc for r in results)
                print(f"mask scan row y={sy:02d} ... best_so_far={best_so_far:.4f}")

    results_sorted = sorted(results, key=lambda r: r.acc, reverse=True)

    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)

    if results_sorted:
        best = results_sorted[0]
        best_mask32 = make_mask_7_squares_10(
            shift_x=best.shift_x,
            shift_y=best.shift_y,
            center_anchor_xy=(11, 11),
            square_size=10,
            step=10,
            device=device,
            dtype=dtype,
        )

        m = (best_mask32.squeeze(-1).detach().to("cpu").numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(m, mode="L")

        bmp_path = out_dir / "query7_best_mask.bmp"
        img.save(bmp_path)

    print("=== Query 7 ===")
    print(f"Device: {device}")
    if results_sorted:
        best = results_sorted[0]
        print(f"Best mask: shift_x={best.shift_x}, shift_y={best.shift_y}, acc(all)={best.acc:.4f}")
        print(f"Saved: {out_dir / 'query7_best_mask.bmp'}")


if __name__ == "__main__":
    main()
