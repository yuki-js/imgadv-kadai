"""Query 7 (GPU mask sweep)

Task (from further-discussion-prompt.md):
- Classification accuracy changes by mask size/pattern.
- Use masks based on four 12x12 squares.
- Shift mask by 8 pixels and explore patterns.
- For each mask pattern: build subspace classifier pipeline and evaluate test accuracy.
- Mask generation and application must be on GPU.

Implementation:
- For each (shift_x, shift_y) in {0,8,16} x {0,8,16}:
  - build mask32 on GPU
  - apply mask on GPU
  - build diff-subspace (3D)
  - fit quadratic classifier in that 3D space
  - evaluate test accuracy
- Save the best mask as an image (BMP only).

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
    make_mask_from_squares,
    torch_rng,
    train_test_split_indices,
)
from src.quadratic_classifier import accuracy, fit_quadratic_ridge


@dataclass(frozen=True)
class SweepResult:
    shift_x: int
    shift_y: int
    acc: float


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

    # Split once
    a_tr_idx, a_te_idx = train_test_split_indices(int(a.shape[0]), 0.2, g=g, device=device)
    b_tr_idx, b_te_idx = train_test_split_indices(int(b.shape[0]), 0.2, g=g, device=device)

    # Keep original (unmasked) splits on GPU; apply masks per sweep.
    a_train0 = a.index_select(0, a_tr_idx)
    a_test0 = a.index_select(0, a_te_idx)
    b_train0 = b.index_select(0, b_tr_idx)
    b_test0 = b.index_select(0, b_te_idx)

    # Sweep shifts: 0..16 step 8 (as instructed). 24 would make 2x2 block overflow (12*2=24).
    shifts = [0, 8, 16]

    results: list[SweepResult] = []

    for sy in shifts:
        for sx in shifts:
            mask32 = make_mask_from_squares(square_size=12, shift_x=sx, shift_y=sy, gap=0, device=device, dtype=dtype)

            a_train = apply_mask_32(a_train0, mask32=mask32)
            a_test = apply_mask_32(a_test0, mask32=mask32)
            b_train = apply_mask_32(b_train0, mask32=mask32)
            b_test = apply_mask_32(b_test0, mask32=mask32)

            n_train = min(int(a_train.shape[0]), int(b_train.shape[0]))
            d = int(a_train.shape[1])
            k = min(3, n_train - 1, d)
            if k <= 0:
                raise RuntimeError("Not enough training samples to construct a subspace")

            s1 = fit_subspace(a_train, k=k, center=True)
            s2 = fit_subspace(b_train, k=k, center=True)
            ds, _evals = difference_subspace(s1, s2, r=3)

            global_mean = torch.cat([a_train, b_train], dim=0).mean(dim=0)

            a_tr = project(a_train, basis=ds.basis, mean=global_mean)
            b_tr = project(b_train, basis=ds.basis, mean=global_mean)
            a_te = project(a_test, basis=ds.basis, mean=global_mean)
            b_te = project(b_test, basis=ds.basis, mean=global_mean)

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

            results.append(SweepResult(shift_x=sx, shift_y=sy, acc=acc))
            print(f"mask shift (x={sx}, y={sy}) -> acc={acc:.4f}")

    results_sorted = sorted(results, key=lambda r: r.acc, reverse=True)

    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)

    # Save best mask as a single BMP. Accuracies are printed only.
    if results_sorted:
        best = results_sorted[0]
        best_mask32 = make_mask_from_squares(
            square_size=12,
            shift_x=best.shift_x,
            shift_y=best.shift_y,
            gap=0,
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
        print(f"Best mask: shift_x={best.shift_x}, shift_y={best.shift_y}, acc={best.acc:.4f}")
        print(f"Saved: {out_dir / 'query7_best_mask.bmp'}")


if __name__ == "__main__":
    main()
