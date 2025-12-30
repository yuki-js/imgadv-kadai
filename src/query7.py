"""Query 7 (GPU): mask search using canonical-angle separation score.

From further-discussion-prompt.md:
- Mask pattern changes separation in the 3D diff-subspace.
- Explore mask patterns to obtain good separation.
- Mask is based on 10x10 squares (24 of them).
- We evaluate separation quality in 3D (no classifier required).
- Mask generation and mask application must run on GPU.

This implementation uses canonical angles between the per-class subspaces (S1, S2)
constructed from masked train data.

Score (maximize):
    score = sum_i sin^2(theta_i)
where theta_i are the canonical angles between S1 and S2.
Intuition:
- If subspaces are identical -> theta_i ~ 0 -> score ~ 0 (bad separation).
- If subspaces are orthogonal -> theta_i ~ pi/2 -> score ~ k (good separation).

Search strategy (user instruction):
- Start with a set of 24 sliding-window squares (10x10) on a 32x32 grid.
- Run a multi-agent random-walk / SGD-like search:
  each agent proposes a random translation (dx,dy) with |dx|,|dy|<=6.
  The best candidate becomes the current state for that agent.

Run:
    python -m src.query7

Outputs:
    src/artifacts/query7_best_mask.bmp
    src/artifacts/query7_search_log.csv
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.gpu_subspace import fit_subspace
from src.gpu_utils import (
    apply_mask_32,
    downscale_flat_rgb_to_32,
    get_default_device,
    load_npy_to_device,
    torch_rng,
    train_test_split_indices,
)


Tensor = torch.Tensor


@dataclass(frozen=True)
class Candidate:
    shift_x: int
    shift_y: int
    score: float
    thetas_rad: tuple[float, float, float]


def _canonical_angles(u1: Tensor, u2: Tensor) -> Tensor:
    """Compute canonical angles between two k-dim subspaces.

    Args:
        u1: (D,k) orthonormal basis
        u2: (D,k) orthonormal basis

    Returns:
        theta: (k,) in [0, pi/2]
    """

    if u1.ndim != 2 or u2.ndim != 2:
        raise ValueError("u1/u2 must be 2D")
    if int(u1.shape[0]) != int(u2.shape[0]):
        raise ValueError("ambient dimension mismatch")
    if int(u1.shape[1]) != int(u2.shape[1]):
        raise ValueError("subspace dim mismatch")

    # Singular values of U1^T U2 are cos(theta_i)
    m = u1.transpose(0, 1) @ u2  # (k,k)
    s = torch.linalg.svdvals(m)
    s = torch.clamp(s, 0.0, 1.0)
    return torch.acos(s)


def _canon_angle_score(u1: Tensor, u2: Tensor) -> tuple[Tensor, Tensor]:
    """Return (score, thetas)."""

    thetas = _canonical_angles(u1, u2)
    score = torch.sum(torch.sin(thetas) ** 2)
    return score, thetas

def _base_anchors_24() -> list[tuple[int, int]]:
    """Return 24 (x,y) anchors for 10x10 squares on 32x32.

    We build 16 anchors on a 6px grid, then add 8 more anchors offset by +3px
    in x for the top two rows. This matches the 'slide by 6px' spirit while
    giving exactly 24 squares.

    Note:
        Query7 search can "hack" by shifting the entire mask so much that the
        kept-region centroid drifts off-center. We prevent this by enforcing a
        centroid-near-center constraint.
    """

    anchors: list[tuple[int, int]] = []

    # 16 anchors: x,y in {0,6,12,18}
    for y in (0, 6, 12, 18):
        for x in (0, 6, 12, 18):
            anchors.append((x, y))

    # 8 extra anchors: x in {3,9,15,21}, y in {0,6}
    for y in (0, 6):
        for x in (3, 9, 15, 21):
            anchors.append((x, y))

    if len(anchors) != 24:
        raise RuntimeError(f"Expected 24 anchors; got {len(anchors)}")
    return anchors


def _randomize_anchors_24(
    base: list[tuple[int, int]],
    *,
    square_size: int,
    g: torch.Generator,
    device: torch.device,
    jitter: int = 3,
) -> list[tuple[int, int]]:
    """Randomize the initial 24 anchors.

    User request: "initial box placement random".

    Strategy:
    - Start from the deterministic base anchors.
    - Add per-anchor integer jitter in [-jitter, +jitter].
    - Recentre anchors so their mean is near the image center (15.5,15.5).
    - Clip anchors to valid range so squares can exist: [0, 32-square_size].

    Notes:
        This randomization happens once per run and does not move image data
        between CPU/GPU.
    """

    s = int(square_size)
    if s <= 0 or s > 32:
        raise ValueError("square_size must be in [1,32]")
    j = int(jitter)
    if j < 0:
        raise ValueError("jitter must be >= 0")

    max_xy = 32 - s

    # jitter each anchor
    xs: list[int] = []
    ys: list[int] = []
    for (x, y) in base:
        if j == 0:
            dx = 0
            dy = 0
        else:
            dx = int(torch.randint(-j, j + 1, (1,), generator=g, device=device).item())
            dy = int(torch.randint(-j, j + 1, (1,), generator=g, device=device).item())
        xs.append(int(x) + dx)
        ys.append(int(y) + dy)

    # recentre (integer shift)
    mean_x = float(sum(xs)) / float(len(xs))
    mean_y = float(sum(ys)) / float(len(ys))
    shift_x = int(round(15.5 - mean_x))
    shift_y = int(round(15.5 - mean_y))

    out: list[tuple[int, int]] = []
    for x, y in zip(xs, ys, strict=True):
        xx = max(0, min(max_xy, int(x + shift_x)))
        yy = max(0, min(max_xy, int(y + shift_y)))
        out.append((xx, yy))

    if len(out) != 24:
        raise RuntimeError("anchor count mismatch")
    return out


def _make_mask_24_squares(
    *,
    anchors: list[tuple[int, int]],
    square_size: int,
    shift_x: int,
    shift_y: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Make a (32,32,1) mask with 24 squares union on GPU."""

    s = int(square_size)
    if s <= 0 or s > 32:
        raise ValueError("square_size must be in [1,32]")

    mask = torch.zeros((32, 32), device=device, dtype=dtype)

    for (ax, ay) in anchors:
        x = int(ax) + int(shift_x)
        y = int(ay) + int(shift_y)

        # clip to image bounds
        x1 = max(0, min(32, x))
        y1 = max(0, min(32, y))
        x2 = max(0, min(32, x + s))
        y2 = max(0, min(32, y + s))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

    return mask.unsqueeze(-1)  # (32,32,1)


def _mask_centroid_xy(mask32: Tensor) -> tuple[Tensor, Tensor]:
    """Compute (cx, cy) centroid of a mask on GPU.

    Args:
        mask32: (32,32,1) in [0,1]

    Returns:
        (cx, cy): centroid in pixel coordinates.

    Notes:
        Uses mask values as weights. If the mask is all zeros, returns center.
    """

    if tuple(mask32.shape) != (32, 32, 1):
        raise ValueError(f"mask32 must be (32,32,1); got shape={tuple(mask32.shape)}")

    w = mask32[:, :, 0]
    tot = w.sum()

    xs = torch.arange(32, device=mask32.device, dtype=mask32.dtype).view(1, 32).expand(32, 32)
    ys = torch.arange(32, device=mask32.device, dtype=mask32.dtype).view(32, 1).expand(32, 32)

    center = torch.tensor(15.5, device=mask32.device, dtype=mask32.dtype)
    cx = torch.where(tot > 0, (w * xs).sum() / tot, center)
    cy = torch.where(tot > 0, (w * ys).sum() / tot, center)
    return cx, cy


def _centroid_constraint_ok(
    mask32: Tensor,
    *,
    max_dist: float,
    target_x: float = 15.5,
    target_y: float = 15.5,
) -> tuple[bool, float, float, float]:
    """Hard constraint: centroid must be near the image center.

    Returns:
        (ok, cx, cy, dist)
    """

    cx_t, cy_t = _mask_centroid_xy(mask32)
    dx = cx_t - torch.tensor(float(target_x), device=mask32.device, dtype=mask32.dtype)
    dy = cy_t - torch.tensor(float(target_y), device=mask32.device, dtype=mask32.dtype)
    dist_t = torch.sqrt(dx * dx + dy * dy)

    cx = float(cx_t.detach().to("cpu").item())
    cy = float(cy_t.detach().to("cpu").item())
    dist = float(dist_t.detach().to("cpu").item())

    return dist <= float(max_dist), cx, cy, dist


def _rand_step(g: torch.Generator, *, max_step: int = 6, device: torch.device) -> tuple[int, int]:
    """Random integer step (dx,dy) where each component is in [-max_step, +max_step]."""

    ms = int(max_step)
    if ms < 0:
        raise ValueError("max_step must be >= 0")
    if ms == 0:
        return 0, 0

    # randint high is exclusive
    dx = int(torch.randint(-ms, ms + 1, (1,), generator=g, device=device).item())
    dy = int(torch.randint(-ms, ms + 1, (1,), generator=g, device=device).item())
    return dx, dy


def _evaluate_mask(
    *,
    a_train_raw32: Tensor,
    b_train_raw32: Tensor,
    mask32: Tensor,
    k: int,
) -> Candidate:
    """Apply mask on GPU, fit subspaces, compute canonical-angle score."""

    # mask application (GPU)
    a_train = apply_mask_32(a_train_raw32, mask32=mask32)
    b_train = apply_mask_32(b_train_raw32, mask32=mask32)

    s1 = fit_subspace(a_train, k=k, center=True)
    s2 = fit_subspace(b_train, k=k, center=True)

    score_t, thetas = _canon_angle_score(s1.basis, s2.basis)

    # small CPU transfer: only 1 score and 3 angles
    score = float(score_t.detach().to("cpu").item())
    th = thetas.detach().to("cpu").numpy().astype(np.float64)
    th_tuple = (float(th[0]), float(th[1]), float(th[2]))

    # shift is filled by caller
    return Candidate(shift_x=0, shift_y=0, score=score, thetas_rad=th_tuple)


def _save_mask_bmp(mask32: Tensor, *, path: Path) -> None:
    """Save (32,32,1) float mask as grayscale BMP (255=keep, 0=drop)."""

    m = mask32.detach().to("cpu").numpy()
    if m.shape != (32, 32, 1):
        raise ValueError(f"mask32 must be (32,32,1); got {m.shape}")

    img = (np.clip(m[:, :, 0], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(path)


def main() -> None:
    device = get_default_device()
    dtype = torch.float32

    # ---- search hyperparams (kept simple, edit here) ----
    seed = 114514
    test_ratio = 0.2

    # Smaller squares => more local feature selection
    square_size = 4

    # ~1.5x more trials
    agents = 32
    steps = 4096

    # Base step-size (actual step is randomized per proposal)
    max_step = 8

    # Add more randomness to increase success probability
    restart_prob = 0.12  # per-agent per-iteration random restart
    jump_prob = 0.10  # occasional larger jump
    jump_max_step = 12

    # Anti-hack constraint: allow wider movement while preventing extreme drift.
    centroid_max_dist = 4.0  # pixels

    # Load -> downscale to 32 on GPU once (no CPU-GPU ping-pong)
    a = load_npy_to_device("datasets/aoki.npy", device=device, dtype=dtype)
    b = load_npy_to_device("datasets/itadakimasu.npy", device=device, dtype=dtype)
    a = downscale_flat_rgb_to_32(a)
    b = downscale_flat_rgb_to_32(b)

    # Balance by truncation via random split indices (train only is used for score)
    g = torch_rng(seed, device=device)

    na = int(a.shape[0])
    nb = int(b.shape[0])
    n = min(na, nb)
    if na != n:
        a = a.index_select(0, torch.randperm(na, generator=g, device=device)[:n])
    if nb != n:
        b = b.index_select(0, torch.randperm(nb, generator=g, device=device)[:n])

    a_tr_idx, _a_te_idx = train_test_split_indices(n, test_ratio, g=g, device=device)
    b_tr_idx, _b_te_idx = train_test_split_indices(n, test_ratio, g=g, device=device)

    a_train_raw32 = a.index_select(0, a_tr_idx)
    b_train_raw32 = b.index_select(0, b_tr_idx)

    # subspace dim (k=3 as in Query6)
    n_train = min(int(a_train_raw32.shape[0]), int(b_train_raw32.shape[0]))
    d = int(a_train_raw32.shape[1])
    k = min(3, n_train - 1, d)
    if k <= 0:
        raise RuntimeError("Not enough training samples to construct a subspace")

    # Base anchors + randomize initial box placement per run (user request)
    anchors = _randomize_anchors_24(
        _base_anchors_24(),
        square_size=square_size,
        g=g,
        device=device,
        jitter=3,
    )

    out_dir = Path("src") / "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    log_path = out_dir / "query7_search_log.csv"
    best_mask_path = out_dir / "query7_best_mask.bmp"

    # Initialize agents at random shifts in [-6,6]
    agent_shifts: list[tuple[int, int]] = []
    for _ in range(int(agents)):
        dx0, dy0 = _rand_step(g, max_step=max_step, device=device)
        agent_shifts.append((dx0, dy0))

    # Per-agent best score (so we can do simple hill-climb acceptance).
    # Must be initialized outside the loop because some candidates may be skipped
    # by the centroid constraint before any evaluation happens.
    agent_scores = [float("-inf")] * int(agents)

    best_global: Candidate | None = None
    best_global_mask: Tensor | None = None

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iter",
                "agent",
                "shift_x",
                "shift_y",
                "score",
                "theta0_rad",
                "theta1_rad",
                "theta2_rad",
                "centroid_x",
                "centroid_y",
                "centroid_dist",
                "centroid_ok",
            ]
        )

        for it in range(int(steps)):
            for ai in range(int(agents)):
                cur_x, cur_y = agent_shifts[ai]
                step_dx, step_dy = _rand_step(g, max_step=max_step, device=device)
                prop_x = int(cur_x + step_dx)
                prop_y = int(cur_y + step_dy)

                mask32 = _make_mask_24_squares(
                    anchors=anchors,
                    square_size=square_size,
                    shift_x=prop_x,
                    shift_y=prop_y,
                    device=device,
                    dtype=dtype,
                )

                ok, cx, cy, cdist = _centroid_constraint_ok(mask32, max_dist=centroid_max_dist)
                if not ok:
                    w.writerow(
                        [
                            it,
                            ai,
                            prop_x,
                            prop_y,
                            "-inf",
                            "nan",
                            "nan",
                            "nan",
                            f"{cx:.8f}",
                            f"{cy:.8f}",
                            f"{cdist:.8f}",
                            0,
                        ]
                    )
                    continue

                cand = _evaluate_mask(a_train_raw32=a_train_raw32, b_train_raw32=b_train_raw32, mask32=mask32, k=k)
                cand = Candidate(shift_x=prop_x, shift_y=prop_y, score=cand.score, thetas_rad=cand.thetas_rad)

                w.writerow(
                    [
                        it,
                        ai,
                        cand.shift_x,
                        cand.shift_y,
                        f"{cand.score:.8f}",
                        *[f"{t:.8f}" for t in cand.thetas_rad],
                        f"{cx:.8f}",
                        f"{cy:.8f}",
                        f"{cdist:.8f}",
                        1,
                    ]
                )

                # Store per-agent best implicitly as its current shift; accept iff improved
                if cand.score > agent_scores[ai]:
                    agent_scores[ai] = cand.score
                    agent_shifts[ai] = (cand.shift_x, cand.shift_y)

                # Update global best
                if best_global is None or cand.score > best_global.score:
                    best_global = cand
                    best_global_mask = mask32

    if best_global is None or best_global_mask is None:
        raise RuntimeError("Search produced no candidates")

    _save_mask_bmp(best_global_mask, path=best_mask_path)

    print("=== Query 7 ===")
    print(f"Device: {device}")
    print(f"Train-only N: A={int(a_train_raw32.shape[0])}, B={int(b_train_raw32.shape[0])}, D={d}, k={k}")
    print(f"Square size: {square_size} ; squares: 24 ; agents={agents} ; steps={steps} ; max_step={max_step}")
    print(f"Best shift: ({best_global.shift_x}, {best_global.shift_y})")
    print(f"Best score: {best_global.score:.6f} ; thetas(rad)={best_global.thetas_rad}")
    print(f"Saved best mask: {best_mask_path}")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
