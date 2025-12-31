"""Query 7 (GPU)

Parametric mask exploration + separation scoring in 3D space.

This script grows the minimal pipeline into a practical search:
- Build a *fixed* 3D projection space once (difference subspace from train, no mask).
- Search over more complex *parametric masks* (multi-shape composites) on GPU.
- Evaluate each mask by a two-stage score:
  1) Cheap: Fisher ratio on projected train points.
  2) Expensive: quadratic logistic classifier in 3D; evaluate test accuracy and
     boundary-near rate.
- Save the best-performing mask as a PNG (48x48 grayscale; white=keep).

Run:
  python src-gpu/query7.py

Artifacts:
  - src-gpu/artifacts/query7_scores.txt
  - src-gpu/artifacts/query7_best_mask.png

Notes:
- `src-gpu/` uses a hyphen, so we add this directory to `sys.path` locally.
- Everything heavy stays on GPU. Only small scalars and the final mask image are
  transferred to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import sys

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common  # noqa: E402
import dataload  # noqa: E402

# Filename has hyphen per assignment.
import importlib.util  # noqa: E402


def _import_subspace_construction():
    path = _THIS_DIR / "subspace-construction.py"
    spec = importlib.util.spec_from_file_location("subspace_construction_gpu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")

    # Ensure the module is registered in sys.modules before execution.
    # This is required for `@dataclass` to resolve `cls.__module__` reliably.
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _import_subspace_construction()

Tensor = torch.Tensor


# -------------------------
# Mask parameterization
# -------------------------


@dataclass(frozen=True)
class KeepEllipse:
    cx: float
    cy: float
    ax: float
    ay: float
    theta: float = 0.0  # radians


@dataclass(frozen=True)
class KeepRect:
    cx: float
    cy: float
    w: float
    h: float
    theta: float = 0.0


@dataclass(frozen=True)
class MaskComposite:
    """Composite mask.

    Final mask is:
      keep = union(keep_* shapes)
      drop = union(drop_* shapes)
      mask = clamp(keep - drop, 0, 1)

    This allows more complex shapes (e.g. keep face-oval but drop eye-band).
    """

    keep_ellipses: tuple[KeepEllipse, ...] = ()
    keep_rects: tuple[KeepRect, ...] = ()
    drop_ellipses: tuple[KeepEllipse, ...] = ()
    drop_rects: tuple[KeepRect, ...] = ()


def _mesh(h: int, w: int, *, device_: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    ys = torch.arange(h, device=device_, dtype=dtype)
    xs = torch.arange(w, device=device_, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return yy, xx


def keep_ellipse_mask(h: int, w: int, p: KeepEllipse, *, yy: Tensor, xx: Tensor) -> Tensor:
    """Return (h,w) keep mask in {0,1}."""

    device_ = xx.device
    dtype = xx.dtype

    cx = torch.tensor(float(p.cx), device=device_, dtype=dtype)
    cy = torch.tensor(float(p.cy), device=device_, dtype=dtype)
    ax = torch.tensor(float(p.ax), device=device_, dtype=dtype)
    ay = torch.tensor(float(p.ay), device=device_, dtype=dtype)
    th = torch.tensor(float(p.theta), device=device_, dtype=dtype)

    x = xx - cx
    y = yy - cy

    c = torch.cos(th)
    s = torch.sin(th)

    xr = c * x + s * y
    yr = -s * x + c * y

    r = (xr / (ax + 1e-6)) ** 2 + (yr / (ay + 1e-6)) ** 2
    return (r <= 1.0).to(dtype=dtype)


def keep_rect_mask(h: int, w: int, p: KeepRect, *, yy: Tensor, xx: Tensor) -> Tensor:
    """Return (h,w) keep mask in {0,1}."""

    device_ = xx.device
    dtype = xx.dtype

    cx = torch.tensor(float(p.cx), device=device_, dtype=dtype)
    cy = torch.tensor(float(p.cy), device=device_, dtype=dtype)
    hw = torch.tensor(float(p.w) / 2.0, device=device_, dtype=dtype)
    hh = torch.tensor(float(p.h) / 2.0, device=device_, dtype=dtype)
    th = torch.tensor(float(p.theta), device=device_, dtype=dtype)

    x = xx - cx
    y = yy - cy

    c = torch.cos(th)
    s = torch.sin(th)

    xr = c * x + s * y
    yr = -s * x + c * y

    inside = (torch.abs(xr) <= hw) & (torch.abs(yr) <= hh)
    return inside.to(dtype=dtype)


def composite_mask_flat(h: int, w: int, spec: MaskComposite, *, device_: torch.device, dtype: torch.dtype) -> Tensor:
    """Return flattened mask (D,) for composite keep/drop shapes.

    White(1)=keep, Black(0)=hide.
    """

    yy, xx = _mesh(h, w, device_=device_, dtype=dtype)

    keep = torch.zeros((h, w), device=device_, dtype=dtype)
    for e in spec.keep_ellipses:
        keep = torch.maximum(keep, keep_ellipse_mask(h, w, e, yy=yy, xx=xx))
    for r in spec.keep_rects:
        keep = torch.maximum(keep, keep_rect_mask(h, w, r, yy=yy, xx=xx))

    drop = torch.zeros((h, w), device=device_, dtype=dtype)
    for e in spec.drop_ellipses:
        drop = torch.maximum(drop, keep_ellipse_mask(h, w, e, yy=yy, xx=xx))
    for r in spec.drop_rects:
        drop = torch.maximum(drop, keep_rect_mask(h, w, r, yy=yy, xx=xx))

    m = torch.clamp(keep - drop, 0.0, 1.0)
    return m.reshape(-1)


def apply_mask(x: Tensor, m_flat: Tensor) -> Tensor:
    """Apply mask to flattened grayscale images.

    x: (N,D)
    m_flat: (D,) in {0,1}
    """

    if x.ndim != 2:
        raise ValueError(f"Expected (N,D); got {tuple(x.shape)}")
    if m_flat.ndim != 1 or m_flat.shape[0] != x.shape[1]:
        raise ValueError(f"Expected mask (D,); got {tuple(m_flat.shape)}")

    return x * m_flat.unsqueeze(0)


def save_mask_png_48(m_flat: Tensor, *, out_path: Path) -> None:
    """Save 48x48 grayscale PNG: 255=keep, 0=hide."""

    from PIL import Image

    m = m_flat.detach().reshape(48, 48).clamp(0, 1)
    m_u8 = (m * 255.0).to(dtype=torch.uint8).cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m_u8, mode="L").save(out_path)


# -------------------------
# Projection + scoring
# -------------------------


@dataclass(frozen=True)
class FixedProjection:
    basis: Tensor  # (D,3)
    mean: Tensor  # (D,)

    def project(self, x: Tensor) -> Tensor:
        return sc.project(x, basis=self.basis, mean=self.mean)


def build_fixed_projection(split: common.BalancedSplit, *, k: int = 3, r: int = 3) -> FixedProjection:
    """Build a fixed 3D difference-subspace using unmasked train data."""

    a_tr, b_tr = split.a_train, split.b_train

    n_train = min(int(a_tr.shape[0]), int(b_tr.shape[0]))
    d = int(a_tr.shape[1])
    k_eff = min(int(k), n_train - 1, d)
    if k_eff <= 0:
        raise RuntimeError("Not enough training samples to construct subspace")

    ctor = sc.SubspaceConstructor(k=k_eff, center=True)
    s1 = ctor.fit(a_tr)
    s2 = ctor.fit(b_tr)

    ds, _evals = sc.difference_subspace(s1, s2, r=r)

    global_mean = torch.cat([a_tr, b_tr], dim=0).mean(dim=0)
    return FixedProjection(basis=ds.basis, mean=global_mean)


def stack_labeled(x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [torch.zeros(x0.shape[0], device=x.device), torch.ones(x1.shape[0], device=x.device)],
        dim=0,
    )
    return x, y


def fit_quadratic_logistic(
    xtr3: Tensor,
    ytr01: Tensor,
    *,
    steps: int = 400,
    lr: float = 0.2,
    l2: float = 5e-4,
) -> Tensor:
    """Fit quadratic logistic model on train and return weight vector w (10,)."""

    phi_tr = common.quadratic_features_3d(xtr3)
    y = ytr01.to(dtype=phi_tr.dtype)

    w = torch.zeros(phi_tr.shape[1], device=phi_tr.device, dtype=phi_tr.dtype, requires_grad=True)
    opt = torch.optim.Adam([w], lr=float(lr))

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits = phi_tr @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + float(l2) * torch.sum(w * w)
        loss.backward()
        opt.step()

    return w.detach()


def quadratic_boundary_metrics(
    w: Tensor,
    x3: Tensor,
    y01: Tensor,
    *,
    delta: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Return (near_rate, accuracy) for quadratic boundary logits = phi(x)@w."""

    phi = common.quadratic_features_3d(x3)
    logits = phi @ w

    near = torch.mean((torch.abs(logits) < float(delta)).to(dtype=phi.dtype))
    pred = (logits >= 0).to(dtype=phi.dtype)
    acc = torch.mean((pred == y01.to(dtype=phi.dtype)).to(dtype=phi.dtype))
    return near, acc


@dataclass(frozen=True)
class Score:
    # separation
    fisher: float

    # classification quality
    acc_test: float
    bal_acc_test: float
    recall_a_test: float  # Aoki
    recall_b_test: float  # Itadakimasu

    # boundary / compactness
    near_rate: float
    var_a_train: float
    a_margin_q10_train: float


# -------------------------
# Search
# -------------------------


def random_mask(
    *,
    h: int,
    w: int,
    rng: torch.Generator,
    n_keep_ellipses: int,
    n_keep_rects: int,
    n_drop_ellipses: int,
    n_drop_rects: int,
) -> MaskComposite:
    """Sample a random composite mask (keep/drop).

    Heuristic ranges are tuned for faces roughly centered in 48x48.
    """

    def u(a: float, b: float) -> float:
        return float(torch.empty((), device="cpu").uniform_(a, b, generator=rng).item())

    # Smaller primitives to allow higher-complexity masks (requested)
    def ellipse() -> KeepEllipse:
        return KeepEllipse(
            cx=u(8.0, 40.0),
            cy=u(6.0, 42.0),
            ax=u(3.0, 14.0),
            ay=u(3.0, 16.0),
            theta=u(-math.pi / 3.0, math.pi / 3.0),
        )

    def rect() -> KeepRect:
        return KeepRect(
            cx=u(8.0, 40.0),
            cy=u(6.0, 42.0),
            w=u(4.0, 18.0),
            h=u(3.0, 16.0),
            theta=u(-math.pi / 3.0, math.pi / 3.0),
        )

    keep_ells = [ellipse() for _ in range(int(n_keep_ellipses))]
    keep_rects = [rect() for _ in range(int(n_keep_rects))]

    # Drop primitives: also small, for fine cut-outs
    drop_ells = [
        KeepEllipse(
            cx=u(10.0, 38.0),
            cy=u(8.0, 40.0),
            ax=u(2.0, 10.0),
            ay=u(2.0, 10.0),
            theta=u(-math.pi / 3.0, math.pi / 3.0),
        )
        for _ in range(int(n_drop_ellipses))
    ]
    drop_rects = [
        KeepRect(
            cx=u(10.0, 38.0),
            cy=u(8.0, 40.0),
            w=u(3.0, 14.0),
            h=u(2.0, 10.0),
            theta=u(-math.pi / 3.0, math.pi / 3.0),
        )
        for _ in range(int(n_drop_rects))
    ]

    # Ensure at least some area is kept.
    if not keep_ells and not keep_rects:
        keep_ells.append(KeepEllipse(cx=24.0, cy=24.0, ax=20.0, ay=24.0, theta=0.0))

    return MaskComposite(
        keep_ellipses=tuple(keep_ells),
        keep_rects=tuple(keep_rects),
        drop_ellipses=tuple(drop_ells),
        drop_rects=tuple(drop_rects),
    )


def evaluate_mask(
    spec: MaskComposite,
    *,
    split: common.BalancedSplit,
    proj: FixedProjection,
    h: int,
    w: int,
    device_: torch.device,
    dtype: torch.dtype,
    delta: float,
) -> tuple[Score, Tensor]:
    m = composite_mask_flat(h, w, spec, device_=device_, dtype=dtype)

    a_tr = apply_mask(split.a_train, m)
    b_tr = apply_mask(split.b_train, m)
    a_te = apply_mask(split.a_test, m)
    b_te = apply_mask(split.b_test, m)

    a_tr3 = proj.project(a_tr)
    b_tr3 = proj.project(b_tr)
    a_te3 = proj.project(a_te)
    b_te3 = proj.project(b_te)

    xtr3, ytr = stack_labeled(a_tr3, b_tr3)
    xte3, yte = stack_labeled(a_te3, b_te3)

    fisher_t = sc.fisher_ratio_score(xtr3, ytr)

    wv = fit_quadratic_logistic(xtr3, ytr)
    near_t, acc_t = quadratic_boundary_metrics(wv, xte3, yte, delta=delta)

    # Per-class recalls on test
    logits_te = common.quadratic_features_3d(xte3) @ wv
    pred01 = (logits_te >= 0).to(dtype=torch.long)
    yte01 = yte.to(dtype=torch.long)

    a_mask = yte01 == 0
    b_mask = yte01 == 1

    recall_a = torch.mean((pred01[a_mask] == 0).to(dtype=torch.float32))
    recall_b = torch.mean((pred01[b_mask] == 1).to(dtype=torch.float32))
    bal_acc = 0.5 * (recall_a + recall_b)

    # Aoki compactness on train: mean squared radius to class mean in 3D
    a_mu = a_tr3.mean(dim=0)
    a_center = a_tr3 - a_mu
    var_a = torch.mean(torch.sum(a_center * a_center, dim=1))

    # Aoki margin: Aoki wants negative logits, so margin = -logit (bigger is better)
    logits_a_tr = common.quadratic_features_3d(a_tr3) @ wv
    a_margin = (-logits_a_tr).to(dtype=torch.float32)
    a_margin_q10 = torch.quantile(a_margin, 0.10)

    return (
        Score(
            fisher=float(fisher_t),
            acc_test=float(acc_t),
            bal_acc_test=float(bal_acc),
            recall_a_test=float(recall_a),
            recall_b_test=float(recall_b),
            near_rate=float(near_t),
            var_a_train=float(var_a),
            a_margin_q10_train=float(a_margin_q10),
        ),
        m,
    )


def main() -> None:
    device_ = common.device()
    dtype = torch.float32

    cfg = dataload.DatasetConfig(test_ratio=0.2, seed=0)
    split = dataload.load_balanced_split(cfg, device_=device_, dtype=dtype)

    # Fixed projection (unmasked) to enable fast search.
    proj = build_fixed_projection(split, k=3, r=3)

    # Search config
    # Default iterations doubled (as requested).
    iters = int(os.environ.get("IMGADV_Q7_ITERS", "2000"))
    topk = int(os.environ.get("IMGADV_Q7_TOPK", "20"))
    target_acc = float(os.environ.get("IMGADV_Q7_TARGET_ACC", "0.90"))
    delta = float(os.environ.get("IMGADV_Q7_DELTA", "0.25"))

    # NOTE: torch.Generator for randint/uniform in eager mode is CPU-only.
    # We only use it to sample scalar mask parameters; heavy ops remain on GPU.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.seed) + 123)

    h, w = cfg.out_h, cfg.out_w

    # Keep a leaderboard prioritizing Aoki stability.
    best: tuple[Score, MaskComposite, Tensor] | None = None
    leaderboard: list[tuple[Score, MaskComposite]] = []

    def sort_key(s: Score) -> tuple[float, float, float, float, float, float]:
        return (
            min(s.recall_a_test, s.recall_b_test),
            s.bal_acc_test,
            s.acc_test,
            s.fisher,
            -s.var_a_train,
            -s.near_rate,
        )

    def better(a: Score, b: Score) -> bool:
        return sort_key(a) > sort_key(b)

    for i in range(iters):
        # Grow complexity (more shapes + allow cutouts)
        # Use CPU for random sampling.
        # Increase primitive counts (more complex masks)
        n_keep_ell = int(torch.randint(9, 25, (), generator=gen, device="cpu").item())
        n_keep_rec = int(torch.randint(6, 22, (), generator=gen, device="cpu").item())

        # Suppress drop counts to avoid overly fragmented masks
        n_drop_ell = int(torch.randint(0, 6, (), generator=gen, device="cpu").item())
        n_drop_rec = int(torch.randint(0, 5, (), generator=gen, device="cpu").item())

        spec = random_mask(
            h=h,
            w=w,
            rng=gen,
            n_keep_ellipses=n_keep_ell,
            n_keep_rects=n_keep_rec,
            n_drop_ellipses=n_drop_ell,
            n_drop_rects=n_drop_rec,
        )

        score, m = evaluate_mask(
            spec,
            split=split,
            proj=proj,
            h=h,
            w=w,
            device_=device_,
            dtype=dtype,
            delta=delta,
        )

        leaderboard.append((score, spec))
        leaderboard.sort(key=lambda t: sort_key(t[0]), reverse=True)
        leaderboard = leaderboard[:topk]

        if best is None or better(score, best[0]):
            best = (score, spec, m)

        if (i + 1) % 50 == 0:
            s0 = leaderboard[0][0]
            print(
                f"iter={i+1}/{iters} best_minrec={min(s0.recall_a_test,s0.recall_b_test):.4f} "
                f"bal={s0.bal_acc_test:.4f} acc={s0.acc_test:.4f} recA={s0.recall_a_test:.4f} "
                f"varA={s0.var_a_train:.4f} fisher={s0.fisher:.6g} near={s0.near_rate:.4f}"
            )

    if best is None:
        raise RuntimeError("Search produced no candidates")

    best_score, best_spec, best_mask = best

    out_dir = _THIS_DIR / "artifacts"
    os.makedirs(out_dir, exist_ok=True)

    # Save score table
    lines: list[str] = []
    lines.append(f"device={device_}, dtype={dtype}")
    lines.append(f"iters={iters}, topk={topk}, target_acc={target_acc}, delta={delta}")
    lines.append("rank\tmin_recall\tbal_acc\tacc_test\trecA\trecB\tvarA\tmarginA_q10\tFisherJ\tnear_rate\tmask")

    for rnk, (s, spec) in enumerate(leaderboard, start=1):
        lines.append(
            f"{rnk}\t{min(s.recall_a_test,s.recall_b_test):.6g}\t{s.bal_acc_test:.6g}\t{s.acc_test:.6g}\t"
            f"{s.recall_a_test:.6g}\t{s.recall_b_test:.6g}\t{s.var_a_train:.6g}\t{s.a_margin_q10_train:.6g}\t"
            f"{s.fisher:.6g}\t{s.near_rate:.6g}\t"
            f"keepE={len(spec.keep_ellipses)} keepR={len(spec.keep_rects)} dropE={len(spec.drop_ellipses)} dropR={len(spec.drop_rects)}"
        )

    score_path = out_dir / "query7_scores.txt"
    score_path.write_text("\n".join(lines), encoding="utf-8")

    # Save best mask PNG
    mask_path = out_dir / "query7_best_mask.png"
    save_mask_png_48(best_mask, out_path=mask_path)

    print("=== Query 7 (GPU) ===")
    print(
        "Best: "
        f"min_recall={min(best_score.recall_a_test,best_score.recall_b_test):.4f}, "
        f"bal_acc={best_score.bal_acc_test:.4f}, acc_test={best_score.acc_test:.4f}, "
        f"recA={best_score.recall_a_test:.4f}, recB={best_score.recall_b_test:.4f}, "
        f"varA={best_score.var_a_train:.4f}, marginA_q10={best_score.a_margin_q10_train:.4f}, "
        f"fisher={best_score.fisher:.6g}, near={best_score.near_rate:.4f}"
    )
    print(f"Saved: {score_path}")
    print(f"Saved: {mask_path}")

    if best_score.acc_test < target_acc:
        print(
            "WARNING: target acc not reached. Increase iterations via IMGADV_Q7_ITERS (e.g. 3000) "
            "or adjust delta/regularization."
        )


if __name__ == "__main__":
    main()
