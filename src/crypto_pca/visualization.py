"""PCA on log prices visualization utilities."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from . import config

# Paths
LOADINGS_CSV = os.path.join(config.OUTPUT_DIR, "pc_loadings_by_asset.csv")
STRENGTH_PLOT_PATH = os.path.join(config.OUTPUT_DIR, "component_strength_3d.png")


def _load_explained_variance() -> pd.DataFrame:
    try:
        df = pd.read_csv(config.EXPLAINED_VARIANCE_CSV)
        return df
    except Exception:
        return pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative_explained_variance"])


def _load_loadings() -> pd.DataFrame:
    df = pd.read_csv(LOADINGS_CSV)
    # Ensure sorted by rank_by_volume if present
    if "rank_by_volume" in df.columns:
        df = df.sort_values("rank_by_volume", ascending=True, kind="mergesort")
    return df

def _overlay_cluster_means(ax: plt.Axes, y: np.ndarray, N: int, b1: int, b2: int) -> None:
    """Overlay dashed horizontal lines for cluster means across asset order."""
    if N <= 1:
        return
    try:
        m1 = float(np.nanmean(y[:max(0, b1)]))
        m2 = float(np.nanmean(y[max(0, b1):max(0, b2)]))
        m3 = float(np.nanmean(y[max(0, b2):N]))
        ax.hlines(m1, 1, b1, colors="tab:green", linestyles="--", linewidth=1.2, alpha=0.85)
        ax.hlines(m2, b1, b2, colors="tab:orange", linestyles="--", linewidth=1.2, alpha=0.85)
        ax.hlines(m3, b2, N, colors="tab:red", linestyles="--", linewidth=1.2, alpha=0.85)
    except Exception:
        pass


def plot_first_n_pcs(n: int = 9, save_path: Optional[str] = None) -> str:
    """Plot first n PCs as asset load curves ordered by rank_by_volume.
    - Y: loading in log-price PCA space
    - X: asset order by rank_by_volume (market-cap proxy)
    Cluster boundaries drawn at 30 and 80 to indicate:
      [1-30]=magnificent_30, [31-80]=middle_price, [81-100]=shitcoin_20
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    load_df = _load_loadings()
    ev = _load_explained_variance()

    # Determine available PCs
    pc_cols = [c for c in load_df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC columns found in loadings CSV.")
    n = min(n, len(pc_cols))
    selected_pc_cols = pc_cols[:n]
    # Symmetric y-limit across PCs for comparability
    try:
        yabsmax = max(float(np.nanmax(np.abs(load_df[c].to_numpy(dtype=float)))) for c in selected_pc_cols)
    except Exception:
        yabsmax = None

    # x positions
    N = len(load_df)
    x = np.arange(1, N + 1, dtype=int)

    # cluster boundaries
    b1, b2 = 30, 80

    # Setup figure with constrained layout
    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for i in range(n):
        pc = f"PC{i+1}"
        y = load_df[pc].to_numpy(dtype=float)
        ax = axes[i]
        ax.plot(x, y, color="tab:blue", linewidth=1.4)

        # shaded regions by cluster
        ax.axvspan(1, b1, color="tab:green", alpha=0.08)
        ax.axvspan(b1, b2, color="tab:orange", alpha=0.08)
        ax.axvspan(b2, N, color="tab:red", alpha=0.08)

        # boundaries
        ax.axvline(b1 + 0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(b2 + 0.5, color="gray", linestyle="--", linewidth=0.8)

        # overlay cluster means
        _overlay_cluster_means(ax, y, N, b1, b2)

        # in-plot corner text (avoid axis titles)
        try:
            evr_val = float(ev.loc[ev["component"] == (i + 1), "explained_variance_ratio"].values[0])
            ax.text(0.02, 0.95, f"{pc}  EVR={evr_val:.2%}", transform=ax.transAxes,
                    fontsize=9, ha="left", va="top", alpha=0.9)
        except Exception:
            ax.text(0.02, 0.95, f"{pc}", transform=ax.transAxes,
                    fontsize=9, ha="left", va="top", alpha=0.9)

        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_xlim(1, N)
        if yabsmax and np.isfinite(yabsmax):
            ax.set_ylim(-yabsmax * 1.05, yabsmax * 1.05)

    # concise labels
    for j in range(6, 9):
        axes[j].set_xlabel("Rank")
    for j in range(0, 9, 3):
        axes[j].set_ylabel("Loading")

    out_path = save_path or config.PC_PLOT_PATH
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_component_strength_3d(n: int = 9, save_path: Optional[str] = None) -> str:
    """3D surface of component strength over time.
    Strength defined as squared PC scores (temporal components): S^2.
    Axes: X=time index, Y=component index, Z=strength.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    pc_df = pd.read_csv(config.PC_COMPONENTS_CSV)

    # Determine PCs available
    pc_cols = [c for c in pc_df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC columns found in PC components CSV.")
    n = min(n, len(pc_cols))
    pc_cols = pc_cols[:n]

    dates = pc_df["Date"].astype(str).tolist()
    T = len(dates)

    # Build Z matrix (n x T) of strengths
    Z = np.vstack([np.square(pc_df[c].to_numpy(dtype=float)) for c in pc_cols])
    X = np.arange(T, dtype=float)[None, :].repeat(n, axis=0)
    Y = (np.arange(1, n + 1, dtype=float)[:, None]).repeat(T, axis=1)

    # Plot
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, rstride=4, cstride=8)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Component")
    ax.set_zlabel("Strength (score^2)")

    # ticks: start, middle, end
    xticks_idx = [0, T // 2, T - 1]
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels([dates[i] for i in xticks_idx], rotation=20, ha="right")
    ax.set_yticks(np.arange(1, n + 1, dtype=int))

    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)

    out_path = save_path or STRENGTH_PLOT_PATH
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Consolidated in bottom CLI
    pass

# --- PC0 (mean) + PC1..PC9 grid plotting ---

def _load_mean_log_price_by_asset() -> pd.Series:
    """Compute per-asset mean of log prices across all dates from PRICES_CSV."""
    df = pd.read_csv(config.PRICES_CSV, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    # Handle non-positive values defensively
    df = df.mask(df <= 0).ffill(axis=1).bfill(axis=1)
    df = df.reindex(columns=sorted(df.columns))
    log_df = np.log(df.astype(float))
    mean_s = log_df.mean(axis=1)  # mean across dates -> Series(index=symbol)
    return mean_s


def plot_pc0_to_pc9(save_path: Optional[str] = None) -> str:
    """Plot PC0 (mean log price across time) + first nine PCs as asset curves ordered by rank_by_volume.
    3x4 grid: [PC0, PC1..PC9] with remaining slots hidden.
    - Y: loading (PCs) or mean log price (PC0)
    - X: asset order by rank_by_volume (market-cap proxy)
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    load_df = _load_loadings()
    ev = _load_explained_variance()

    # Ensure sorted by rank
    if "rank_by_volume" in load_df.columns:
        load_df = load_df.sort_values("rank_by_volume", ascending=True, kind="mergesort")
    symbols = load_df["symbol"].astype(str).tolist()

    # Prepare PC columns
    pc_cols = [c for c in load_df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PC columns found in loadings CSV.")
    # Take first 9 PCs
    pc_cols = pc_cols[:9]

    # Compute PC0 = mean log price per asset over time, aligned to symbols order
    mean_s = _load_mean_log_price_by_asset()
    y0 = mean_s.reindex(symbols).to_numpy(dtype=float)

    # x positions and cluster boundaries
    N = len(symbols)
    x = np.arange(1, N + 1, dtype=int)
    b1, b2 = 30, 80

    # Setup figure: 3 rows x 4 cols with constrained layout
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, constrained_layout=True)
    axes = axes.flatten()

    # Plot PC0 (mean) without axis title
    ax0 = axes[0]
    ax0.plot(x, y0, color="tab:purple", linewidth=1.4)
    ax0.axvspan(1, b1, color="tab:green", alpha=0.08)
    ax0.axvspan(b1, b2, color="tab:orange", alpha=0.08)
    ax0.axvspan(b2, N, color="tab:red", alpha=0.08)
    ax0.axvline(b1 + 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax0.axvline(b2 + 0.5, color="gray", linestyle="--", linewidth=0.8)
    _overlay_cluster_means(ax0, y0, N, b1, b2)
    ax0.text(0.02, 0.95, "PC0 (mean)", transform=ax0.transAxes,
             fontsize=9, ha="left", va="top", alpha=0.9)
    ax0.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax0.set_xlim(1, N)
    try:
        y0min, y0max = float(np.nanmin(y0)), float(np.nanmax(y0))
        pad = (y0max - y0min) * 0.05
        ax0.set_ylim(y0min - pad, y0max + pad)
    except Exception:
        pass

    # Symmetric y-limit across PCs for comparability
    try:
        yabsmax = max(float(np.nanmax(np.abs(load_df[c].to_numpy(dtype=float)))) for c in pc_cols)
    except Exception:
        yabsmax = None

    # Plot PC1..PC9 with in-plot corner labels (no axis titles)
    for i, pc in enumerate(pc_cols, start=1):
        y = load_df[pc].to_numpy(dtype=float)
        ax = axes[i]
        ax.plot(x, y, color="tab:blue", linewidth=1.4)
        ax.axvspan(1, b1, color="tab:green", alpha=0.08)
        ax.axvspan(b1, b2, color="tab:orange", alpha=0.08)
        ax.axvspan(b2, N, color="tab:red", alpha=0.08)
        ax.axvline(b1 + 0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(b2 + 0.5, color="gray", linestyle="--", linewidth=0.8)
        _overlay_cluster_means(ax, y, N, b1, b2)

        # label + EVR in corner
        try:
            evr_val = float(ev.loc[ev["component"] == (i), "explained_variance_ratio"].values[0])
            ax.text(0.02, 0.95, f"{pc}  EVR={evr_val:.2%}", transform=ax.transAxes,
                    fontsize=9, ha="left", va="top", alpha=0.9)
        except Exception:
            ax.text(0.02, 0.95, f"{pc}", transform=ax.transAxes,
                    fontsize=9, ha="left", va="top", alpha=0.9)

        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_xlim(1, N)
        if yabsmax and np.isfinite(yabsmax):
            ax.set_ylim(-yabsmax * 1.05, yabsmax * 1.05)

    # Hide any unused panels (12 total in 3x4)
    used = 1 + len(pc_cols)
    for k in range(used, 12):
        axes[k].axis("off")

    # Minimal axis labels to reduce clutter
    for j in range(8, 12):
        axes[j].set_xlabel("Rank")
    axes[0].set_ylabel("Mean log price")
    axes[4].set_ylabel("Loading")
    axes[8].set_ylabel("Loading")

    out_path = save_path or os.path.join(config.OUTPUT_DIR, "principal_components_pc0_pc9.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_pc_trajectory_3d(components: tuple = (1, 2, 3), save_path: Optional[str] = None) -> str:
    """
    Plot 3D trajectory of daily projections in the given PC subspace (components).
    Uses PC scores (pc_components.csv). Draws a grey path and color-coded scatter by time.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    pc_df = pd.read_csv(config.PC_COMPONENTS_CSV)

    labels = [f"PC{int(c)}" for c in components]
    for lab in labels:
        if lab not in pc_df.columns:
            raise RuntimeError(f"Component {lab} not found in PC components CSV.")

    X = pc_df[labels[0]].to_numpy(dtype=float)
    Y = pc_df[labels[1]].to_numpy(dtype=float)
    Z = pc_df[labels[2]].to_numpy(dtype=float)
    T = len(pc_df)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Base line for trajectory
    ax.plot3D(X, Y, Z, color="gray", linewidth=1.0, alpha=0.6)

    # Color-coded scatter along time progression
    cvals = np.linspace(0.0, 1.0, T)
    sc = ax.scatter(X, Y, Z, c=cvals, cmap=cm.hsv, vmin=0.0, vmax=1.0, s=14, alpha=0.9)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    # Minimal in-plot label to avoid layout overflow
    ax.text2D(0.02, 0.97, f"Trajectory in {labels[0]}-{labels[1]}-{labels[2]} space",
              transform=ax.transAxes, fontsize=10)

    cb = fig.colorbar(sc, shrink=0.65, aspect=14, pad=0.06)
    cb.set_label("Time progression")

    out_path = save_path or os.path.join(
        config.OUTPUT_DIR, f"pc_3d_pc{components[0]}_{components[1]}_{components[2]}.png"
    )
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Extend CLI entrypoint to also produce PC0..PC9 grid and 3D trajectories
if __name__ == "__main__":
    grid = plot_first_n_pcs(n=9)
    grid2 = plot_pc0_to_pc9()
    surface = plot_component_strength_3d(n=9)
    traj1 = plot_pc_trajectory_3d((1, 2, 3), os.path.join(config.OUTPUT_DIR, "pc_3d_pc1_3.png"))
    traj2 = plot_pc_trajectory_3d((4, 5, 6), os.path.join(config.OUTPUT_DIR, "pc_3d_pc4_6.png"))
    traj3 = plot_pc_trajectory_3d((7, 8, 9), os.path.join(config.OUTPUT_DIR, "pc_3d_pc7_9.png"))
    print(f"Saved PC grid plot (PC1..PC9): {grid}")
    print(f"Saved PC0..PC9 grid plot: {grid2}")
    print(f"Saved component strength 3D plot: {surface}")
    print(f"Saved PC1-3 trajectory: {traj1}")
    print(f"Saved PC4-6 trajectory: {traj2}")
    print(f"Saved PC7-9 trajectory: {traj3}")