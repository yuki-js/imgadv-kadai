"""Interactive 3D trajectory visualization using Matplotlib (GUI)."""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib

def _ensure_gui_backend() -> str:
    """Try to select an interactive GUI backend for Matplotlib on Windows."""
    for backend in ("Qt5Agg", "TkAgg", "WXAgg"):
        try:
            matplotlib.use(backend)
            return backend
        except Exception:
            continue
    # Fallback: try default import; if fails, use Agg
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return matplotlib.get_backend()
    except Exception:
        matplotlib.use("Agg")
        return "Agg"

BACKEND = _ensure_gui_backend()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from . import config

def _load_scores() -> pd.DataFrame:
    df = pd.read_csv(config.PC_COMPONENTS_CSV)
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    df = df[["Date"] + pc_cols]
    return df

def show_pc_trajectory_interactive_matplotlib(components: Tuple[int, int, int] = (1, 2, 3)) -> None:
    """Open a GUI window showing 3D trajectory in selected PC subspace using Matplotlib."""
    df = _load_scores()
    labels = [f"PC{int(c)}" for c in components]
    for lab in labels:
        if lab not in df.columns:
            raise ValueError(f"Missing column {lab} in {config.PC_COMPONENTS_CSV}")
    X = df[labels[0]].to_numpy(dtype=float)
    Y = df[labels[1]].to_numpy(dtype=float)
    Z = df[labels[2]].to_numpy(dtype=float)
    T = len(df)
    cvals = np.linspace(0.0, 1.0, T)

    fig = plt.figure(figsize=(11, 8))
    try:
        fig.canvas.manager.set_window_title(f"PCA Trajectory: {labels[0]}–{labels[1]}–{labels[2]} (backend={BACKEND})")
    except Exception:
        pass
    ax = fig.add_subplot(111, projection="3d")

    ax.plot3D(X, Y, Z, color="gray", linewidth=1.0, alpha=0.6, label="trajectory")
    sc = ax.scatter(X, Y, Z, c=cvals, cmap="hsv", vmin=0.0, vmax=1.0, s=10, alpha=0.9)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(f"Trajectory in {labels[0]}–{labels[1]}–{labels[2]} space", fontsize=10)

    cbar = fig.colorbar(sc, shrink=0.7, aspect=14, pad=0.06)
    try:
        cbar.set_label("Time progression")
    except Exception:
        pass

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D PCA trajectory viewer (Matplotlib)")
    parser.add_argument("--components", type=str, default="1,2,3",
                        help="Comma-separated components, e.g., 1,2,3 or 4,5,6")
    args = parser.parse_args()
    parts = [p.strip() for p in args.components.split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit("Provide exactly three components, e.g., --components 1,2,3")
    comps = tuple(int(p) for p in parts)
    show_pc_trajectory_interactive_matplotlib(comps)

if __name__ == "__main__":
    main()