"""Interactive 3D trajectory visualization for PCA scores."""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config

# Vivid red→...→red cyclic colorscale for time progression
HSV_RED_CYCLE = [[i / 12.0, f"hsl({int(i / 12.0 * 360)}, 100%, 50%)"] for i in range(13)]


def _load_scores() -> pd.DataFrame:
    df = pd.read_csv(config.PC_COMPONENTS_CSV)
    # Ensure PCs sorted by component name
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    df = df[["Date"] + pc_cols]
    return df


def _build_trajectory_fig(df: pd.DataFrame, comps: Tuple[int, int, int]) -> go.Figure:
    labels = [f"PC{int(c)}" for c in comps]
    for lab in labels:
        if lab not in df.columns:
            raise ValueError(f"Missing column {lab} in {config.PC_COMPONENTS_CSV}")
    X = df[labels[0]].to_numpy(dtype=float)
    Y = df[labels[1]].to_numpy(dtype=float)
    Z = df[labels[2]].to_numpy(dtype=float)
    T = len(df)
    cvals = np.linspace(0.0, 1.0, T)

    # Line trace (grey path)
    line = go.Scatter3d(
        x=X, y=Y, z=Z,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.6)", width=3),
        name="trajectory",
        showlegend=False,
    )
    # Scatter trace (colored by time)
    scatter = go.Scatter3d(
        x=X, y=Y, z=Z,
        mode="markers",
        marker=dict(
            size=3,
            color=cvals,
            colorscale=HSV_RED_CYCLE,
            cmin=0.0,
            cmax=1.0,
            opacity=0.9,
            colorbar=dict(title="Time progression"),
        ),
        name="samples",
        showlegend=False,
    )

    title = f"Trajectory in {labels[0]}–{labels[1]}–{labels[2]} space"
    fig = go.Figure(data=[line, scatter])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2],
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def show_pc_trajectory_interactive(components: Tuple[int, int, int] = (1, 2, 3)) -> None:
    df = _load_scores()
    fig = _build_trajectory_fig(df, components)
    fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D PCA trajectory viewer")
    parser.add_argument("--components", type=str, default="1,2,3",
                        help="Comma-separated components, e.g., 1,2,3 or 4,5,6")
    args = parser.parse_args()
    parts = [p.strip() for p in args.components.split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit("Provide exactly three components, e.g., --components 1,2,3")
    comps = tuple(int(p) for p in parts)
    show_pc_trajectory_interactive(comps)


if __name__ == "__main__":
    main()