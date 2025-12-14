"""PCA on log prices for crypto dataset."""
from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config

# Output for per-asset loadings (n_assets x n_components)
LOADINGS_CSV = os.path.join(config.OUTPUT_DIR, "pc_loadings_by_asset.csv")


def _load_prices_log() -> pd.DataFrame:
    """Load prices CSV (rows=symbol, cols=YYYY-MM-DD) and return log(price) DataFrame."""
    df = pd.read_csv(config.PRICES_CSV, index_col=0)
    # ensure numeric and strictly positive
    df = df.apply(pd.to_numeric, errors="coerce")
    if (df <= 0).any().any():
        # replace non-positive with NaN (shouldn't happen for prices) and forward/backward fill along rows
        df = df.mask(df <= 0).ffill(axis=1).bfill(axis=1)
    # sort columns (dates) asc
    df = df.reindex(columns=sorted(df.columns))
    log_df = np.log(df.astype(float))
    return log_df


def _prepare_rank_and_clusters(assets: pd.Index) -> pd.DataFrame:
    """Return a DataFrame with symbol, rank_by_volume, and cluster for provided assets."""
    try:
        top = pd.read_csv(config.TOP_SYMBOLS_CSV)
        top["symbol"] = top["symbol"].astype(str)
    except Exception:
        # create minimal mapping if top symbols CSV missing
        top = pd.DataFrame({"symbol": list(assets), "rank_by_volume": list(range(1, len(assets)+1))})
    # keep only needed columns
    cols = [c for c in ("symbol", "rank_by_volume") if c in top.columns]
    top = top[cols].drop_duplicates(subset=["symbol"])
    # restrict and order to assets
    m = pd.DataFrame({"symbol": list(assets)})
    out = m.merge(top, on="symbol", how="left")
    # fill missing ranks with large numbers to push to end
    out["rank_by_volume"] = out["rank_by_volume"].fillna(1e9).astype(float)
    # cluster labels
    def _cluster(rank: float) -> str:
        if rank <= 30:
            return "magnificent_30"
        if rank <= 80:
            return "middle_price"
        if rank <= 100:
            return "shitcoin_20"
        return "others"
    out["cluster"] = out["rank_by_volume"].apply(_cluster)
    return out


def run_pca_on_prices(n_components: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run PCA on log prices.

    Returns:
        pc_time_df: DataFrame (Date, PC1..PCk) temporal principal components (scores)
        explained_df: DataFrame (component, explained_variance_ratio, cumulative_explained_variance)
    """
    config.ensure_dirs()
    log_prices = _load_prices_log()
    # M: n_days x n_assets
    M = log_prices.T
    dates = list(M.index)
    assets = list(M.columns)

    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False)
    scores = pca.fit_transform(M.values)  # shape (T, K)
    comps = pca.components_               # shape (K, N_features=N_assets)
    evr = pca.explained_variance_ratio_

    # Temporal components (scores)
    n_comp = scores.shape[1]
    # create column names
    pc_cols = [f"PC{i}" for i in range(1, n_comp+1)]
    pc_time_df = pd.DataFrame(scores, index=dates, columns=pc_cols)
    pc_time_df.index.name = "Date"
    pc_time_df.to_csv(config.PC_COMPONENTS_CSV)

    # Explained variance CSV
    explained_df = pd.DataFrame({
        "component": list(range(1, n_comp+1)),
        "explained_variance_ratio": evr,
    })
    explained_df["cumulative_explained_variance"] = explained_df["explained_variance_ratio"].cumsum()
    explained_df.to_csv(config.EXPLAINED_VARIANCE_CSV, index=False)

    # Asset loadings L: n_assets x n_components
    L = comps.T  # (N, K)
    loadings_df = pd.DataFrame(L, index=assets, columns=pc_cols)
    loadings_df.index.name = "symbol"
    # attach rank and clusters
    meta = _prepare_rank_and_clusters(loadings_df.index)
    merged = meta.merge(loadings_df.reset_index(), on="symbol", how="left")
    # order by rank
    merged = merged.sort_values("rank_by_volume", ascending=True, kind="mergesort")
    merged.to_csv(LOADINGS_CSV, index=False)

    return pc_time_df, explained_df


if __name__ == "__main__":
    pc_time_df, explained_df = run_pca_on_prices()
    print(f"Saved: {config.PC_COMPONENTS_CSV}")
    print(f"Saved: {config.EXPLAINED_VARIANCE_CSV}")
    print(f"Saved: {LOADINGS_CSV}")