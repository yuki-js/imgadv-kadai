"""End-to-end orchestrator for crypto PCA pipeline."""
from __future__ import annotations

import argparse
import os
import pandas as pd

from . import config, symbol_selection, data_collection, pca_analysis, visualization


def _exists(path: str) -> bool:
    return os.path.exists(path)


def orchestrate(force_select: bool = False,
               force_collect: bool = False,
               force_pca: bool = False,
               force_plot: bool = False) -> None:
    """Run symbol selection, data collection, PCA, and visualization in order."""
    config.ensure_dirs()

    print("== Step 1: Symbol selection ==")
    if force_select or not _exists(config.TOP_SYMBOLS_CSV):
        symbol_selection.verify_and_save(target_symbols=config.TARGET_SYMBOLS,
                                         outfile=config.SYMBOL_SELECTION_CSV,
                                         preview=20)
    else:
        print(f"Using existing: {config.TOP_SYMBOLS_CSV}")

    print("\n== Step 2: Data collection ==")
    if force_collect or not _exists(config.PRICES_CSV):
        out_prices = data_collection.collect_and_save_prices()
        print(f"Collected and saved prices: {out_prices}")
    else:
        print(f"Using existing: {config.PRICES_CSV}")

    print("\n== Step 3: PCA analysis ==")
    if force_pca or not (_exists(config.EXPLAINED_VARIANCE_CSV) and _exists(config.PC_COMPONENTS_CSV)):
        pc_components_df, explained_df = pca_analysis.run_pca_on_prices()
        print(f"Saved explained variance: {config.EXPLAINED_VARIANCE_CSV}")
        print(f"Saved PC components (temporal patterns): {config.PC_COMPONENTS_CSV}")
    else:
        print(f"Using existing: {config.EXPLAINED_VARIANCE_CSV} and {config.PC_COMPONENTS_CSV}")
        explained_df = pd.read_csv(config.EXPLAINED_VARIANCE_CSV)

    print("\n== Step 4: Visualization ==")
    if force_plot or not _exists(config.PC_PLOT_PATH):
        out_plot = visualization.plot_first_n_pcs(n=9)
        print(f"Saved PC grid plot: {out_plot}")
    else:
        print(f"Using existing: {config.PC_PLOT_PATH}")

    # Summary
    try:
        if 'explained_df' not in locals():
            explained_df = pd.read_csv(config.EXPLAINED_VARIANCE_CSV)
        cum = float(explained_df["cumulative_explained_variance"].iloc[-1])
        evr_list = explained_df["explained_variance_ratio"].round(4).tolist()
        print("\n== Results Summary ==")
        print(f"Explained variance ratios: {evr_list}")
        print(f"Cumulative explained variance (PC1..PC{len(evr_list)}): {cum:.4f}")
        print(f"PC plot: {config.PC_PLOT_PATH}")
    except Exception as e:
        print(f"Summary generation warning: {e}")

    print("\nPCA DONE")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end PCA pipeline.")
    parser.add_argument("--force-select", action="store_true", help="Force re-run symbol selection.")
    parser.add_argument("--force-collect", action="store_true", help="Force re-run data collection.")
    parser.add_argument("--force-pca", action="store_true", help="Force re-run PCA.")
    parser.add_argument("--force-plot", action="store_true", help="Force re-generate plot.")
    args = parser.parse_args()

    orchestrate(force_select=args.force_select,
                force_collect=args.force_collect,
                force_pca=args.force_pca,
                force_plot=args.force_plot)


if __name__ == "__main__":
    main()