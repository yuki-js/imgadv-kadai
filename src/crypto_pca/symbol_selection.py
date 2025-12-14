"""Symbol selection and verification for top-N crypto assets from Binance.

This module:
- Pulls Binance exchangeInfo and 24hr ticker stats
- Filters to spot, TRADING, USDT-quoted pairs
- Excludes:
    * Leveraged tokens (UP, DOWN, BULL, BEAR suffixes on base)
    * Stablecoins and wrapped assets as base (e.g., USDT, USDC, BUSD, FDUSD, DAI, WBTC, WETH, WBETH, etc.)
- Ranks remaining symbols by 24h quoteVolume (descending) as a proxy for liquidity/popularity
- Produces a verification CSV containing both kept/excluded with reasons and the rank of kept symbols
- Prints a concise summary and highlights the Nth symbol

Usage:
    python src/crypto_pca/symbol_selection.py --target 100
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests

from . import config


# Exclude base assets that should not be included as constituents
EXCLUDE_BASE_ASSETS = {
    # Stablecoins and fiat-referenced tokens
    "USDT", "USDC", "BUSD", "TUSD", "FDUSD", "USDE", "DAI", "USDP", "PAX", "SUSD",
    "EUR", "TRY", "BRL", "UAH", "BIDR", "BKRW", "IDRT", "NGN", "RUB", "GBP", "AUD", "JPY", "ZAR", "PLN",
    "USD1", "XUSD", "AEUR", "EURI", "BFUSD",
    # Wrapped and synthetic representations
    "WBTC", "WETH", "WBETH",
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# Allowlist for bases that end with 'UP' but are NOT leveraged tokens
# e.g., JUP (Jupiter), SYRUP (governance), etc.
ALLOWLIST_NOT_LEVERAGED = {"JUP", "SYRUP"}

# Existence filter relative to 3-year start
EXISTENCE_TOLERANCE_DAYS = 7
START_DATE_MS = config.to_ms(config.START_DATE)
TOL_MS = EXISTENCE_TOLERANCE_DAYS * 24 * 60 * 60 * 1000


@dataclass
class SymbolInfo:
    symbol: str
    baseAsset: str
    quoteAsset: str
    status: str
    isSpotTradingAllowed: bool
    # filled later
    quoteVolume: float = 0.0
    excluded_reason: Optional[str] = None


def _request_json(path: str, params: Optional[Dict] = None) -> Dict:
    """HTTP GET wrapper with retries and backoff based on config."""
    url = f"{config.BASE_URL}{path}"
    last_err = None
    for attempt in range(1, config.RETRY_MAX + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(config.RETRY_BACKOFF * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(config.SLEEP_BETWEEN_CALLS + (config.RETRY_BACKOFF ** attempt))
    raise RuntimeError(f"Failed GET {url} after {config.RETRY_MAX} attempts: {last_err}")


def _fetch_usdt_spot_symbols() -> List[SymbolInfo]:
    """Fetch Binance exchangeInfo and return USDT-quoted, spot symbols as SymbolInfo items."""
    info = _request_json("/api/v3/exchangeInfo")
    out: List[SymbolInfo] = []
    for s in info.get("symbols", []):
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed", False)
        ):
            out.append(
                SymbolInfo(
                    symbol=s.get("symbol", ""),
                    baseAsset=s.get("baseAsset", ""),
                    quoteAsset="USDT",
                    status=s.get("status", ""),
                    isSpotTradingAllowed=bool(s.get("isSpotTradingAllowed", False)),
                )
            )
    return out


def _fetch_quote_volumes() -> Dict[str, float]:
    """Return a mapping of symbol -> 24h quoteVolume (float)."""
    tickers = _request_json("/api/v3/ticker/24hr")
    vol_map: Dict[str, float] = {}
    for t in tickers:
        sym = t.get("symbol")
        qv = t.get("quoteVolume", "0")
        try:
            vol_map[sym] = float(qv or 0.0)
        except Exception:
            vol_map[sym] = 0.0
    return vol_map

def _get_first_open_time_ms(symbol: str) -> Optional[int]:
    """Return earliest kline openTime (ms) for the given symbol, or None if not available."""
    try:
        data = _request_json("/api/v3/klines", params={
            "symbol": symbol,
            "interval": config.INTERVAL,
            "startTime": 0,
            "limit": 1,
        })
        if data:
            return int(data[0][0])
        return None
    except Exception:
        return None


def _is_leveraged_base(base: str) -> bool:
    """Determine whether a base asset name indicates a leveraged token, with allowlist overrides."""
    base_upper = base.upper()
    if base_upper in ALLOWLIST_NOT_LEVERAGED:
        return False
    # Common convention: e.g., BTCUP, BTCDOWN, ETHBULL, ETHBEAR
    for suf in LEVERAGED_SUFFIXES:
        if base_upper.endswith(suf) and len(base_upper) > len(suf):
            return True
    return False


def _annotate_exclusions(items: List[SymbolInfo]) -> None:
    """Annotate each SymbolInfo with excluded_reason if excluded."""
    for it in items:
        reasons = []
        if _is_leveraged_base(it.baseAsset):
            reasons.append("leveraged_token")
        if it.baseAsset in EXCLUDE_BASE_ASSETS:
            reasons.append("excluded_base_asset")
        # If any reason exists, mark excluded; else keep
        it.excluded_reason = ";".join(reasons) if reasons else None


def build_selection_dataframe() -> pd.DataFrame:
    """Build a DataFrame with columns:
        symbol, baseAsset, quoteAsset, status, isSpotTradingAllowed, quoteVolume, excluded_reason
    """
    symbols = _fetch_usdt_spot_symbols()
    vol_map = _fetch_quote_volumes()

    # Fill volumes and exclusion flags (exclude stablecoins/wrapped/leveraged bases)
    for it in symbols:
        it.quoteVolume = vol_map.get(it.symbol, 0.0)
    _annotate_exclusions(symbols)

    df = pd.DataFrame(
        [{
            "symbol": it.symbol,
            "baseAsset": it.baseAsset,
            "quoteAsset": it.quoteAsset,
            "status": it.status,
            "isSpotTradingAllowed": it.isSpotTradingAllowed,
            "quoteVolume": it.quoteVolume,
            "excluded_reason": it.excluded_reason or "kept",
        } for it in symbols]
    )

    # Rank only for kept
    kept_mask = df["excluded_reason"] == "kept"
    df_kept = df.loc[kept_mask].copy()
    df_kept = df_kept.sort_values("quoteVolume", ascending=False)
    df_kept["rank_by_volume"] = range(1, len(df_kept) + 1)

    # Merge rank back
    df = df.merge(
        df_kept[["symbol", "rank_by_volume"]],
        on="symbol",
        how="left"
    )
    return df.sort_values(["excluded_reason", "rank_by_volume", "quoteVolume"], ascending=[True, True, False])


def verify_and_save(target_symbols: int, outfile: Optional[str] = None, preview: int = 20) -> str:
    """Create selection DataFrame, print summary, and save CSV. Returns path."""
    config.ensure_dirs()
    df = build_selection_dataframe()

    kept = df[df["excluded_reason"] == "kept"].sort_values("quoteVolume", ascending=False).reset_index(drop=True)

    # Enforce existence at START_DATE (3 years ago) with tolerance
    accepted_rows: List[pd.Series] = []
    excluded_late_listed = 0

    for _, row in kept.iterrows():
        sym = row["symbol"]
        first_ms = _get_first_open_time_ms(sym)
        exists_at_start = (first_ms is not None) and (first_ms <= START_DATE_MS + TOL_MS)
        if exists_at_start:
            accepted_rows.append(row)
        else:
            excluded_late_listed += 1
        if len(accepted_rows) >= target_symbols:
            break

    top = pd.DataFrame(accepted_rows).copy()
    nth_idx = target_symbols - 1

    print("=== Symbol Selection Summary ===")
    print(f"Total USDT spot TRADING symbols: {len(df)}")
    print(f"Kept (after base/leveraged exclusions): {len(kept)}")
    print(f"Excluded due to listed after START_DATE (tolerance {EXISTENCE_TOLERANCE_DAYS}d): {excluded_late_listed}")

    print(f"\nTop {min(preview, len(top))} by 24h quoteVolume (kept & existed 3y ago):")
    if len(top) > 0:
        print(top[["rank_by_volume", "symbol", "baseAsset", "quoteVolume"]].head(preview).to_string(index=False))
    else:
        print("(none)")

    if len(top) > nth_idx:
        nth_row = top.iloc[nth_idx]
        print(f"\n{target_symbols}th asset (after 3y existence filter): "
              f"{nth_row['symbol']} (base={nth_row['baseAsset']}, quoteVolume={nth_row['quoteVolume']:.0f})")
    else:
        print(f"\nWarning: fewer than {target_symbols} symbols existed 3 years ago (accepted={len(top)}).")

    # Save CSV with both kept and excluded for auditability
    out_path = outfile or config.SYMBOL_SELECTION_CSV
    df.to_csv(out_path, index=False)
    print(f"\nSaved verification CSV: {out_path}")

    # Also save a concise top-N list for convenience
    top_path = config.TOP_SYMBOLS_CSV
    top[["rank_by_volume", "symbol", "baseAsset", "quoteVolume"]].to_csv(top_path, index=False)
    print(f"Saved top-N CSV: {top_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Verify Binance top-N symbols (USDT spot) and save selection CSV.")
    parser.add_argument("--target", type=int, default=config.TARGET_SYMBOLS, help="Number of top symbols to keep.")
    parser.add_argument("--outfile", type=str, default=None, help="Optional output CSV path for full selection.")
    parser.add_argument("--preview", type=int, default=20, help="Rows to preview in console for kept symbols.")
    args = parser.parse_args()

    verify_and_save(target_symbols=args.target, outfile=args.outfile, preview=args.preview)


if __name__ == "__main__":
    main()