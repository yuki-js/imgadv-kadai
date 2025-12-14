"""Data collection from Binance for daily closing prices (last ~6 years).

This module:
- Selects top USDT spot symbols by 24h quote volume as a proxy for high-liquidity assets
- Fetches 1d klines (close prices) between START_DATE and END_DATE from config
- Builds a dataset with rows=symbols, columns=YYYY-MM-DD, values=close price
- Saves to CSV at config.PRICES_CSV

Notes:
- Binance does not expose market cap rankings; we approximate with 24h quote volume.
- We exclude leveraged tokens (UP/DOWN/BULL/BEAR) and require spot trading to be allowed.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm

from . import config


INTERVAL_MS = 24 * 60 * 60 * 1000  # 1d in ms
CANDIDATE_MULTIPLIER = 3  # fetch this many candidates vs target
MIN_DAYS_THRESHOLD = 250  # minimal days to accept a symbol to avoid excessive gaps


def _request_json(path: str, params: Optional[Dict] = None) -> Dict:
    """HTTP GET wrapper with retries and backoff."""
    url = f"{config.BASE_URL}{path}"
    last_err = None
    for attempt in range(1, config.RETRY_MAX + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 429:
                # Too many requests: back off harder
                sleep_s = config.RETRY_BACKOFF * attempt
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            sleep_s = (config.SLEEP_BETWEEN_CALLS + config.RETRY_BACKOFF ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed GET {url} after {config.RETRY_MAX} attempts: {last_err}")


def _get_usdt_spot_symbols() -> List[Dict]:
    """Return exchangeInfo symbols filtered to USDT spot, trading, excluding leveraged tokens."""
    info = _request_json("/api/v3/exchangeInfo")
    symbols = []
    for s in info.get("symbols", []):
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed", False)
        ):
            sym = s.get("symbol", "")
            # Exclude leveraged/ETF-style symbols
            if any(tag in sym for tag in ["UP", "DOWN", "BULL", "BEAR"]):
                continue
            symbols.append(s)
    return symbols


def _rank_symbols_by_24h_quote_volume(usdt_symbols: List[Dict]) -> List[str]:
    """Rank given USDT symbols by 24h quoteVolume descending and return symbol list."""
    tickers = _request_json("/api/v3/ticker/24hr")
    vol_map: Dict[str, float] = {}
    allowed = {s["symbol"] for s in usdt_symbols}
    for t in tickers:
        sym = t.get("symbol")
        if sym in allowed:
            try:
                vol_map[sym] = float(t.get("quoteVolume", "0") or 0.0)
            except Exception:
                vol_map[sym] = 0.0
    ranked = sorted(vol_map.items(), key=lambda kv: kv[1], reverse=True)
    return [sym for sym, _ in ranked]


def load_top_symbols(path: Optional[str] = None, n: int = config.TARGET_SYMBOLS) -> Optional[List[str]]:
    """Load preselected top-N symbols from CSV produced by symbol_selection."""
    try:
        csv_path = path or config.TOP_SYMBOLS_CSV
        df = pd.read_csv(csv_path)
        if "symbol" not in df.columns:
            return None
        symbols = df["symbol"].astype(str).tolist()
        if n is not None and n > 0:
            symbols = symbols[:n]
        return symbols
    except Exception:
        return None


def _fetch_klines_all(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = config.INTERVAL,
    limit: int = config.LIMIT,
) -> List[List]:
    """Fetch klines for a symbol across [start_ms, end_ms] with pagination."""
    out: List[List] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        data = _request_json("/api/v3/klines", params=params)
        if not data:
            break
        out.extend(data)
        last_open = data[-1][0]
        next_cursor = last_open + INTERVAL_MS
        # Protection against no progress
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(config.SLEEP_BETWEEN_CALLS)
        # If we fetched fewer than limit, we likely reached end
        if len(data) < limit:
            break
    return out


def _klines_to_series(symbol: str, klines: List[List]) -> pd.Series:
    """Convert klines to a pandas Series of closes with date (YYYY-MM-DD) string columns.
    Fill missing days via time-based interpolation during collection."""
    if not klines:
        return pd.Series(dtype="float64", name=symbol)
    dates: List[str] = []
    closes: List[float] = []
    for k in klines:
        open_time = int(k[0])
        # Keep only bars strictly within the config window (inclusive)
        if open_time < config.to_ms(config.START_DATE) or open_time > config.to_ms(config.END_DATE):
            continue
        dt = config.from_ms(open_time).date().isoformat()  # 'YYYY-MM-DD'
        close = float(k[4])
        dates.append(dt)
        closes.append(close)
    if not dates:
        return pd.Series(dtype="float64", name=symbol)
    # Base series (string date index)
    s = pd.Series(data=closes, index=dates, name=symbol, dtype="float64")
    # Deduplicate any duplicate dates by keeping last
    s = s[~pd.Index(s.index).duplicated(keep="last")]
    # Build full daily date range and interpolate missing values
    start_date = config.START_DATE.date().isoformat()
    end_date = config.END_DATE.date().isoformat()
    full_idx_dt = pd.date_range(start=start_date, end=end_date, freq="D")
    s_dt = pd.Series(s.values, index=pd.to_datetime(s.index), name=symbol, dtype="float64")
    s_dt = s_dt.reindex(full_idx_dt)
    # Interpolate by time, then forward/backward fill edges
    s_dt = s_dt.interpolate(method="time").ffill().bfill()
    # Convert index back to 'YYYY-MM-DD' strings
    s_dt.index = [d.date().isoformat() for d in s_dt.index]
    return s_dt.sort_index()


def collect_prices_dataframe(target_symbols: int = config.TARGET_SYMBOLS, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Collect daily close prices for provided symbols (preferred) or top USDT pairs by volume;
    return DataFrame with rows=symbols, cols=dates."""
    config.ensure_dirs()
    start_ms = config.to_ms(config.START_DATE)
    end_ms = config.to_ms(config.END_DATE)

    # Prefer pre-selected list from symbol_selection (ensures existed 3y ago)
    symbol_list: Optional[List[str]] = symbols
    if symbol_list is None:
        symbol_list = load_top_symbols(n=target_symbols)

    if symbol_list:
        candidates = list(dict.fromkeys(symbol_list))  # preserve order, unique
    else:
        usdt_symbols = _get_usdt_spot_symbols()
        ranked = _rank_symbols_by_24h_quote_volume(usdt_symbols)
        # Take more than needed to account for sparse histories
        candidates = ranked[: max(target_symbols * CANDIDATE_MULTIPLIER, target_symbols)]

    series_map: Dict[str, pd.Series] = {}

    with tqdm(total=len(candidates), desc="Fetching klines", unit="sym") as pbar:
        for sym in candidates:
            try:
                kl = _fetch_klines_all(sym, start_ms, end_ms, interval=config.INTERVAL, limit=config.LIMIT)
                s = _klines_to_series(sym, kl)
                # Only accept if enough days to be useful; else skip
                if s.size >= MIN_DAYS_THRESHOLD:
                    series_map[sym] = s
            except Exception:
                # Skip problematic symbols
                pass
            finally:
                pbar.update(1)
            if not symbol_list and len(series_map) >= target_symbols:
                # Only early-break when we're using dynamic ranking, not a fixed list
                break

    if not series_map:
        raise RuntimeError("No symbols collected; aborting.")

    # Build DataFrame with rows=symbols, columns=dates
    df = pd.DataFrame.from_dict(series_map, orient="index")
    # Restrict to date columns strictly within window and sorted
    keep_cols = [c for c in df.columns if c >= config.START_DATE.date().isoformat() and c <= config.END_DATE.date().isoformat()]
    df = df.reindex(columns=sorted(keep_cols))

    return df


def collect_and_save_prices(symbols: Optional[List[str]] = None) -> str:
    """Collect prices and save CSV; returns path."""
    df = collect_prices_dataframe(config.TARGET_SYMBOLS, symbols=symbols)
    df.index.name = "symbol"
    df.to_csv(config.PRICES_CSV, float_format="%.10g")
    return config.PRICES_CSV


if __name__ == "__main__":
    path = collect_and_save_prices()
    print(f"Saved prices CSV to: {path}")