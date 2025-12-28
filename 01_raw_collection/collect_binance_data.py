#!/usr/bin/env python3
"""
Phase 1: Raw Binance API data collection (no classification).

Features:
- Robust HTTP client with retries, backoff, and 429/5xx handling
- File-based JSON caching (TTL-controlled) without external configs
- Canonical outputs + timestamped snapshots for traceability
- Script-to-output linkage via manifest (includes script SHA256)
- No YAML, flexible defaults with CLI overrides

Outputs (relative to this script):
- ./output/raw_exchange_info.json (+ timestamped snapshot)
- ./output/raw_ticker_24hr.json (+ snapshot)
- ./output/raw_ticker_price.json (+ snapshot)
- ./output/raw_book_ticker.json (+ snapshot)
- ./output/manifest.json

Example:
  python 01_raw_collection/collect_binance_data.py
  python 01_raw_collection/collect_binance_data.py --ttl-hours 6 --force
  python 01_raw_collection/collect_binance_data.py --base-url https://api.binance.com --timeout 25 --retries 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError
import urllib.request
import random
import re


JSONType = Union[Dict[str, Any], list]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HttpClient:
    def __init__(self, timeout: int = 20, retries: int = 3, backoff_factor: float = 1.8) -> None:
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff_factor = backoff_factor

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> JSONType:
        full_url = url
        if params:
            # Sort params for reproducible URLs
            full_url = f"{url}?{urlencode(sorted(params.items()))}"

        default_headers = {
            "User-Agent": "imgadv-collector/1.0 (+https://github.com/)",
            "Accept": "application/json",
            "Connection": "close",
        }
        req_headers = dict(default_headers)
        if headers:
            req_headers.update(headers)

        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(full_url, headers=req_headers, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = getattr(resp, "status", resp.getcode())
                    if status and status >= 400:
                        raise HTTPError(full_url, status, f"HTTP {status}", resp.headers, None)
                    raw = resp.read()
                    # Handle possible compression
                    try:
                        enc_header = resp.headers.get("Content-Encoding", "")
                        enc = enc_header.lower() if enc_header else ""
                    except Exception:
                        enc = ""
                    data_bytes = raw
                    if "gzip" in enc:
                        try:
                            import gzip
                            data_bytes = gzip.decompress(raw)
                        except Exception:
                            # fall back to raw bytes if decompression fails
                            data_bytes = raw
                    elif "deflate" in enc:
                        try:
                            import zlib
                            # try raw deflate first, then zlib wrapper
                            try:
                                data_bytes = zlib.decompress(raw, -zlib.MAX_WBITS)
                            except Exception:
                                data_bytes = zlib.decompress(raw)
                        except Exception:
                            data_bytes = raw
                    # Decode to text
                    try:
                        data_str = data_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        data_str = data_bytes.decode("utf-8", errors="replace")
                try:
                    return json.loads(data_str)
                except json.JSONDecodeError as e:
                    preview = (data_str[:200].replace("\n", " ").replace("\r", " "))
                    raise RuntimeError(f"JSON decode error at {full_url}: {e}; preview='{preview}'")
            except HTTPError as e:
                last_err = e
                # Handle 429 (rate limit) and 418/5xx with backoff
                if e.code in (429, 418) or 500 <= e.code < 600:
                    sleep_s = self._compute_backoff(attempt, e.headers)
                    self._warn(f"HTTPError {e.code} on {full_url}; retrying in {sleep_s:.2f}s (attempt {attempt+1}/{self.retries})")
                    time.sleep(sleep_s)
                    continue
                else:
                    break
            except URLError as e:
                last_err = e
                sleep_s = self._compute_backoff(attempt, None)
                self._warn(f"URLError '{e.reason}' on {full_url}; retrying in {sleep_s:.2f}s (attempt {attempt+1}/{self.retries})")
                time.sleep(sleep_s)
                continue
            except Exception as e:
                last_err = e
                break

        if last_err:
            raise last_err
        raise RuntimeError("Unknown error in HTTP client")

    def _compute_backoff(self, attempt: int, headers: Optional[Dict[str, Any]]) -> float:
        # Respect Retry-After if provided
        retry_after = None
        if headers:
            ra = headers.get("Retry-After") if hasattr(headers, "get") else None
            if ra:
                try:
                    retry_after = float(ra)
                except Exception:
                    retry_after = None
        if retry_after is not None:
            return max(1.0, retry_after)
        # Exponential backoff with jitter
        base = self.backoff_factor * (2 ** attempt)
        jitter = random.uniform(0.0, 0.5)
        return base + jitter

    @staticmethod
    def _warn(msg: str) -> None:
        print(f"[WARN] {msg}", file=sys.stderr)


class BinanceAPICache:
    """
    File-based JSON cache keyed by (endpoint, params).
    - Stores wrapper with metadata + data
    - TTL-based cache freshness control
    - Falls back to stale cache if a refresh fails
    """
    def __init__(self, cache_dir: Path, client: HttpClient) -> None:
        self.cache_dir = cache_dir
        self.client = client
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_fetch(
        self,
        tag: str,
        url: str,
        params: Optional[Dict[str, Any]],
        ttl_hours: float,
        force: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> JSONType:
        key = self._make_cache_key(url, params)
        cache_file = self.cache_dir / f"{tag}__{key}.json"

        # If usable cache exists and not forced
        if cache_file.exists() and not force:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                fetched_epoch = float(payload.get("fetched_at_epoch", 0.0))
                age_hours = (time.time() - fetched_epoch) / 3600.0
                if age_hours <= ttl_hours:
                    return payload["data"]
            except Exception:
                # Corrupted cache, proceed to fetch
                pass

        # Try to fetch fresh
        try:
            data = self.client.get_json(url, params=params, headers=headers)
            wrapper = {
                "fetched_at": iso_now(),
                "fetched_at_epoch": time.time(),
                "endpoint": url,
                "params": params,
                "data": data,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(wrapper, f, ensure_ascii=False, indent=2)
            return data
        except Exception as e:
            # Fallback to stale cache if present
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    print(f"[WARN] Fetch failed ({e}); using stale cache for tag='{tag}'", file=sys.stderr)
                    return payload["data"]
                except Exception:
                    pass
            raise

    @staticmethod
    def _make_cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
        s = f"{url}|{json.dumps(params or {}, sort_keys=True, ensure_ascii=False)}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def collect_raw(
    base_url: str,
    ttl_hours: float,
    force: bool,
    timeout: int,
    retries: int,
    script_path: Path,
) -> int:
    script_dir = script_path.resolve().parent
    cache_dir = script_dir / "cache"
    out_dir = script_dir / "output"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = HttpClient(timeout=timeout, retries=retries)
    cache = BinanceAPICache(cache_dir, client)

    # Define endpoints
    endpoints = [
        ("exchange_info", "/api/v3/exchangeInfo", None),
        ("ticker_24hr", "/api/v3/ticker/24hr", None),
        ("ticker_price", "/api/v3/ticker/price", None),
        ("book_ticker", "/api/v3/ticker/bookTicker", None),
    ]

    summary: Dict[str, Dict[str, Any]] = {}
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for tag, path, params in endpoints:
        url = f"{base_url.rstrip('/')}{path}"
        print(f"[INFO] Fetching '{tag}' from {url} (force={force}, ttl_hours={ttl_hours})")
        data = cache.get_or_fetch(tag=tag, url=url, params=params, ttl_hours=ttl_hours, force=force)

        # Determine count
        if tag == "exchange_info" and isinstance(data, dict):
            count = len(data.get("symbols", []))
        elif isinstance(data, list):
            count = len(data)
        else:
            # Fallback for unexpected shapes
            try:
                count = len(data)  # type: ignore
            except Exception:
                count = -1

        # Save canonical and snapshot
        canonical = out_dir / f"raw_{tag}.json"
        snapshot = out_dir / f"raw_{tag}_{ts_str}.json"
        write_json(canonical, data)
        write_json(snapshot, data)

        print(f"[INFO] Saved {tag}: canonical={canonical}, snapshot={snapshot}, count={count}")
        summary[tag] = {
            "canonical": str(canonical),
            "snapshot": str(snapshot),
            "count": count,
        }

    # Attempt best-effort Binance web BAPI endpoints for market cap/supply (if available)
    bapi_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.binance.com/",
        "Origin": "https://www.binance.com",
        "Connection": "close",
    }
    bapi_endpoints = [
        ("bapi_symbol_list", "https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list", None),
        ("bapi_get_products", "https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products", {"includeEtf": "true"}),
        ("bapi_coin_list", "https://www.binance.com/bapi/composite/v1/public/marketing/coin/list", None),
    ]
    for btag, burl, bparams in bapi_endpoints:
        try:
            print(f"[INFO] Fetching best-effort '{btag}' from {burl}")
            bdata = cache.get_or_fetch(tag=btag, url=burl, params=bparams, ttl_hours=ttl_hours, force=force, headers=bapi_headers)
            bcanon = out_dir / f"raw_{btag}.json"
            bsnap = out_dir / f"raw_{btag}_{ts_str}.json"
            write_json(bcanon, bdata)
            write_json(bsnap, bdata)
            # Count heuristic
            if isinstance(bdata, list):
                bcount = len(bdata)
            elif isinstance(bdata, dict):
                # try common list fields
                for key in ("data", "symbolList", "coins", "products"):
                    if key in bdata and isinstance(bdata[key], list):
                        bcount = len(bdata[key])
                        break
                else:
                    try:
                        bcount = len(bdata)
                    except Exception:
                        bcount = -1
            else:
                try:
                    bcount = len(bdata)  # type: ignore
                except Exception:
                    bcount = -1
            print(f"[INFO] Saved {btag}: canonical={bcanon}, snapshot={bsnap}, count={bcount}")
            summary[btag] = {
                "canonical": str(bcanon),
                "snapshot": str(bsnap),
                "count": bcount,
            }
        except Exception as e:
            print(f"[WARN] BAPI fetch failed for {btag}: {e}", file=sys.stderr)

    # Compute liquidity-based market cap proxy from 24h USDT quote volumes
    try:
        t24_path = out_dir / "raw_ticker_24hr.json"
        with open(t24_path, "r", encoding="utf-8") as f:
            t24 = json.load(f)
        proxy = []
        if isinstance(t24, list):
            for item in t24:
                try:
                    sym = item.get("symbol", "")
                    # USDT spot pairs; exclude leveraged tokens, perpetuals, etc.
                    if not (isinstance(sym, str) and sym.endswith("USDT")):
                        continue
                    if any(x in sym for x in ("UP", "DOWN", "BULL", "BEAR", "PERP")):
                        continue
                    qv = float(item.get("quoteVolume", "0") or 0.0)
                    lp = float(item.get("lastPrice", "0") or 0.0)
                    proxy.append({"symbol": sym, "quoteVolumeUSDT_24h": qv, "lastPrice": lp})
                except Exception:
                    continue
            proxy.sort(key=lambda d: d.get("quoteVolumeUSDT_24h", 0.0), reverse=True)
            # Keep a broad set for downstream selection
            proxy = proxy[:2000]
        proxy_path = out_dir / "raw_marketcap_proxy.json"
        proxy_snap = out_dir / f"raw_marketcap_proxy_{ts_str}.json"
        write_json(proxy_path, proxy)
        write_json(proxy_snap, proxy)
        print(f"[INFO] Saved marketcap proxy: canonical={proxy_path}, snapshot={proxy_snap}, count={len(proxy)}")
        summary["marketcap_proxy"] = {
            "canonical": str(proxy_path),
            "snapshot": str(proxy_snap),
            "count": len(proxy),
        }
    except Exception as e:
        print(f"[WARN] Failed to compute marketcap proxy: {e}", file=sys.stderr)

    # Manifest linking code and outputs
    manifest = {
        "generated_at": iso_now(),
        "base_url": base_url,
        "args": {
            "ttl_hours": ttl_hours,
            "force": force,
            "timeout": timeout,
            "retries": retries,
        },
        "script_path": str(script_path),
        "script_sha256": compute_file_sha256(script_path),
        "outputs": summary,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"[INFO] Wrote manifest: {manifest_path}")

    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Raw Binance API data collection with caching")
    p.add_argument(
        "--base-url",
        default="https://api.binance.com",
        help="Binance API base URL (default: https://api.binance.com)",
    )
    p.add_argument(
        "--ttl-hours",
        type=float,
        default=24.0,
        help="Cache TTL in hours (default: 24.0). Use --force to bypass.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force refresh from API, bypassing cache",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds (default: 20)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP retry attempts for transient errors (default: 3)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return collect_raw(
            base_url=args.base_url,
            ttl_hours=args.ttl_hours,
            force=args.force,
            timeout=args.timeout,
            retries=args.retries,
            script_path=Path(__file__),
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())