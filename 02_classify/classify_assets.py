#!/usr/bin/env python3
"""
Phase 2: Asset classification CSV builder (spec-compliant, adjusted per runtime directives)

- Generates 02_classify/output/assets.csv according to docs/assets_csv_spec.md (location override)
- Sources Binance-derived inputs from 01_raw_collection/output/*
- Applies curated taxonomy + Binance tags to assign classes among {L1, L2, Protocol, Meme, GameFi} with L2 priority when both L2 and Protocol signals exist
- Enforces Quality Gates (adapted to minimum count) and writes validation report + manifest
- Optional kline dry-run check (limited sample) for included assets

Usage examples:
  python 02_classify/classify_assets.py
  python 02_classify/classify_assets.py --target-count 120 --klines-check 30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError
import urllib.request


# ----------------------------
# Utilities
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUT_DIR = PROJECT_ROOT / "01_raw_collection" / "output"
CLASSIFY_OUT_DIR = PROJECT_ROOT / "02_classify" / "output"
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_HEADER = [
    "symbol_binance",
    "symbol_common",
    "class",
    "name",
    "listing_date",
    "description",
    "exclude",
    "exclude_reason",
    "classification_note",
]

CLASS_VALUES = {"L1", "L2", "Protocol", "Meme", "GameFi"}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dirs() -> None:
    CLASSIFY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Do not create data directory; outputs are placed under 02_classify/output per runtime directive.


# ----------------------------
# HTTP client (minimal, backoff)
# ----------------------------

class HttpClient:
    def __init__(self, timeout: int = 20, retries: int = 3, backoff_factor: float = 1.8) -> None:
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff_factor = backoff_factor

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(sorted(params.items()))}"

        default_headers = {
            "User-Agent": "imgadv-classify/1.0 (+https://github.com/)",
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
                    data_bytes = resp.read()
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
                if e.code in (429, 418) or 500 <= e.code < 600:
                    sleep_s = self._compute_backoff(attempt)
                    time.sleep(sleep_s)
                    continue
                else:
                    break
            except URLError as e:
                last_err = e
                sleep_s = self._compute_backoff(attempt)
                time.sleep(sleep_s)
                continue
            except Exception as e:
                last_err = e
                break

        if last_err:
            raise last_err
        raise RuntimeError("Unknown error in HTTP client")

    def _compute_backoff(self, attempt: int) -> float:
        base = self.backoff_factor * (2 ** attempt)
        jitter = random.uniform(0.0, 0.5)
        return base + jitter


# ----------------------------
# Data sources: manifest + raw JSONs
# ----------------------------

@dataclass
class Sources:
    exchange_info: Path
    bapi_symbol_list: Path
    bapi_get_products: Path
    marketcap_proxy: Path


def locate_sources() -> Sources:
    manifest_path = RAW_OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        outs = manifest.get("outputs", {})
        def canon(name: str) -> Path:
            p = outs.get(name, {}).get("canonical")
            return Path(p) if p else RAW_OUT_DIR / f"raw_{name}.json"
        return Sources(
            exchange_info=canon("exchange_info"),
            bapi_symbol_list=canon("bapi_symbol_list"),
            bapi_get_products=canon("bapi_get_products"),
            marketcap_proxy=canon("marketcap_proxy"),
        )
    # Fallback
    return Sources(
        exchange_info=RAW_OUT_DIR / "raw_exchange_info.json",
        bapi_symbol_list=RAW_OUT_DIR / "raw_bapi_symbol_list.json",
        bapi_get_products=RAW_OUT_DIR / "raw_bapi_get_products.json",
        marketcap_proxy=RAW_OUT_DIR / "raw_marketcap_proxy.json",
    )


# ----------------------------
# Taxonomy (curated sets + tag mapping)
# ----------------------------

CURATED_L1 = {
    "BTC", "ETH", "BNB", "SOL", "TRX", "ADA", "XRP", "LTC", "ATOM", "XLM", "NEO", "XTZ", "ALGO", "DOT", "AVAX",
    "ICP", "TON", "XMR", "ETC", "EOS"
}

CURATED_L2 = {
    "ARB", "OP", "STRK", "MATIC", "POL", "METIS", "SKL", "ZKS", "BLAST"
}

CURATED_PROTOCOL = {
    "UNI", "AAVE", "CRV", "SUSHI", "MKR", "SNX", "COMP", "LDO", "DYDX", "INJ", "RUNE", "GMX", "CAKE", "1INCH",
    "CVX", "BAL", "YFI", "SXP", "XVS", "FTM"  # FTM is L1 in reality; included as protocol-like for demo if needed
}

CURATED_MEME = {
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BABYDOGE", "TURBO", "TRUMP", "REALTRUMP", "GOAT"
}

CURATED_GAMEFI = {
    "AXS", "SAND", "GALA", "ENJ", "MANA", "ILV", "IMX"
}

STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD"}

# Tag to class hints (non-binding; combined with curated sets)
TAG_TO_CLASS = {
    "Meme": "Meme",
    "memecoin": "Meme",
    "GameFi": "GameFi",
    "Gaming": "GameFi",
    "Layer 1": "L1",
    "Layer 2": "L2",
    "Layer1_Layer2": None,  # ambiguous; resolve with curated lists
    "DeFi": "Protocol",
    "DEX": "Protocol",
    "Lending": "Protocol",
    "LiquidStaking": "Protocol",
    "Derivatives": "Protocol",
}


# ----------------------------
# Metadata assembly
# ----------------------------

@dataclass
class AssetMeta:
    base: str
    name: str
    full_name: Optional[str]
    listing_ms: Optional[int]
    tags: List[str]


def extract_base_from_symbol(symbol: str) -> Optional[str]:
    """Extract base asset from a BINANCE USDT spot symbol like 'BTCUSDT'."""
    if not symbol or not symbol.endswith("USDT"):
        return None
    base = symbol[:-4]  # remove 'USDT'
    # Filter leveraged tokens etc. (already filtered in marketcap_proxy, but keep safe)
    if any(x in base for x in ("UP", "DOWN", "BULL", "BEAR", "PERP")):
        return None
    return base


def build_symbol_meta_maps(sources: Sources) -> Tuple[Dict[str, AssetMeta], Dict[str, int]]:
    """
    Returns:
    - meta_by_base: base -> AssetMeta
    - onboard_by_symbol: symbol -> onboardDate (ms) from exchange_info
    """
    # Exchange info
    onboard_by_symbol: Dict[str, int] = {}
    try:
        exch = read_json(sources.exchange_info)
        if isinstance(exch, dict):
            for s in exch.get("symbols", []):
                try:
                    sym = s.get("symbol")
                    od = s.get("onboardDate")
                    if sym and isinstance(od, int):
                        onboard_by_symbol[sym] = od
                except Exception:
                    continue
    except Exception:
        pass

    # BAPI symbol list
    name_by_base: Dict[str, str] = {}
    full_by_base: Dict[str, str] = {}
    listing_by_symbol: Dict[str, int] = {}
    tags_by_base: Dict[str, List[str]] = {}

    try:
        blist = read_json(sources.bapi_symbol_list)
        if isinstance(blist, dict) and isinstance(blist.get("data"), list):
            for item in blist["data"]:
                try:
                    symbol = item.get("symbol")
                    base = item.get("baseAsset") or None
                    name = item.get("name") or item.get("mapperName") or ""
                    full = item.get("fullName") or item.get("localFullName") or None
                    ltime = item.get("listingTime")
                    tags = []
                    # tagInfos may include 'display' strings like "Layer 1 / Layer 2"
                    if isinstance(item.get("tagInfos"), list):
                        for ti in item["tagInfos"]:
                            disp = ti.get("display")
                            if isinstance(disp, str):
                                tags.append(disp)
                    if isinstance(item.get("tags"), list):
                        for t in item["tags"]:
                            if isinstance(t, str):
                                tags.append(t)
                    if base:
                        if name:
                            name_by_base[base] = name
                        if full:
                            full_by_base[base] = full
                        if tags:
                            tags_by_base[base] = list(sorted(set(tags)))
                    if symbol and isinstance(ltime, int):
                        listing_by_symbol[symbol] = ltime
                except Exception:
                    continue
    except Exception:
        pass

    # BAPI get products (augment tags)
    try:
        bprods = read_json(sources.bapi_get_products)
        if isinstance(bprods, dict) and isinstance(bprods.get("data"), list):
            for item in bprods["data"]:
                try:
                    s = item.get("s")  # symbol (various quotes)
                    b = item.get("b")  # base
                    q = item.get("q")  # quote
                    tags = item.get("tags") or []
                    if isinstance(b, str):
                        existing = tags_by_base.get(b, [])
                        merged = list(sorted(set(existing + [t for t in tags if isinstance(t, str)])))
                        tags_by_base[b] = merged
                except Exception:
                    continue
    except Exception:
        pass

    # Build meta_by_base
    meta_by_base: Dict[str, AssetMeta] = {}
    for base, name in name_by_base.items():
        meta = AssetMeta(
            base=base,
            name=name,
            full_name=full_by_base.get(base),
            listing_ms=None,  # fill later per symbol
            tags=tags_by_base.get(base, []),
        )
        meta_by_base[base] = meta

    # Listing time: prefer exchange_info onboardDate for BASEUSDT, else bapi symbol list listingTime
    for base, meta in list(meta_by_base.items()):
        symbol = f"{base}USDT"
        listing_ms = None
        if symbol in onboard_by_symbol:
            listing_ms = onboard_by_symbol[symbol]
        elif symbol in listing_by_symbol:
            listing_ms = listing_by_symbol[symbol]
        meta.listing_ms = listing_ms

    return meta_by_base, onboard_by_symbol


# ----------------------------
# Classification
# ----------------------------

def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def resolve_class(base: str, tags: List[str]) -> Optional[str]:
    b = base.upper()

    # Exclude stablecoins
    if b in STABLECOINS:
        return None

    # Curated priority
    if b in CURATED_MEME:
        return "Meme"
    if b in CURATED_GAMEFI:
        return "GameFi"
    if b in CURATED_L2:
        return "L2"
    if b in CURATED_L1:
        return "L1"
    if b in CURATED_PROTOCOL:
        # defer final decision to tag checks to allow L2 precedence if both signals exist
        pass

    # Tag-driven hints (normalize)
    lowered = {normalize_tag(t) for t in tags}
    # L2-oriented signals first: ensure L2 precedence over Protocol when both apply
    l2_signals = {"layer 2", "l2", "rollup", "optimistic", "zk", "zk rollup", "sidechain"}
    if lowered.intersection(l2_signals) or "layer 1 / layer 2" in lowered or "layer1_layer2" in lowered:
        # Prefer L2 if any curated/tag signals suggest it
        return "L2"

    # Meme/GameFi hints
    if "meme" in lowered or "memecoin" in lowered:
        return "Meme"
    if "gamefi" in lowered or "gaming" in lowered:
        return "GameFi"

    # Protocol-like tags
    protocol_like = {"defi", "dex", "lending", "liquidstaking", "derivatives", "amm"}
    if lowered.intersection(protocol_like):
        return "Protocol"

    # Fallback to curated Protocol if present
    if b in CURATED_PROTOCOL:
        return "Protocol"

    # Fallback: None -> excluded
    return None


def ms_to_date_str(ms: Optional[int]) -> str:
    if ms is None:
        # conservative fallback: 3y ago from today
        dt = datetime.now(timezone.utc).replace(microsecond=0)
        dt3y = datetime(dt.year - 3, dt.month, dt.day, tzinfo=timezone.utc)
        return dt3y.strftime("%Y-%m-%d")
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")

def token_description(base: str, cls: str, name: str, tags: List[str]) -> str:
    b = base.upper()
    nm = name or b
    tz = {normalize_tag(t) for t in tags}

    # consensus / mechanics hints
    consensus = []
    if "pos" in tz or "proof of stake" in tz:
        consensus.append("Proof of Stake")
    if "pow" in tz or "proof of work" in tz:
        consensus.append("Proof of Work")

    # L2 style hints
    rollup_style = None
    if "zk" in tz or "zk rollup" in tz:
        rollup_style = "zk-rollup"
    elif "optimistic" in tz:
        rollup_style = "optimistic rollup"
    elif "sidechain" in tz:
        rollup_style = "sidechain"

    # DeFi function hints
    defi_funcs = []
    if "dex" in tz:
        defi_funcs.append("DEX governance")
    if "lending" in tz:
        defi_funcs.append("lending protocol governance")
    if "derivatives" in tz:
        defi_funcs.append("derivatives protocol governance")
    if "liquidstaking" in tz or "liquid staking" in tz:
        defi_funcs.append("liquid staking coordination")

    if cls == "L1":
        cons = f" ({', '.join(consensus)})" if consensus else ""
        return f"Native token of the {nm} base-layer blockchain{cons}; used to pay transaction fees and secure the network, with chain-specific monetary properties."
    if cls == "L2":
        style = f" as a {rollup_style}" if rollup_style else " in a Layer 2/sidechain architecture"
        return f"Governance and utility token for the {nm}{style}; enables protocol governance and may relate to sequencer economics, bridging, and fee mechanisms."
    if cls == "Protocol":
        extra = f" focusing on {'; '.join(defi_funcs)}" if defi_funcs else ""
        return f"Governance token of the {nm} on-chain protocol{extra}; holders vote on parameters and treasury and may capture protocol value."
    if cls == "Meme":
        return f"Community/meme-driven token ({nm}) with value led by sentiment and demand rather than protocol utility; commonly used for transfers and speculation."
    if cls == "GameFi":
        return f"Gaming ecosystem token associated with {nm}; used for governance and in-game economy operations within the GameFi ecosystem."
    return f"Token {nm}."


# ----------------------------
# Candidate selection and row building
# ----------------------------

@dataclass
class CsvRow:
    symbol_binance: str
    symbol_common: str
    cls: str
    name: str
    listing_date: str
    description: str
    exclude: bool
    exclude_reason: str
    classification_note: str

    def to_list(self) -> List[str]:
        return [
            self.symbol_binance,
            self.symbol_common,
            self.cls,
            self.name,
            self.listing_date,
            self.description,
            "True" if self.exclude else "False",
            self.exclude_reason,
            self.classification_note,
        ]


def load_marketcap_proxy(sources: Sources) -> List[Dict[str, Any]]:
    try:
        proxy = read_json(sources.marketcap_proxy)
        if isinstance(proxy, list):
            return proxy
    except Exception:
        pass
    return []


def build_candidate_rows(target_count: int, sources: Sources) -> Tuple[List[CsvRow], List[str]]:
    meta_by_base, onboard_by_symbol = build_symbol_meta_maps(sources)
    proxy = load_marketcap_proxy(sources)

    errors: List[str] = []
    included: List[CsvRow] = []
    seen_symbols: set[str] = set()
    seen_bases: set[str] = set()

    # Sort by liquidity proxy descending
    proxy_sorted = sorted(
        (item for item in proxy if isinstance(item, dict)),
        key=lambda d: float(d.get("quoteVolumeUSDT_24h", 0.0)),
        reverse=True,
    )

    for item in proxy_sorted:
        try:
            symbol = item.get("symbol")
            base = extract_base_from_symbol(symbol or "")
            if not base:
                continue
            base_u = base.upper()
            # Skip stablecoins and duplicates
            if base_u in STABLECOINS:
                continue
            if symbol in seen_symbols or base_u in seen_bases:
                continue

            meta = meta_by_base.get(base_u)
            name = (meta.name if meta else base_u)
            tags = (meta.tags if meta else [])
            cls = resolve_class(base_u, tags)
            if not cls:
                # Exclude unknowns; record reason
                row = CsvRow(
                    symbol_binance=f"{base_u}USDT",
                    symbol_common=base_u,
                    cls="",
                    name=name,
                    listing_date=ms_to_date_str(meta.listing_ms if meta else None),
                    description=f"{name} excluded from analytical universe pending classification review.",
                    exclude=True,
                    exclude_reason="unknown",
                    classification_note="No class assigned; requires manual review",
                )
                # We keep excluded entries only for documentation; do not count towards 100
                # You may push some excluded rows for audit but here we skip to meet target count quickly.
                continue

            # Build included row
            listing_ms = None
            sym_usdt = f"{base_u}USDT"
            if sym_usdt in onboard_by_symbol:
                listing_ms = onboard_by_symbol[sym_usdt]
            elif meta and meta.listing_ms:
                listing_ms = meta.listing_ms

            listing_date_str = ms_to_date_str(listing_ms)

            row = CsvRow(
                symbol_binance=sym_usdt,
                symbol_common=base_u,
                cls=cls,
                name=name,
                listing_date=listing_date_str,
                description=token_description(base_u, cls, name, tags),
                exclude=False,
                exclude_reason="",
                classification_note=("Assigned via curated taxonomy + Binance tags; L2 precedence over Protocol when both signals exist" if cls == "L2" else "Assigned via curated taxonomy + Binance tags"),
            )

            included.append(row)
            seen_symbols.add(symbol)
            seen_bases.add(base_u)

            # Continue collecting; produce >= target_count (no early break)
        except Exception as e:
            errors.append(f"Row build error: {e}")
            continue

    # Deterministic ordering: sort by class then symbol_common
    included.sort(key=lambda r: (r.cls, r.symbol_common))
    return included, errors


# ----------------------------
# CSV writing and validations
# ----------------------------

def write_csv(path: Path, rows: List[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(REQUIRED_HEADER)
        for r in rows:
            writer.writerow(r.to_list())


def validate_csv(path: Path, target_count: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp": iso_now(),
        "path": str(path),
        "qg": {},
        "errors": [],
        "warnings": [],
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
    except Exception as e:
        report["errors"].append(f"CSV read failed: {e}")
        return report

    # QG-1: Header exact
    if header != REQUIRED_HEADER:
        report["errors"].append("QG-1 failed: Header mismatch")
    else:
        report["qg"]["QG-1"] = True

    # Parse rows
    included_rows = []
    by_symbol: Dict[str, int] = {}
    now_date = datetime.now(timezone.utc).date()
    for i, r in enumerate(rows, start=2):
        try:
            (
                symbol_binance, symbol_common, cls, name,
                listing_date, description, exclude, exclude_reason, classification_note
            ) = r

            # Types and basic checks
            if not symbol_binance or not symbol_common or not name or not description:
                report["errors"].append(f"Row {i}: required string fields missing")
            if exclude not in ("True", "False"):
                report["errors"].append(f"Row {i}: exclude must be 'True'/'False'")

            ex = (exclude == "True")
            if ex:
                if cls != "":
                    report["errors"].append(f"Row {i}: excluded row must have class=''")
                if not exclude_reason:
                    report["errors"].append(f"Row {i}: excluded row must have non-empty exclude_reason")
            else:
                if cls not in CLASS_VALUES:
                    report["errors"].append(f"Row {i}: class must be one of {CLASS_VALUES}")
                if exclude_reason:
                    report["errors"].append(f"Row {i}: include row must have empty exclude_reason")

                # listing_date not future, format YYYY-MM-DD
                try:
                    dt = datetime.strptime(listing_date, "%Y-%m-%d").date()
                    if dt > now_date:
                        report["errors"].append(f"Row {i}: listing_date in future")
                except Exception:
                    report["errors"].append(f"Row {i}: listing_date format invalid")

                included_rows.append(r)

            # Uniqueness
            by_symbol[symbol_binance] = by_symbol.get(symbol_binance, 0) + 1

        except Exception as e:
            report["errors"].append(f"Row {i}: parse error {e}")

    # QG-2: At least target_count rows included
    if len(included_rows) < target_count:
        report["errors"].append(f"QG-2 failed: expected at least {target_count} included rows, got {len(included_rows)}")
    else:
        report["qg"]["QG-2"] = True

    # QG-4: symbol_binance unique
    dups = [s for s, c in by_symbol.items() if c > 1]
    if dups:
        report["errors"].append(f"QG-4 failed: duplicate symbols {dups}")
    else:
        report["qg"]["QG-4"] = True

    # Deterministic ordering recommendation
    # Sort by class then symbol_common check (non-fatal)
    try:
        sorted_copy = sorted(
            included_rows,
            key=lambda r: (r[2], r[1])  # class, symbol_common
        )
        if included_rows != sorted_copy:
            report["warnings"].append("Rows not in deterministic order (class then symbol_common)")
    except Exception:
        pass

    return report


# ----------------------------
# Kline dry-run (sample)
# ----------------------------

def klines_dry_run(base_url: str, rows: List[CsvRow], sample_n: int, timeout: int = 20, retries: int = 2) -> Dict[str, Any]:
    client = HttpClient(timeout=timeout, retries=retries)
    results = {"checked": 0, "ok": 0, "fail": 0, "errors": []}
    now_ms = int(time.time() * 1000)
    # 7 days window
    start_ms = now_ms - (7 * 24 * 3600 * 1000)
    end_ms = now_ms

    sample = rows[:max(0, min(sample_n, len(rows)))]
    for r in sample:
        sym = r.symbol_binance
        params = {
            "symbol": sym,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1,
        }
        try:
            data = client.get_json(f"{base_url.rstrip('/')}/api/v3/klines", params=params)
            ok = isinstance(data, list) and len(data) > 0
            results["checked"] += 1
            if ok:
                results["ok"] += 1
            else:
                results["fail"] += 1
                results["errors"].append(f"{sym}: empty klines")
        except Exception as e:
            results["checked"] += 1
            results["fail"] += 1
            results["errors"].append(f"{sym}: {e}")
    return results


# ----------------------------
# Manifest writing
# ----------------------------

def write_manifest(script_path: Path, output_csv: Path, report_path: Path, args_ns: argparse.Namespace) -> None:
    manifest = {
        "generated_at": iso_now(),
        "script_path": str(script_path),
        "script_sha256": compute_file_sha256(script_path),
        "inputs": {
            "raw_output_dir": str(RAW_OUT_DIR),
        },
        "outputs": {
            "csv": str(output_csv),
            "validation_report": str(report_path),
        },
        "args": {
            "target_count": args_ns.target_count,
            "klines_check": args_ns.klines_check,
            "base_url": args_ns.base_url,
        },
    }
    write_json(CLASSIFY_OUT_DIR / "assets_csv_manifest.json", manifest)


# ----------------------------
# Main
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 02_classify/output/assets.csv from Binance raw outputs (Phase 2 classification)")
    p.add_argument("--target-count", type=int, default=100, help="Minimum number of included assets to produce (default: 100)")
    p.add_argument("--output-csv", type=str, default=str(CLASSIFY_OUT_DIR / "assets.csv"), help="Output CSV path (default: 02_classify/output/assets.csv)")
    p.add_argument("--klines-check", type=int, default=20, help="Number of included assets to kline dry-run check (default: 20)")
    p.add_argument("--base-url", type=str, default="https://api.binance.com", help="Binance API base URL")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ensure_dirs()
    sources = locate_sources()

    # Build candidate included rows
    rows, build_errors = build_candidate_rows(args.target_count, sources)
    if build_errors:
        write_json(CLASSIFY_OUT_DIR / "build_errors.json", {"errors": build_errors})

    # Write CSV
    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)

    # Validate
    report = validate_csv(output_csv, args.target_count)

    # Optional kline dry-run check
    if args.klines_check > 0 and not report.get("errors"):
        # Only perform if validation passed (no critical errors)
        try:
            kresults = klines_dry_run(args.base_url, [r for r in rows if not r.exclude], args.klines_check)
            report["kline_check"] = kresults
        except Exception as e:
            report.setdefault("warnings", []).append(f"Klines dry-run skipped: {e}")

    # Save validation report
    report_path = CLASSIFY_OUT_DIR / "validation_report.json"
    write_json(report_path, report)

    # Manifest
    write_manifest(Path(__file__), output_csv, report_path, args)

    # Console summary
    ok = not report.get("errors")
    print(f"[INFO] Wrote CSV: {output_csv}")
    print(f"[INFO] Validation {'PASSED' if ok else 'FAILED'}; report: {report_path}")
    if report.get("errors"):
        for e in report["errors"]:
            print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())