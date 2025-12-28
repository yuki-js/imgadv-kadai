import argparse
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import numpy as np
from PIL import Image

# Make `kaodake/libaoki` importable without installing as a package
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "kaodake"))

from libaoki.faceprocessor import FaceProcessor
from libaoki.interfaces import LoadStrategy, OutputStrategy, PostprocessStrategy, PreprocessStrategy


@dataclass(frozen=True)
class NftRow:
    token_id: str
    image_url: str


class IdentityPreprocessor(PreprocessStrategy):
    def process(self, image: Image.Image) -> Image.Image:
        return image


class RGBPostprocessor(PostprocessStrategy):
    def process(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


class BytesLoader(LoadStrategy[bytes]):
    def process(self, data: bytes) -> Image.Image:
        # ensure PIL fully loads (avoid lazy errors after BytesIO goes out of scope)
        im = Image.open(io.BytesIO(data))
        im.load()
        return im


class NumpySquareOutputter(OutputStrategy[np.ndarray]):
    def __init__(self, size: int):
        self._size = int(size)

    def process(self, image: Image.Image) -> np.ndarray:
        return np.array(image.resize((self._size, self._size), Image.BILINEAR))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_rows(csv_path: Path) -> List[NftRow]:
    """Read `itadakimasu-man.csv`.

    Note: the CSV has 2 header lines; the first one is Japanese labels.
    """

    rows: List[NftRow] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        # Skip the first (Japanese) header line
        _ = f.readline()

        reader = csv.DictReader(f)
        for r in reader:
            token_id = (r.get("tokenId") or "").strip()
            image_url = (r.get("image_url") or "").strip()
            if not token_id or not image_url:
                continue
            rows.append(NftRow(token_id=token_id, image_url=image_url))
    return rows


def _normalize_ipfs_url(url: str, gateway: str) -> str:
    u = url.strip()
    if u.startswith("ipfs://"):
        # ipfs://<cid>/<path>
        return gateway.rstrip("/") + "/ipfs/" + u[len("ipfs://") :].lstrip("/")

    # some rows in the CSV seem to have malformed `ipfs:/...`
    if u.startswith("ipfs:/") and not u.startswith("ipfs://"):
        return gateway.rstrip("/") + "/ipfs/" + u[len("ipfs:/") :].lstrip("/")

    return u


def _http_get(url: str, timeout_s: int, user_agent: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "image/*,*/*;q=0.8",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _ordered_gateways_for_token(gateways: List[str], token_id: str) -> List[str]:
    """Return a deterministic permutation of gateways per token.

    This reduces load concentration on the first gateway (e.g. ipfs.io).
    """

    if not gateways:
        return []

    # Deterministic shuffle: stable across processes
    rng = random.Random(hash(token_id) & 0xFFFFFFFF)
    gws = list(gateways)
    rng.shuffle(gws)
    return gws


def _download_with_fallbacks(
    image_url: str,
    gateways: List[str],
    timeout_s: int,
    user_agent: str,
    retries: int,
    sleep_s: float,
) -> Tuple[bytes, str]:
    last_err: Optional[Exception] = None
    for gw in gateways:
        resolved = _normalize_ipfs_url(image_url, gw)
        for attempt in range(retries):
            try:
                blob = _http_get(resolved, timeout_s=timeout_s, user_agent=user_agent)
                if not blob:
                    raise RuntimeError("empty response")
                return blob, resolved
            except Exception as e:
                last_err = e
                if attempt + 1 < retries:
                    time.sleep(sleep_s)
                continue
    assert last_err is not None
    raise last_err


def _make_face_processor(out_size: int) -> FaceProcessor:
     # NOTE: Using device="cpu" to avoid CUDA memory overhead and copy latency.
     # CPU mode is more efficient for this dataset processing workflow.
     from io import BytesIO  # local import

     class _BytesLoader(LoadStrategy[bytes]):
         def process(self, data: bytes) -> Image.Image:
             im = Image.open(BytesIO(data))
             im.load()
             return im

     return FaceProcessor(
         preprocessor=IdentityPreprocessor(),
         postprocessor=RGBPostprocessor(),
         loader=_BytesLoader(),
         outputter=NumpySquareOutputter(out_size),
         # match `kaodake/processor.py` defaults used in existing dataset
         fine_down_expand_ratio=1.5,
         fine_up_expand_ratio=0.7,
         fine_side_expand_ratio=0.5,
         coarse_resize_ratio=0.125,
         coarse_expand_ratio=0.25,
     )


def _save_rgb_flat_npy(out_path: Path, img_rgb: np.ndarray) -> None:
    # store flattened (H*W*3) uint8 row
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), img_rgb.reshape(-1))


def _iter_candidate_raw_paths(cache_roots: Iterable[Path]) -> Iterable[Path]:
    for root in cache_roots:
        if not root.exists():
            continue
        # common layout: <out>/<run-id>/raw/<token>.jpeg
        for p in root.rglob("raw"):
            if not p.is_dir():
                continue
            yield from p.glob("*.jpeg")
            yield from p.glob("*.jpg")
            yield from p.glob("*.png")


def _build_raw_cache_index(cache_roots: List[Path]) -> Dict[str, Path]:
    """Build token_id -> raw file path index.

    If the same token exists in multiple runs, the first discovered is used.
    """

    idx: Dict[str, Path] = {}
    for p in _iter_candidate_raw_paths(cache_roots):
        token = p.stem
        if token and token not in idx:
            idx[token] = p
    return idx


def _read_cached_raw_bytes(
    token_id: str,
    raw_path: Path,
    raw_cache_index: Dict[str, Path],
) -> Optional[bytes]:
    # Prefer current run output path (if resuming / re-running)
    try:
        if raw_path.exists():
            return raw_path.read_bytes()
    except Exception:
        pass

    cached = raw_cache_index.get(token_id)
    if cached is None:
        return None
    try:
        return cached.read_bytes()
    except Exception:
        return None


@dataclass(frozen=True)
class _WorkerArgs:
    row: NftRow
    idx: int
    total: int
    out_size: int
    max_faces: int
    timeout_s: int
    retries: int
    sleep_s: float
    user_agent: str
    gateways: List[str]
    raw_dir: str
    crops_dir: str
    meta_dir: str
    raw_cache_index: Dict[str, str]  # token -> path (strings for pickling)


def _process_one_nft(a: _WorkerArgs) -> Dict[str, Any]:
    token = a.row.token_id
    raw_dir = Path(a.raw_dir)
    crops_dir = Path(a.crops_dir)
    meta_dir = Path(a.meta_dir)

    raw_path = raw_dir / f"{token}.jpeg"
    raw_cache_index: Dict[str, Path] = {k: Path(v) for k, v in a.raw_cache_index.items()}

    try:
        # IMPORTANT: create a new FaceProcessor per image.
        # `kaodake/libaoki/faceprocessor.py` keeps `_coarse_faces` as a class-level default
        # and doesn't reset it on `load()`, so reusing a single instance can leak detections
        # across different tokens (leading to duplicated crops).
        face = _make_face_processor(a.out_size)

        blob = _read_cached_raw_bytes(token, raw_path=raw_path, raw_cache_index=raw_cache_index)
        resolved_url = ""
        cache_hit = blob is not None

        if blob is None:
            # Spread traffic across gateways (deterministic per token)
            gw_order = _ordered_gateways_for_token(a.gateways, token_id=token)
            blob, resolved_url = _download_with_fallbacks(
                a.row.image_url,
                gateways=gw_order,
                timeout_s=a.timeout_s,
                user_agent=a.user_agent,
                retries=a.retries,
                sleep_s=a.sleep_s,
            )
            # NOTE: the bytes returned by IPFS are typically JPEG, so store with .jpeg (not .bin)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(blob)

        # run face crop
        face.load(blob)
        face.crop_face_coarsely()
        if len(face.coarse_faces) == 0:
            raise RuntimeError("no coarse face detected")

        # Limit coarse faces to prevent explosion (MTCNN keep_all + 4 rotations)
        try:
            max_faces = max(int(a.max_faces), 1)
            face._coarse_faces = face._coarse_faces[:max_faces]  # type: ignore[attr-defined]
        except Exception:
            max_faces = 1

        face.crop_face_finely()
        outs = face.output_with_param
        if len(outs) == 0:
            raise RuntimeError("no fine face detected")

        outs = outs[:max_faces]

        out_items = []
        crop_files: List[str] = []
        for j, out in enumerate(outs):
            img: np.ndarray = out["image"]
            param = out["param"]
            crop_npy = crops_dir / f"{token}_{j}.npy"
            _save_rgb_flat_npy(crop_npy, img)
            crop_files.append(str(crop_npy))
            out_items.append(
                {
                    "token_id": token,
                    "index": j,
                    "crop_npy": str(crop_npy),
                    "param": param,
                }
            )

        meta = {
            "token_id": token,
            "image_url": a.row.image_url,
            "resolved_url": resolved_url,
            "cache_hit": bool(cache_hit),
            "raw_file": str(raw_path),
            "outputs": out_items,
        }
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"{token}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

        return {
            "ok": True,
            "token_id": token,
            "meta": meta,
            "crop_files": crop_files,
        }

    except Exception as e:
        return {
            "ok": False,
            "token_id": token,
            "error": {
                "token_id": token,
                "image_url": a.row.image_url,
                "error": repr(e),
            },
        }


def _load_flat_rows_from_crops(crop_files: List[str]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for p in crop_files:
        arr = np.load(p)
        rows.append(arr.reshape(-1).astype(np.uint8))
    if not rows:
        raise RuntimeError("No crops produced; dataset would be empty.")
    return np.stack(rows, axis=0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample N images from itadakimasu-man CSV (deterministic random), "
            "download from IPFS (with gateway load distribution + cache), crop faces using kaodake, "
            "and build an allimgs_rgb.npy dataset."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(Path("itadakimasu-man") / "itadakimasu-man.csv"),
        help="Path to itadakimasu-man.csv",
    )
    parser.add_argument(
        "--out",
        default=str(Path("itadakimasu-man") / "dataset"),
        help="Output base directory. A unique run subdir will be created under this path.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier. If empty, a unique id like seed<seed>_<timestamp> is used.",
    )
    parser.add_argument("--count", type=int, default=10, help="Target number of NFTs to attempt")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--out-size",
        type=int,
        default=256,
        help="Face crop output size (square)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Max faces to keep per image (prevents OOM when MTCNN detects many faces)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per gateway",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between retries",
    )
    parser.add_argument(
        "--user-agent",
        default="imgadv/itadakimasu-dl",
        help="User-Agent header",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1))),
        help=(
            "Parallel worker processes. Note: FaceProcessor uses CPU mode; "
            "set 1 if you hit OOM."
        ),
    )
    parser.add_argument(
        "--gateways",
        nargs="*",
        default=[
            # Put ipfs.io later; it is heavily used globally.
            "https://cloudflare-ipfs.com",
            "https://gateway.pinata.cloud",
            "https://dweb.link",
            "https://w3s.link",
            "https://ipfs.io",
        ],
        help="List of IPFS gateways (base URL, without /ipfs)",
    )
    parser.add_argument(
        "--raw-cache-roots",
        nargs="*",
        default=[],
        help=(
            "Directories to scan for existing <run>/raw/* files to use as cache. "
            "If omitted, uses --out (dataset base directory)."
        ),
    )

    args = parser.parse_args()

    out_base_dir = Path(args.out)
    run_id = str(args.run_id).strip() or f"seed{int(args.seed)}_{int(time.time())}"
    out_dir = out_base_dir / run_id

    # Record the latest run id for convenience
    try:
        out_base_dir.mkdir(parents=True, exist_ok=True)
        (out_base_dir / "LATEST").write_text(run_id, encoding="utf-8")
    except Exception:
        pass

    raw_dir = out_dir / "raw"
    crops_dir = out_dir / "crops"
    meta_dir = out_dir / "meta"
    _ensure_dir(raw_dir)
    _ensure_dir(crops_dir)
    _ensure_dir(meta_dir)

    csv_path = Path(args.csv)
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows read from CSV: {csv_path}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: max(args.count, 1)]

    cache_roots = [Path(p) for p in (args.raw_cache_roots or [str(out_base_dir)])]
    # Always include out_base_dir to satisfy "既存のrawディレクトリがある場合".
    if out_base_dir not in cache_roots:
        cache_roots.append(out_base_dir)

    # Build cache index once in parent process (avoid repeated scans in workers)
    raw_cache_index = _build_raw_cache_index(cache_roots)

    manifest: Dict[str, Any] = {
        "source_csv": str(csv_path),
        "out_base": str(out_base_dir),
        "run_id": run_id,
        "out_dir": str(out_dir),
        "count_requested": int(args.count),
        "seed": int(args.seed),
        "out_size": int(args.out_size),
        "workers": int(args.workers),
        "gateways": list(args.gateways),
        "raw_cache_roots": [str(p) for p in cache_roots],
        "raw_cache_index_size": int(len(raw_cache_index)),
        "items": [],
        "errors": [],
        "created_at": time.time(),
    }

    # Launch parallel workers
    futures: List[Future] = []
    crop_files_all: List[str] = []

    # Convert cache index to strings for pickling
    raw_cache_index_str: Dict[str, str] = {k: str(v) for k, v in raw_cache_index.items()}

    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        for idx, r in enumerate(rows):
            wa = _WorkerArgs(
                row=r,
                idx=idx,
                total=len(rows),
                out_size=int(args.out_size),
                max_faces=int(args.max_faces),
                timeout_s=int(args.timeout),
                retries=int(args.retries),
                sleep_s=float(args.sleep),
                user_agent=str(args.user_agent),
                gateways=list(args.gateways),
                raw_dir=str(raw_dir),
                crops_dir=str(crops_dir),
                meta_dir=str(meta_dir),
                raw_cache_index=raw_cache_index_str,
            )
            futures.append(ex.submit(_process_one_nft, wa))

        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            token = res.get("token_id", "")
            if res.get("ok"):
                meta = res["meta"]
                manifest["items"].append(meta)
                crop_files_all.extend(res.get("crop_files", []))
                n_faces = len(meta.get("outputs", []))
                cache_hit = bool(meta.get("cache_hit"))
                print(f"[{done}/{len(rows)}] token={token} ok faces={n_faces} cache={cache_hit}")
            else:
                err = res.get("error") or {"token_id": token, "error": "unknown"}
                manifest["errors"].append(err)
                print(f"[{done}/{len(rows)}] token={token} ERROR: {err.get('error')}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    allimgs = _load_flat_rows_from_crops(crop_files_all)
    np.save(str(out_dir / "allimgs_rgb.npy"), allimgs)
    print(f"Saved {allimgs.shape} to {out_dir / 'allimgs_rgb.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
