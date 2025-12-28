import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    # NOTE: faceprocessor internally forces device="cuda:0". If CUDA isn't available,
    # it will raise. We'll handle errors at call-site.
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample N images from itadakimasu-man CSV (deterministic random), download from IPFS, crop faces using kaodake, and build an allimgs_rgb.npy dataset."
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
        "--gateways",
        nargs="*",
        default=[
            "https://ipfs.io",
            "https://cloudflare-ipfs.com",
            "https://gateway.pinata.cloud",
        ],
        help="List of IPFS gateways (base URL, without /ipfs)",
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

    manifest: Dict[str, Any] = {
        "source_csv": str(csv_path),
        "out_base": str(out_base_dir),
        "run_id": run_id,
        "out_dir": str(out_dir),
        "count_requested": int(args.count),
        "seed": int(args.seed),
        "out_size": int(args.out_size),
        "gateways": list(args.gateways),
        "items": [],
        "errors": [],
        "created_at": time.time(),
    }

    flat_rows: List[np.ndarray] = []

    for idx, r in enumerate(rows):
        token = r.token_id
        try:
            # IMPORTANT: create a new FaceProcessor per image.
            # `kaodake/libaoki/faceprocessor.py` keeps `_coarse_faces` as a class-level default
            # and doesn't reset it on `load()`, so reusing a single instance can leak detections
            # across different tokens (leading to duplicated crops).
            face = _make_face_processor(args.out_size)

            blob, resolved_url = _download_with_fallbacks(
                r.image_url,
                gateways=list(args.gateways),
                timeout_s=int(args.timeout),
                user_agent=str(args.user_agent),
                retries=int(args.retries),
                sleep_s=float(args.sleep),
            )

            # NOTE: the bytes returned by IPFS are typically JPEG, so store with .jpeg (not .bin)
            raw_path = raw_dir / f"{token}.jpeg"
            raw_path.write_bytes(blob)

            # run face crop
            face.load(blob)
            face.crop_face_coarsely()
            if len(face.coarse_faces) == 0:
                raise RuntimeError("no coarse face detected")

            # Limit coarse faces to prevent explosion (MTCNN keep_all + 4 rotations)
            try:
                max_faces = max(int(args.max_faces), 1)
                face._coarse_faces = face._coarse_faces[:max_faces]  # type: ignore[attr-defined]
            except Exception:
                max_faces = 1

            face.crop_face_finely()
            outs = face.output_with_param
            if len(outs) == 0:
                raise RuntimeError("no fine face detected")

            outs = outs[:max_faces]

            out_items = []
            for j, out in enumerate(outs):
                img: np.ndarray = out["image"]
                param = out["param"]
                crop_npy = crops_dir / f"{token}_{j}.npy"
                _save_rgb_flat_npy(crop_npy, img)
                flat_rows.append(img.reshape(-1).astype(np.uint8))
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
                "image_url": r.image_url,
                "resolved_url": resolved_url,
                "raw_file": str(raw_path),
                "outputs": out_items,
            }
            (meta_dir / f"{token}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

            manifest["items"].append(meta)
            print(f"[{idx+1}/{len(rows)}] token={token} ok faces={len(out_items)}")

        except Exception as e:
            err = {
                "token_id": token,
                "image_url": r.image_url,
                "error": repr(e),
            }
            manifest["errors"].append(err)
            print(f"[{idx+1}/{len(rows)}] token={token} ERROR: {e}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not flat_rows:
        raise RuntimeError("No crops produced; dataset would be empty.")

    # Build allimgs_rgb.npy: (N, out_size*out_size*3) uint8
    allimgs = np.stack(flat_rows, axis=0).astype(np.uint8)
    np.save(str(out_dir / "allimgs_rgb.npy"), allimgs)
    print(f"Saved {allimgs.shape} to {out_dir / 'allimgs_rgb.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
