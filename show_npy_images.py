import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Best-effort conversion to uint8 for display."""
    if img.dtype == np.uint8:
        return img

    # If float in [0,1], scale to [0,255]
    if np.issubdtype(img.dtype, np.floating):
        mn = float(np.nanmin(img))
        mx = float(np.nanmax(img))
        if 0.0 <= mn and mx <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0.0, 255.0)
        return img.astype(np.uint8)

    # For ints, clip to [0,255]
    if np.issubdtype(img.dtype, np.integer):
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    # Fallback
    img = np.nan_to_num(img)
    img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _normalize_shape(imgs: np.ndarray) -> Tuple[np.ndarray, str]:
    """Normalize to (N,H,W,C) where C in {1,3,4}. Returns (imgs, info)."""
    info = f"dtype={imgs.dtype}, shape={imgs.shape}"

    if imgs.ndim == 4:
        # (N,H,W,C) or (N,C,H,W)
        if imgs.shape[-1] in (1, 3, 4):
            return imgs, info
        if imgs.shape[1] in (1, 3, 4):
            return np.transpose(imgs, (0, 2, 3, 1)), info + " (transposed from NCHW)"

    if imgs.ndim == 3:
        # (N,H,W) grayscale
        return imgs[..., None], info + " (expanded grayscale channel)"

    if imgs.ndim == 2:
        # (N, H*W*C) flattened
        n, d = imgs.shape

        # Try RGB first: d = H*W*3
        if d % 3 == 0:
            hw = d // 3
            h = int(round(hw**0.5))
            if h * h == hw:
                return imgs.reshape(n, h, h, 3), info + f" (reshaped flattened RGB to {h}x{h}x3)"

        # Try grayscale: d = H*W
        h = int(round(d**0.5))
        if h * h == d:
            return imgs.reshape(n, h, h, 1), info + f" (reshaped flattened gray to {h}x{h}x1)"

    raise ValueError(
        "Unsupported array shape for image batch. Expected (N,H,W,C), (N,C,H,W), (N,H,W), or flattened (N,H*W*C). "
        f"Got: {info}"
    )


def _try_load_itadakimasu_manifest(npy_path: str, n_expected: int) -> Optional[List[Dict[str, Any]]]:
    """If the npy is under itadakimasu-man/dataset/, try to load manifest.json.

    Returns a list aligned with rows of allimgs_rgb.npy, each entry containing:
    token_id, index (crop index), crop_npy, raw_file, resolved_url.
    """

    p = Path(npy_path)
    manifest_path = p.parent / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    items = data.get("items", []) or []
    rows: List[Dict[str, Any]] = []

    for it in items:
        token_id = it.get("token_id")
        raw_file = it.get("raw_file")
        resolved_url = it.get("resolved_url")
        for out in it.get("outputs", []) or []:
            rows.append(
                {
                    "token_id": token_id,
                    "index": out.get("index"),
                    "crop_npy": out.get("crop_npy"),
                    "raw_file": raw_file,
                    "resolved_url": resolved_url,
                }
            )

    if len(rows) != int(n_expected):
        return None

    return rows


def main() -> int:
    npy_path = os.environ.get("IMG_NPY", "datasets/allimgs_rgb.npy")
    if len(sys.argv) >= 2:
        npy_path = sys.argv[1]

    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return 2

    imgs = np.load(npy_path)
    imgs, info = _normalize_shape(imgs)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required. Install with: pip install matplotlib")
        print(e)
        return 3

    n = int(imgs.shape[0])
    if n <= 0:
        print(f"No images found in array: {info}")
        return 4

    row_meta = _try_load_itadakimasu_manifest(npy_path=npy_path, n_expected=n)

    idx = 0

    fig = plt.figure("NPY Image Viewer")
    ax = fig.add_subplot(111)

    def render(i: int) -> None:
        ax.clear()
        img = imgs[i]
        img = _to_uint8(img)

        if img.shape[-1] == 1:
            ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)

        extra = ""
        if row_meta is not None:
            m = row_meta[i]
            extra = f" | token={m.get('token_id')} crop_index={m.get('index')}"

        ax.set_title(f"{os.path.basename(npy_path)}  |  {i + 1}/{n}{extra}  |  {info}")
        ax.axis("off")
        fig.canvas.draw_idle()

        if row_meta is not None:
            m = row_meta[i]
            print(
                f"[{i+1}/{n}] token={m.get('token_id')} crop_index={m.get('index')} crop_npy={m.get('crop_npy')} raw_file={m.get('raw_file')}"
            )

    def on_key(event):
        nonlocal idx
        if event.key in ("right", "d", " "):
            idx = (idx + 1) % n
            render(idx)
        elif event.key in ("left", "a"):
            idx = (idx - 1) % n
            render(idx)
        elif event.key in ("home",):
            idx = 0
            render(idx)
        elif event.key in ("end",):
            idx = n - 1
            render(idx)
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    render(idx)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
