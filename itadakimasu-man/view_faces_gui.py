import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Make `kaodake/libaoki` importable without installing as a package
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "kaodake"))

from PIL import Image

from libaoki.faceprocessor import FaceProcessor
from libaoki.interfaces import LoadStrategy, OutputStrategy, PostprocessStrategy, PreprocessStrategy


class IdentityPreprocessor(PreprocessStrategy):
    def process(self, image: Image.Image) -> Image.Image:
        return image


class RGBPostprocessor(PostprocessStrategy):
    def process(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


class BytesLoader(LoadStrategy[bytes]):
    def process(self, data: bytes) -> Image.Image:
        import io

        im = Image.open(io.BytesIO(data))
        im.load()
        return im


class NumpySquareOutputter(OutputStrategy[np.ndarray]):
    def __init__(self, size: int = 256):
        self._size = int(size)

    def process(self, image: Image.Image) -> np.ndarray:
        return np.array(image.resize((self._size, self._size), Image.BILINEAR))


def _make_face_processor(out_size: int) -> FaceProcessor:
    return FaceProcessor(
        preprocessor=IdentityPreprocessor(),
        postprocessor=RGBPostprocessor(),
        loader=BytesLoader(),
        outputter=NumpySquareOutputter(out_size),
        # same params as dataset builder
        fine_down_expand_ratio=1.5,
        fine_up_expand_ratio=0.7,
        fine_side_expand_ratio=0.5,
        coarse_resize_ratio=0.125,
        coarse_expand_ratio=0.25,
    )


def _load_bytes_from_dataset(out_dir: Path, token: str) -> bytes:
    # NOTE: builder saves IPFS image bytes as JPEG with `.jpeg` extension (not `.bin`).
    raw_path = out_dir / "raw" / f"{token}.jpeg"
    return raw_path.read_bytes()


def _list_tokens_from_manifest(out_dir: Path) -> List[str]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    return [str(it.get("token_id")) for it in items if it.get("token_id") is not None]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GUI viewer for multi-face detections (coarse boxes + fine crops)."
    )
    parser.add_argument(
        "--dataset",
        default=str(Path("itadakimasu-man") / "dataset"),
        help="Output dir created by build_itadakimasu_dataset.py (contains raw/, meta/, manifest.json)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Token ID to open (must exist under dataset/raw/<token>.jpeg)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index into manifest items (when --token is not provided)",
    )
    parser.add_argument(
        "--out-size",
        type=int,
        default=256,
        help="Crop output size",
    )
    args = parser.parse_args()

    out_dir = Path(args.dataset)

    tokens: List[str] = _list_tokens_from_manifest(out_dir)

    if args.token is None:
        if not tokens:
            raise SystemExit(
                f"No manifest tokens found. Run the builder first or pass --token. dataset={out_dir}"
            )
        token_idx = int(args.index) % len(tokens)
        token = tokens[token_idx]
    else:
        token = str(args.token)
        token_idx = tokens.index(token) if token in tokens else -1

    def load_token(token_id: str):
        blob = _load_bytes_from_dataset(out_dir, token_id)
        fp = _make_face_processor(int(args.out_size))
        fp.load(blob)
        fp.crop_face_coarsely()

        overlays: List[Image.Image] = [sub.render_area() for sub in fp.coarse_faces]

        fp.crop_face_finely()
        fine_crops: List[np.ndarray] = list(fp.output)

        return fp, overlays, fine_crops

    fp, overlays, fine_crops = load_token(token)

    # --- GUI ---
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required. Install with: pip install matplotlib")
        print(e)
        return 3

    def _counts():
        return len(overlays), len(fine_crops)

    n_coarse, n_fine = _counts()

    fig = plt.figure(f"itadakimasu-man faces | token={token}")

    # Layout: left = overlay, right = fine crop (if exists)
    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    cur = 0

    def render(i: int) -> None:
        nonlocal cur
        nonlocal n_coarse, n_fine
        cur = i
        n_coarse, n_fine = _counts()

        ax_left.clear()
        ax_right.clear()

        if n_coarse > 0:
            ax_left.imshow(overlays[i])
            ax_left.set_title(f"coarse[{i}] / {n_coarse} (overlay on original)")
        else:
            ax_left.text(0.5, 0.5, "NO COARSE FACES", ha="center", va="center")

        if n_fine > 0 and i < n_fine:
            ax_right.imshow(fine_crops[i])
            ax_right.set_title(f"fine[{i}] / {n_fine} (cropped)")
        else:
            ax_right.text(0.5, 0.5, "NO FINE CROP", ha="center", va="center")

        for ax in (ax_left, ax_right):
            ax.axis("off")

        token_pos = f"{token_idx+1}/{len(tokens)}" if token_idx >= 0 and tokens else "(manual)"
        hint = "←/→ or a/d: face  |  n/p: next/prev token  |  q: quit"
        if n_coarse <= 1:
            hint = "(only 0-1 face) " + hint

        fig.suptitle(
            f"token={token} ({token_pos}) | coarse={n_coarse} fine={n_fine} | {hint}",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    def _switch_token(delta: int) -> None:
        nonlocal token
        nonlocal token_idx
        nonlocal fp, overlays, fine_crops
        if not tokens:
            return
        token_idx = (token_idx + delta) % len(tokens)
        token = tokens[token_idx]
        fp, overlays, fine_crops = load_token(token)
        render(0)

    def on_key(event):
        nonlocal cur
        if event.key in ("right", "d", " "):
            if n_coarse > 1:
                render((cur + 1) % n_coarse)
        elif event.key in ("left", "a"):
            if n_coarse > 1:
                render((cur - 1) % n_coarse)
        elif event.key in ("n", "down"):
            _switch_token(+1)
        elif event.key in ("p", "up"):
            _switch_token(-1)
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    render(0)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
