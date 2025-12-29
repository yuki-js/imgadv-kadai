from __future__ import annotations

from pathlib import Path


def write_mask_template_bmp(path: Path, *, w: int = 32, h: int = 32, ax: float = 12.0, ay: float = 14.5) -> None:
    """Write a simple 32x32 BMP mask template editable in MS Paint.

    - White (255): keep
    - Black (0): drop

    BMP is written as 24-bit uncompressed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    # 24-bit BMP rows padded to 4 bytes
    row_bytes = ((w * 3 + 3) // 4) * 4
    pixel_data_size = row_bytes * h
    file_header_size = 14
    dib_header_size = 40
    offset = file_header_size + dib_header_size
    file_size = offset + pixel_data_size

    def le16(x: int) -> bytes:
        return int(x).to_bytes(2, "little", signed=False)

    def le32(x: int) -> bytes:
        return int(x).to_bytes(4, "little", signed=False)

    def le32s(x: int) -> bytes:
        return int(x).to_bytes(4, "little", signed=True)

    fh = b"BM" + le32(file_size) + le16(0) + le16(0) + le32(offset)
    dib = (
        le32(dib_header_size)
        + le32s(w)
        + le32s(h)
        + le16(1)
        + le16(24)
        + le32(0)
        + le32(pixel_data_size)
        + le32s(2835)
        + le32s(2835)
        + le32(0)
        + le32(0)
    )

    # Ellipse params (centered), white inside, black outside
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    rows: list[bytes] = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            r = ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2
            v = 255 if r <= 1.0 else 0
            # BMP pixel order: B, G, R
            row += bytes([v, v, v])
        row += b"\x00" * (row_bytes - w * 3)
        rows.append(bytes(row))

    # BMP stores rows bottom-up
    pixels = b"".join(reversed(rows))

    path.write_bytes(fh + dib + pixels)


def main() -> None:
    out = Path("src") / "artifacts" / "mask_template_32.bmp"
    write_mask_template_bmp(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
