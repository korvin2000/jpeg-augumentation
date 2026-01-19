"""Image-related helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def png_to_ppm_file(
    png_path: Path, ppm_path: Path, *, background_rgb: tuple[int, int, int] = (255, 255, 255)
) -> None:
    """Convert a PNG into a binary PPM (P6) file."""

    with Image.open(png_path) as im:
        im.load()
        if im.mode in {"RGBA", "LA"} or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGBA", im.size, background_rgb + (255,))
            rgba = im.convert("RGBA")
            comp = Image.alpha_composite(bg, rgba).convert("RGB")
            im_rgb = comp
        else:
            im_rgb = im.convert("RGB")

        w, h = im_rgb.size
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        data = im_rgb.tobytes()
        ppm_path.write_bytes(header + data)
