"""File discovery and output path mirroring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiscoveredFile:
    input_path: Path
    rel_path: Path  # path relative to the chosen source root


def iter_png_files(src_dir: Path, recursive: bool) -> list[DiscoveredFile]:
    """Return PNG files under src_dir.

    The returned list is sorted by relative path to stabilize processing order.
    """

    src_dir = src_dir.resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"src_dir does not exist or is not a directory: {src_dir}")

    pattern = "**/*.png" if recursive else "*.png"
    files: list[DiscoveredFile] = []

    for p in src_dir.glob(pattern):
        if not p.is_file():
            continue
        # Only .png (case-insensitive)
        if p.suffix.lower() != ".png":
            continue
        rel = p.resolve().relative_to(src_dir)
        files.append(DiscoveredFile(input_path=p.resolve(), rel_path=rel))

    files.sort(key=lambda f: f.rel_path.as_posix())
    return files


def mirror_output_path(dst_dir: Path, rel_png_path: Path, mirror_subdirs: bool) -> Path:
    """Compute output JPG path.

    - If mirror_subdirs is True, preserve subdirectories under dst_dir.
    - Always replace suffix with .jpg.
    """

    rel_no_suffix = rel_png_path.with_suffix(".jpg")
    if mirror_subdirs:
        return (dst_dir / rel_no_suffix).resolve()
    return (dst_dir / rel_no_suffix.name).resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
