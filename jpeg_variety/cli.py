"""Command line interface for jpeg_variety."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import PipelineArgs, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jpeg-variety",
        description=(
            "Batch-encode PNG images into 8-bit JPEG using a random encoder and\n"
            "realistically weighted JPEG parameter sampling for artifact diversity."
        ),
    )

    # Exactly 3 required positionals
    p.add_argument("quality", type=int, help="Base JPEG quality (1..100)")
    p.add_argument("src_dir", type=Path, help="Directory containing .png files")
    p.add_argument("dst_dir", type=Path, help="Output directory for .jpg files")

    # Optional flags
    p.add_argument("--seed", type=int, default=None, help="Global seed for deterministic reproducibility")
    p.add_argument("--jobs", type=int, default=0, help="Parallel worker threads (0=auto)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument(
        "--mirror-subdirs",
        action="store_true",
        help="When --recursive, mirror source subdirectories under dst_dir",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path for JSONL manifest (default: dst_dir/encoding_manifest.jsonl)",
    )
    p.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config override")
    p.add_argument("--continue-on-error", action="store_true", help="Log failures and continue")
    p.add_argument("--dry-run", action="store_true", help="Sample encoders/options and write manifest without encoding")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = _build_parser()
    ns = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if ns.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        config = load_config(ns.config)
        args = PipelineArgs(
            base_quality=ns.quality,
            src_dir=ns.src_dir,
            dst_dir=ns.dst_dir,
            recursive=ns.recursive,
            mirror_subdirs=bool(ns.mirror_subdirs),
            jobs=ns.jobs,
            seed=ns.seed,
            manifest_path=ns.manifest,
            continue_on_error=bool(ns.continue_on_error),
            dry_run=bool(ns.dry_run),
            overwrite=bool(ns.overwrite),
        )
        run_pipeline(args, config)
        return 0
    except Exception as e:
        logging.getLogger(__name__).error(str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
