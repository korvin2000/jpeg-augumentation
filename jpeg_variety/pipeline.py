"""Encoding pipeline (file iteration, deterministic RNG, metadata writing)."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .encoders import EncoderFactory
from .encoders.base import EncodeContext, EncoderOptions, JPEGEncoder
from .utils.files import DiscoveredFile, ensure_parent_dir, iter_png_files, mirror_output_path
from .utils.rng import SeedContext, make_run_seed, per_file_seed, rng_for_file
from .utils.sampling import quality_bucket

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineArgs:
    base_quality: int
    src_dir: Path
    dst_dir: Path
    recursive: bool = False
    mirror_subdirs: bool = False
    jobs: int = 0  # 0 => auto
    seed: int | None = None
    manifest_path: Path | None = None
    continue_on_error: bool = False
    dry_run: bool = False
    overwrite: bool = False


@dataclass
class EncodeResult:
    manifest: dict[str, Any]
    ok: bool


def _write_jsonl_line(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _effective_jobs(requested: int) -> int:
    if requested and requested > 0:
        return requested
    # Subprocess-heavy => threads scale decently
    return max(1, (os.cpu_count() or 4))


def run_pipeline(args: PipelineArgs, config: AppConfig) -> Path:
    """Run the batch encoding pipeline.

    Returns the manifest path.
    """

    if not (1 <= args.base_quality <= 100):
        raise ValueError("quality must be in 1..100")

    src = args.src_dir.expanduser().resolve()
    dst = args.dst_dir.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest_path or (dst / "encoding_manifest.jsonl")

    seed_ctx: SeedContext = make_run_seed(args.seed)

    files = iter_png_files(src, args.recursive)
    if not files:
        raise RuntimeError(f"No .png files found in: {src}")

    factory = EncoderFactory(config)
    factory.require_any()

    log.info("Found %d PNG files", len(files))
    log.info("Available encoders: %s", ", ".join(factory.available_names))

    jobs = _effective_jobs(args.jobs)

    def work(item: DiscoveredFile) -> EncodeResult:
        rel = item.rel_path
        pf_seed = per_file_seed(seed_ctx, rel)
        rng = rng_for_file(seed_ctx, rel)

        bucket = quality_bucket(args.base_quality).name
        ctx = EncodeContext(
            src_root=src,
            dst_root=dst,
            rel_path=rel,
            quality_bucket=bucket,
            per_file_seed=pf_seed,
            sampling=config.sampling,
        )

        out_path = mirror_output_path(dst, rel, args.mirror_subdirs)
        ensure_parent_dir(out_path)

        encoder = factory.choose(rng)
        options = encoder.sample_options(args.base_quality, rng, ctx)

        # Encode, unless dry-run / skip existing
        cmd: list[str] | None = None
        ok = True
        error: str | None = None

        if out_path.exists() and not args.overwrite:
            ok = True
            options.normalized["skipped_existing"] = True
            cmd = ["<skipped: exists>"]
        elif args.dry_run:
            ok = True
            options.normalized["dry_run"] = True
            # Some encoders may not be able to fully resolve cmd without IO;
            # they set an approximate cmd template in options.internal.
            cmd = options.internal.get("cmd_template")
            if not isinstance(cmd, list):
                cmd = ["<dry-run>"]
        else:
            try:
                res = encoder.encode(item.input_path, out_path, options)
                cmd = res.cmd
            except Exception as e:
                ok = False
                error = str(e)
            finally:
                # Ensure temp files are removed even if encoding fails.
                encoder.cleanup(options)

        # Some plugins may create temp paths during option sampling.
        # Clean them up in all modes.
        if args.dry_run or (out_path.exists() and not args.overwrite):
            encoder.cleanup(options)

        if cmd is None:
            cmd = ["<unknown>"]

        manifest: dict[str, Any] = {
            "input": str(item.input_path),
            "output": str(out_path),
            "encoder": getattr(encoder, "name", encoder.__class__.__name__),
            "cmd": cmd,
            "seed": seed_ctx.run_seed,
            "per_file_seed": pf_seed,
        }
        # Encoder-normalized fields (quality, subsampling, progressive, ...)
        manifest.update(options.normalized)

        if error is not None:
            manifest["error"] = error

        return EncodeResult(manifest=manifest, ok=ok)

    # Run
    failures = 0
    with manifest_path.open("w", encoding="utf-8") as fp:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(work, f) for f in files]

            for fut in as_completed(futs):
                result = fut.result()
                _write_jsonl_line(fp, result.manifest)

                if not result.ok:
                    failures += 1
                    if not args.continue_on_error:
                        raise RuntimeError(result.manifest.get("error", "encoding failed"))

    if failures:
        log.warning("Completed with %d failures. See manifest: %s", failures, manifest_path)
    else:
        log.info("Completed successfully. Manifest: %s", manifest_path)

    return manifest_path
