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
    min_quality: int
    max_quality: int
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
    log_line: str


def _write_jsonl_line(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _effective_jobs(requested: int) -> int:
    if requested and requested > 0:
        return requested
    # Subprocess-heavy => threads scale decently
    return max(1, (os.cpu_count() or 4))


def _is_path_token(token: str) -> bool:
    if not token or token.startswith("<"):
        return False
    if token.startswith("~"):
        return True
    if token.startswith("./") or token.startswith("../"):
        return True
    if os.sep in token or (os.altsep and os.altsep in token):
        return True
    suffix = Path(token).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg"}


def _format_log_line(image_name: str, encoder_name: str, cmd: list[str]) -> str:
    def escape(text: str) -> str:
        return text.replace('"', r"\"")

    options = [token for token in cmd[1:] if not _is_path_token(token)] if cmd else []
    options_text = " ".join(options)
    return f'"{escape(image_name)}" : "{escape(encoder_name)}" "{escape(options_text)}"\n'


def run_pipeline(args: PipelineArgs, config: AppConfig) -> Path:
    """Run the batch encoding pipeline.

    Returns the manifest path.
    """

    if not (1 <= args.min_quality <= 100) or not (1 <= args.max_quality <= 100):
        raise ValueError("quality must be in 1..100")
    if args.min_quality > args.max_quality:
        raise ValueError("min_quality must be <= max_quality")

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
        base_quality = rng.randint(args.min_quality, args.max_quality)
        bucket = quality_bucket(base_quality).name
        ctx = EncodeContext(
            src_root=src,
            dst_root=dst,
            rel_path=rel,
            quality_bucket=bucket,
            per_file_seed=pf_seed,
            sampling=config.sampling,
            encoder_sampling=config.encoder_sampling,
        )

        out_path = mirror_output_path(dst, rel, args.mirror_subdirs)
        ensure_parent_dir(out_path)

        encoder = factory.choose(rng)
        options = encoder.sample_options(base_quality, rng, ctx)

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

        encoder_name = getattr(encoder, "name", encoder.__class__.__name__)
        manifest: dict[str, Any] = {
            "input": str(item.input_path),
            "output": str(out_path),
            "encoder": encoder_name,
            "cmd": cmd,
            "seed": seed_ctx.run_seed,
            "per_file_seed": pf_seed,
        }
        # Encoder-normalized fields (quality, subsampling, progressive, ...)
        manifest.update(options.normalized)

        if error is not None:
            manifest["error"] = error

        log_line = _format_log_line(item.input_path.name, encoder_name, cmd)
        return EncodeResult(manifest=manifest, ok=ok, log_line=log_line)

    # Run
    failures = 0
    log_path = manifest_path.parent / "encoding.log"
    with manifest_path.open("w", encoding="utf-8") as fp, log_path.open("w", encoding="utf-8") as log_fp:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(work, f) for f in files]

            for fut in as_completed(futs):
                result = fut.result()
                _write_jsonl_line(fp, result.manifest)
                log_fp.write(result.log_line)

                if not result.ok:
                    failures += 1
                    if not args.continue_on_error:
                        raise RuntimeError(result.manifest.get("error", "encoding failed"))

    if failures:
        log.warning("Completed with %d failures. See manifest: %s", failures, manifest_path)
    else:
        log.info("Completed successfully. Manifest: %s", manifest_path)

    return manifest_path
