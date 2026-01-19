"""MozJPEG/libjpeg-turbo compatible `cjpeg` encoder plugin.

We focus on knobs that materially change the produced JPEG bitstream and the
visible artifact distribution:
- baseline vs progressive
- subsampling factors
- quantization table choice (predefined tables; rare custom tables)
- DCT method variants
- restart markers
- progressive scan optimization / trellis-related tuning

Implementation detail:
`cjpeg` does not natively consume PNG, so we convert PNG->PPM (P6) using Pillow
and stream it via stdin.

Option inventory source: cjpeg help output (provided with this project).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..config import ANNEX_K_CHROMA_TABLE, ANNEX_K_LUMA_TABLE
from ..utils.image import png_to_ppm_file
from ..utils.quant_tables import perturb_table, write_qtables_file
from ..utils.sampling import bernoulli, triangular_int, weighted_choice
from ..utils.subprocess import RunResult, run
from . import register_encoder
from .base import EncodeContext, EncoderOptions, JPEGEncoder

log = logging.getLogger(__name__)


@register_encoder
class CjpegEncoder(JPEGEncoder):
    name = "cjpeg"
    default_weight = 0.75

    def __init__(self) -> None:
        self._exe = shutil.which("cjpeg")

    def is_available(self) -> bool:
        return self._exe is not None

    def sample_options(self, base_quality: int, rng, context: EncodeContext) -> EncoderOptions:
        bucket = context.quality_bucket
        sampling = context.sampling
        cfg = context.encoder_sampling.cjpeg

        progressive = bernoulli(rng, cfg.progressive_adjustments.apply(sampling.progressive_prob, bucket))
        subsampling = weighted_choice(rng, cfg.subsampling_overrides.for_bucket(sampling.subsampling_weights, bucket))

        # DCT variants: int dominates, fast/float are legacy and rare.
        dct = weighted_choice(rng, cfg.dct_weights)

        # Quantization table selection.
        quant_weights = sampling.quant_kind_weights
        adjustment = cfg.quant_kind_adjustments.get(bucket)
        if adjustment is not None:
            quant_weights = adjustment.apply(quant_weights)
        quant_kind = weighted_choice(rng, quant_weights)

        if quant_kind == "annex_k":
            quant: dict[str, Any] = {"kind": "predefined", "id": 0}
        elif quant_kind == "imagemagick":
            quant = {"kind": "predefined", "id": 3}
        elif quant_kind == "perceptual":
            qid = weighted_choice(rng, sampling.perceptual_table_weights)
            quant = {"kind": "predefined", "id": int(qid)}
        else:
            strength = cfg.custom_quant_strength_by_bucket.for_bucket(bucket)
            quant = {"kind": "custom", "generator": "perturbed_annexk", "strength": strength}

        # Progressive-related knobs.
        dc_scan_opt: int | None = None
        fastcrush = False
        if progressive:
            weights = cfg.dc_scan_opt_weights_high if bucket == "high" else cfg.dc_scan_opt_weights_mid
            dc_scan_opt = int(weighted_choice(rng, weights))
            fastcrush = bernoulli(rng, cfg.fastcrush_prob_by_bucket.for_bucket(bucket))

        # Trellis toggles and tuning.
        trellis_disable_kind: str | None = None
        if bernoulli(rng, cfg.trellis_disable_prob_by_bucket.for_bucket(bucket)):
            trellis_disable_kind = weighted_choice(rng, cfg.trellis_disable_weights)

        tune_mode = None
        if trellis_disable_kind is None:
            tune_mode = weighted_choice(rng, cfg.tune_mode_weights)

        # Restart markers: occasional, biased off.
        restart: dict[str, Any] | None = None
        restart_p = cfg.restart_adjustments.apply(sampling.restart_prob, bucket)
        if bernoulli(rng, restart_p):
            use_blocks = bernoulli(rng, cfg.restart_blocks_prob)
            if use_blocks:
                blocks = triangular_int(
                    rng, cfg.restart_blocks_range.low, cfg.restart_blocks_range.high, cfg.restart_blocks_range.mode
                )
                restart = {"unit": "blocks", "value": blocks}
            else:
                rows = triangular_int(
                    rng, cfg.restart_rows_range.low, cfg.restart_rows_range.high, cfg.restart_rows_range.mode
                )
                restart = {"unit": "rows", "value": rows}

        # Baseline compatibility when custom tables or baseline mode.
        quant_baseline = False
        if not progressive and bernoulli(rng, cfg.quant_baseline_prob_nonprogressive):
            quant_baseline = True
        if quant.get("kind") == "custom" and bernoulli(rng, cfg.quant_baseline_prob_custom):
            quant_baseline = True

        normalized: dict[str, Any] = {
            "quality": int(base_quality),
            "progressive": bool(progressive),
            "subsampling": subsampling,
            "quant_table": quant,
            "dct": dct,
            "entropy": {"huffman_opt": True, "arithmetic": False},
            "restart": restart,
            "encoder_specific": {
                "dc_scan_opt": dc_scan_opt,
                "fastcrush": fastcrush,
                "trellis_disable": trellis_disable_kind,
                "tune": tune_mode,
                "quant_baseline": quant_baseline,
            },
        }

        # Store runtime details (avoid creating temp files here).
        internal: dict[str, Any] = {
            "progressive": progressive,
            "subsampling": subsampling,
            "dct": dct,
            "dc_scan_opt": dc_scan_opt,
            "fastcrush": fastcrush,
            "trellis_disable_kind": trellis_disable_kind,
            "tune_mode": tune_mode,
            "restart": restart,
            "quant": quant,
            "quant_baseline": quant_baseline,
            "cmd_template": ["cjpeg", "-quality", str(base_quality), "...", "-outfile", "<out.jpg>"]
        }

        # For custom quant, store concrete table matrices.
        if quant.get("kind") == "custom":
            strength = float(quant.get("strength", 0.12))
            internal["custom_tables"] = {
                "luma": perturb_table(rng, [list(row) for row in ANNEX_K_LUMA_TABLE], strength),
                "chroma": perturb_table(rng, [list(row) for row in ANNEX_K_CHROMA_TABLE], strength),
            }

        return EncoderOptions(normalized=normalized, internal=internal)

    def preview_cmd(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> list[str]:
        ppm_path = Path("./input.ppm")
        output_path = Path("./output.jpg")
        return self._build_cmd(ppm_path, output_path, options, qtables_path=Path("./qtables.txt"))

    def _build_cmd(
        self,
        ppm_path: Path,
        output_jpg: Path,
        options: EncoderOptions,
        *,
        qtables_path: Path | None = None,
    ) -> list[str]:
        exe = self._exe or "cjpeg"
        base_quality = int(options.normalized.get("quality", 75))

        cmd: list[str] = [exe, "-quality", str(base_quality)]

        progressive = bool(options.internal.get("progressive", False))
        if progressive:
            cmd.append("-progressive")
        else:
            cmd.append("-baseline")

        # Subsampling
        subsampling = str(options.internal.get("subsampling", "420"))
        sample_map = {
            "444": "1x1,1x1,1x1",
            "422": "2x1,1x1,1x1",
            "420": "2x2,1x1,1x1",
        }
        cmd.extend(["-sample", sample_map.get(subsampling, "2x2,1x1,1x1")])

        # DCT method
        dct = str(options.internal.get("dct", "int"))
        cmd.extend(["-dct", dct])

        # Quantization
        quant = options.internal.get("quant", {})
        if isinstance(quant, dict) and quant.get("kind") == "predefined":
            cmd.extend(["-quant-table", str(int(quant.get("id", 0)))])
        elif isinstance(quant, dict) and quant.get("kind") == "custom":
            if qtables_path is None:
                qtables_path = Path("./qtables.txt")
            cmd.extend(["-qtables", str(qtables_path), "-qslots", "0,1,1"])

        if bool(options.internal.get("quant_baseline", False)):
            cmd.append("-quant-baseline")

        # Progressive scan knobs
        dc_scan_opt = options.internal.get("dc_scan_opt")
        if progressive and dc_scan_opt is not None:
            cmd.extend(["-dc-scan-opt", str(int(dc_scan_opt))])

        if progressive and bool(options.internal.get("fastcrush", False)):
            cmd.append("-fastcrush")

        # Trellis tuning / disable
        tdk = options.internal.get("trellis_disable_kind")
        if tdk == "notrellis":
            cmd.append("-notrellis")
        elif tdk == "notrellis_dc":
            cmd.append("-notrellis-dc")
        else:
            tune_mode = options.internal.get("tune_mode")
            if tune_mode == "tune_psnr":
                cmd.append("-tune-psnr")
            elif tune_mode == "tune_ssim":
                cmd.append("-tune-ssim")
            elif tune_mode == "tune_ms_ssim":
                cmd.append("-tune-ms-ssim")
            else:
                # Default mozjpeg tuning is PSNR-HVS; specifying explicitly improves manifest clarity.
                cmd.append("-tune-hvs-psnr")

        # Restart markers
        restart = options.internal.get("restart")
        if isinstance(restart, dict) and restart.get("value") is not None:
            v = int(restart["value"])
            if restart.get("unit") == "blocks":
                cmd.extend(["-restart", f"{v}B"])
            else:
                cmd.extend(["-restart", str(v)])

        output_jpg = Path(output_jpg).expanduser().resolve()
        cmd.extend(["-outfile", os.fspath(output_jpg), str(ppm_path)])

        return cmd

    def encode(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> RunResult:
        if self._exe is None:
            raise RuntimeError("cjpeg executable not found on PATH")

        
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            ppm_path = Path(f.name)
        options.temp_paths.append(ppm_path)
        png_to_ppm_file(input_png, ppm_path)

        qtables_path: Path | None = None
        quant = options.internal.get("quant", {})
        if isinstance(quant, dict) and quant.get("kind") == "custom":
            tables = options.internal.get("custom_tables")
            if not isinstance(tables, dict) or "luma" not in tables or "chroma" not in tables:
                raise RuntimeError("custom quant selected but custom_tables missing")
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
                qtables_path = Path(f.name)
            write_qtables_file(qtables_path, tables["luma"], tables["chroma"])
            options.temp_paths.append(qtables_path)

        cmd = self._build_cmd(ppm_path, output_jpg, options, qtables_path=qtables_path)

        return run(cmd)
