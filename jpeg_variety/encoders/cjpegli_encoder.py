"""JPEG XL project's cjpegli encoder plugin.

We expose variability knobs that impact the bitstream/artifacts:
- progressive level (0..2, with 0 being sequential)
- chroma subsampling (420/422/444 plus rare 440)
- std quantization tables vs adaptive quantization
- XYB colorspace and adaptive quantization toggles
- fixed Huffman codes (rare; only valid for progressive level 0)

Option inventory source: cjpegli help output (provided with this project).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..utils.image import png_to_rgb_png_file
from ..utils.sampling import bernoulli, weighted_choice
from ..utils.subprocess import RunResult, run
from . import register_encoder
from .base import EncodeContext, EncoderOptions, JPEGEncoder

@register_encoder
class CjpegliEncoder(JPEGEncoder):
    name = "cjpegli"
    default_weight = 0.20

    def __init__(self) -> None:
        self._exe = shutil.which("cjpegli")

    def is_available(self) -> bool:
        return self._exe is not None

    def sample_options(self, base_quality: int, rng, context: EncodeContext) -> EncoderOptions:
        bucket = context.quality_bucket
        sampling = context.sampling
        cfg = context.encoder_sampling.cjpegli

        progressive = bernoulli(rng, cfg.progressive_adjustments.apply(sampling.progressive_prob, bucket))
        progressive_level = 0
        if progressive:
            progressive_level = int(weighted_choice(rng, cfg.progressive_level_weights))

        subsampling = weighted_choice(rng, cfg.subsampling_overrides.for_bucket(sampling.subsampling_weights, bucket))
        if subsampling == "420" and bernoulli(rng, cfg.subsampling_440_prob_by_bucket.for_bucket(bucket)):
            subsampling = "440"

        quant_kind = weighted_choice(rng, sampling.quant_kind_weights)
        std_quant = quant_kind == "annex_k"

        noadaptive_quant = False
        if not std_quant:
            noadaptive_quant = bernoulli(rng, cfg.noadaptive_quant_prob_by_bucket.for_bucket(bucket))

        xyb = bernoulli(rng, cfg.xyb_prob_by_bucket.for_bucket(bucket))

        fixed_code = False
        if progressive_level == 0 and bernoulli(rng, cfg.fixed_code_prob):
            fixed_code = True

        quant_table: dict[str, Any]
        if std_quant:
            quant_table = {"kind": "annex_k"}
        else:
            quant_table = {"kind": "adaptive"}

        normalized: dict[str, Any] = {
            "quality": int(base_quality),
            "progressive": bool(progressive_level > 0),
            "subsampling": subsampling,
            "quant_table": quant_table,
            "dct": None,
            "entropy": {"huffman_opt": not fixed_code, "arithmetic": False},
            "restart": None,
            "encoder_specific": {
                "progressive_level": progressive_level,
                "xyb": bool(xyb),
                "std_quant": bool(std_quant),
                "adaptive_quantization": not noadaptive_quant,
                "fixed_code": bool(fixed_code),
            },
        }

        internal: dict[str, Any] = {
            "progressive_level": progressive_level,
            "subsampling": subsampling,
            "std_quant": std_quant,
            "noadaptive_quant": noadaptive_quant,
            "xyb": xyb,
            "fixed_code": fixed_code,
            "cmd_template": ["cjpegli", "<in.png>", "<out.jpg>", "--quality", str(base_quality), "..."],
        }

        return EncoderOptions(normalized=normalized, internal=internal)

    def preview_cmd(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> list[str]:
        return self._build_cmd(Path("./input.png"), Path("./output.jpg"), options)

    def _build_cmd(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> list[str]:
        exe = self._exe or "cjpegli"
        base_quality = int(options.normalized.get("quality", 90))

        cmd: list[str] = [exe, str(input_png), str(output_jpg), "--quality", str(base_quality)]

        cmd.extend(["--progressive_level", str(int(options.internal.get("progressive_level", 0)))])

        subsampling = str(options.internal.get("subsampling", "420"))
        cmd.append(f"--chroma_subsampling={subsampling}")

        if bool(options.internal.get("std_quant", False)):
            cmd.append("--std_quant")
        if bool(options.internal.get("noadaptive_quant", False)):
            cmd.append("--noadaptive_quantization")
        if bool(options.internal.get("xyb", False)):
            cmd.append("--xyb")
        if bool(options.internal.get("fixed_code", False)):
            cmd.append("--fixed_code")

        return cmd

    def encode(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> RunResult:
        if self._exe is None:
            raise RuntimeError("cjpegli executable not found on PATH")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            rgb_png = Path(f.name)
        options.temp_paths.append(rgb_png)
        png_to_rgb_png_file(input_png, rgb_png)

        cmd = self._build_cmd(rgb_png, output_jpg, options)

        return run(cmd)
