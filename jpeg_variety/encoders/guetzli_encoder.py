"""Guetzli JPEG encoder plugin.

Guetzli exposes a small set of knobs that impact encoding:
- quality target (--quality)
- memory limits (--memlimit / --nomemlimit)

Other flags (verbose) are informational and not used by default.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..utils.image import png_to_rgb_png_file
from ..utils.sampling import bernoulli, triangular_int
from ..utils.subprocess import RunResult, run
from . import register_encoder
from .base import EncodeContext, EncoderOptions, JPEGEncoder


@register_encoder
class GuetzliEncoder(JPEGEncoder):
    name = "guetzli"
    default_weight = 0.10

    def __init__(self) -> None:
        self._exe = shutil.which("guetzli")

    def is_available(self) -> bool:
        return self._exe is not None

    def sample_options(self, base_quality: int, rng, context: EncodeContext) -> EncoderOptions:
        cfg = context.encoder_sampling.guetzli

        nomemlimit = bernoulli(rng, cfg.nomemlimit_prob)
        memlimit_mb = None
        if not nomemlimit:
            memlimit_mb = triangular_int(
                rng, cfg.memlimit_mb_range.low, cfg.memlimit_mb_range.high, cfg.memlimit_mb_range.mode
            )

        normalized: dict[str, Any] = {
            "quality": int(base_quality),
            "progressive": None,
            "subsampling": None,
            "quant_table": None,
            "dct": None,
            "entropy": {"huffman_opt": None, "arithmetic": None},
            "restart": None,
            "encoder_specific": {
                "nomemlimit": bool(nomemlimit),
                "memlimit_mb": memlimit_mb,
            },
        }

        internal = {
            "nomemlimit": nomemlimit,
            "memlimit_mb": memlimit_mb,
            "cmd_template": ["guetzli", "--quality", str(base_quality), "<in.png>", "<out.jpg>"],
        }

        return EncoderOptions(normalized=normalized, internal=internal)

    def preview_cmd(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> list[str]:
        return self._build_cmd(Path("./input.png"), Path("./output.jpg"), options)

    def _build_cmd(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> list[str]:
        exe = self._exe or "guetzli"
        base_quality = int(options.normalized.get("quality", 90))

        cmd: list[str] = [exe, "--quality", str(base_quality)]
        if bool(options.internal.get("nomemlimit", False)):
            cmd.append("--nomemlimit")
        else:
            memlimit_mb = options.internal.get("memlimit_mb")
            if memlimit_mb:
                cmd.extend(["--memlimit", str(int(memlimit_mb))])
        cmd.extend([str(input_png), str(output_jpg)])
        return cmd

    def encode(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> RunResult:
        if self._exe is None:
            raise RuntimeError("guetzli executable not found on PATH")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            rgb_png = Path(f.name)
        options.temp_paths.append(rgb_png)
        png_to_rgb_png_file(input_png, rgb_png)

        cmd = self._build_cmd(rgb_png, output_jpg, options)

        return run(cmd)
