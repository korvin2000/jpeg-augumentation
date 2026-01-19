"""Fraunhofer/Thomas Richter JPEG reference encoder (`jpeg` executable) plugin.

We constrain options to standard 8-bit JPEG production knobs that materially
change the output bitstream/artifacts:
- baseline vs (extended) sequential vs progressive
- subsampling factors (-s)
- quantization table selection (-qt) and rare custom table file (-qtf)
- Huffman optimization (-h) and very rare arithmetic coding (-a)
- restart markers (-z)
- optional artifact shaping knobs (-dz, -oz, -dr) (rare)

Implementation detail:
The tool consumes PPM/PGM inputs; we convert PNG->PPM using Pillow.

Option inventory source: `jpeg` help output (provided with this project).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..config import ANNEX_K_CHROMA_FLAT, ANNEX_K_LUMA_FLAT
from ..utils.image import png_to_ppm_file
from ..utils.quant_tables import perturb_steps, write_qtf_file
from ..utils.sampling import bernoulli, triangular_int, weighted_choice
from ..utils.subprocess import RunResult, run
from . import register_encoder
from .base import EncodeContext, EncoderOptions, JPEGEncoder


@register_encoder
class FraunhoferJPEGEncoder(JPEGEncoder):
    name = "jpeg"
    default_weight = 0.25

    def __init__(self) -> None:
        self._exe = shutil.which("jpeg")

    def is_available(self) -> bool:
        return self._exe is not None

    def sample_options(self, base_quality: int, rng, context: EncodeContext) -> EncoderOptions:
        bucket = context.quality_bucket
        sampling = context.sampling
        cfg = context.encoder_sampling.jpeg

        progressive = bernoulli(rng, cfg.progressive_adjustments.apply(sampling.progressive_prob, bucket))

        # Among non-progressive runs, bias toward baseline process (web realism).
        baseline_process = False
        if not progressive:
            baseline_process = bernoulli(rng, cfg.baseline_process_prob)

        subsampling = weighted_choice(rng, cfg.subsampling_overrides.for_bucket(sampling.subsampling_weights, bucket))

        # Entropy coding knobs
        arithmetic = bernoulli(rng, sampling.arithmetic_prob)
        huffman_opt = bernoulli(rng, cfg.huffman_opt_prob)

        # Quantization: predefined tables 0..8, or rare custom qtf file.
        quant_kind = weighted_choice(rng, sampling.quant_kind_weights)
        quant: dict[str, Any]
        if quant_kind == "annex_k":
            quant = {"kind": "predefined", "id": 0}
        elif quant_kind == "imagemagick":
            quant = {"kind": "predefined", "id": 3}
        elif quant_kind == "perceptual":
            qid = weighted_choice(rng, sampling.perceptual_table_weights)
            quant = {"kind": "predefined", "id": int(qid)}
        else:
            strength = cfg.custom_quant_strength_by_bucket.for_bucket(bucket)
            quant = {"kind": "custom", "generator": "perturbed_annexk", "strength": strength}

        # Progressive scan simplification -qv: uncommon, but exists.
        qv = progressive and bernoulli(rng, cfg.qv_prob_by_bucket.for_bucket(bucket))

        # Restart markers
        restart: dict[str, Any] | None = None
        restart_p = cfg.restart_adjustments.apply(sampling.restart_prob, bucket)
        if bernoulli(rng, restart_p):
            mcus = triangular_int(
                rng, cfg.restart_mcus_range.low, cfg.restart_mcus_range.high, cfg.restart_mcus_range.mode
            )
            restart = {"unit": "mcus", "value": mcus}

        # Rare artifact shaping knobs. Keep them sparse.
        dz = oz = dr = False
        if bernoulli(rng, sampling.artifact_knobs_prob):
            dz = bernoulli(rng, cfg.artifact_knob_probs.get("dz", 0.45))
            oz = bernoulli(rng, cfg.artifact_knob_probs.get("oz", 0.35))
            dr = bernoulli(rng, cfg.artifact_knob_probs.get("dr", 0.25))

        normalized: dict[str, Any] = {
            "quality": int(base_quality),
            "progressive": bool(progressive),
            "subsampling": subsampling,
            "quant_table": quant,
            "dct": None,
            "entropy": {"huffman_opt": bool(huffman_opt), "arithmetic": bool(arithmetic)},
            "restart": restart,
            "encoder_specific": {
                "baseline_process": bool(baseline_process),
                "qv": bool(qv),
                "dz": bool(dz),
                "oz": bool(oz),
                "dr": bool(dr),
            },
        }

        internal: dict[str, Any] = {
            "progressive": progressive,
            "baseline_process": baseline_process,
            "qv": qv,
            "subsampling": subsampling,
            "huffman_opt": huffman_opt,
            "arithmetic": arithmetic,
            "quant": quant,
            "restart": restart,
            "dz": dz,
            "oz": oz,
            "dr": dr,
            "cmd_template": ["jpeg", "-q", str(base_quality), "...", "<in.ppm>", "<out.jpg>"],
        }

        if quant.get("kind") == "custom":
            strength = float(quant.get("strength", 0.12))
            internal["custom_steps"] = {
                "luma": perturb_steps(rng, list(ANNEX_K_LUMA_FLAT), strength),
                "chroma": perturb_steps(rng, list(ANNEX_K_CHROMA_FLAT), strength),
            }

        return EncoderOptions(normalized=normalized, internal=internal)

    def encode(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> RunResult:
        if self._exe is None:
            raise RuntimeError("jpeg executable not found on PATH")

        base_quality = int(options.normalized.get("quality", 75))

        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            ppm_path = Path(f.name)
        options.temp_paths.append(ppm_path)
        png_to_ppm_file(input_png, ppm_path)

        cmd: list[str] = [self._exe, "-q", str(base_quality)]

        # Process selection
        if bool(options.internal.get("progressive", False)):
            cmd.append("-v")
            if bool(options.internal.get("qv", False)):
                cmd.append("-qv")
        else:
            if bool(options.internal.get("baseline_process", False)):
                cmd.append("-bl")

        # Entropy coding
        if bool(options.internal.get("huffman_opt", False)):
            cmd.append("-h")
        if bool(options.internal.get("arithmetic", False)):
            cmd.append("-a")

        # Subsampling factors
        subsampling = str(options.internal.get("subsampling", "444"))
        # From inventory: default 444 is 1x1,1x1,1x1; 420 often used is 1x1,2x2,2x2.
        # For 422, we use a plausible mapping: 1x1,2x1,2x1.
        s_map = {
            "444": "1x1,1x1,1x1",
            "420": "1x1,2x2,2x2",
            "422": "1x1,2x1,2x1",
        }
        cmd.extend(["-s", s_map.get(subsampling, "1x1,2x2,2x2")])

        # Quantization
        quant = options.internal.get("quant", {})
        if isinstance(quant, dict) and quant.get("kind") == "predefined":
            cmd.extend(["-qt", str(int(quant.get("id", 0)))])
        elif isinstance(quant, dict) and quant.get("kind") == "custom":
            steps = options.internal.get("custom_steps")
            if not isinstance(steps, dict) or "luma" not in steps or "chroma" not in steps:
                raise RuntimeError("custom quant selected but custom_steps missing")
            with tempfile.NamedTemporaryFile("w", suffix=".qtf", delete=False, encoding="utf-8") as f:
                qtf_path = Path(f.name)
            options.temp_paths.append(qtf_path)
            write_qtf_file(qtf_path, steps["luma"], steps["chroma"])
            cmd.extend(["-qtf", str(qtf_path)])

        # Restart markers
        restart = options.internal.get("restart")
        if isinstance(restart, dict) and restart.get("value") is not None:
            cmd.extend(["-z", str(int(restart["value"]))])

        # Rare artifact shaping knobs
        if bool(options.internal.get("dz", False)):
            cmd.append("-dz")
        if bool(options.internal.get("oz", False)):
            cmd.append("-oz")
        if bool(options.internal.get("dr", False)):
            cmd.append("-dr")

        cmd.extend([str(ppm_path), str(output_jpg)])

        return run(cmd)
