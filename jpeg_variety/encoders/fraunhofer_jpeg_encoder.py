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

from PIL import Image

from ..utils.sampling import bernoulli, triangular_int, weighted_choice
from ..utils.subprocess import RunResult, run
from . import register_encoder
from .base import EncodeContext, EncoderOptions, JPEGEncoder

# Annex K tables as a seed for rare custom -qtf generation.
ANNEXK_LUMA_FLAT = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

ANNEXK_CHROMA_FLAT = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]


def _png_to_ppm_file(png_path: Path, ppm_path: Path, *, background_rgb=(255, 255, 255)) -> None:
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


def _adjusted_progressive_prob(base_p: float, bucket: str) -> float:
    if bucket == "low":
        return max(0.05, base_p * 0.50)
    if bucket == "high":
        return min(0.60, base_p * 1.33)
    return base_p


def _adjusted_subsampling_weights(base: dict[str, float], bucket: str) -> dict[str, float]:
    if bucket == "low":
        return {"420": 0.88, "444": 0.08, "422": 0.04}
    if bucket == "high":
        return {"420": 0.68, "444": 0.27, "422": 0.05}
    return dict(base)


def _perturb_steps(rng, steps: list[int], strength: float) -> list[int]:
    out: list[int] = []
    for i, v in enumerate(steps):
        hf = i / 63.0
        local = strength * (0.35 + 0.65 * hf)
        nv = int(round(v * (1.0 + rng.uniform(-local, local))))
        out.append(max(1, min(255, nv)))
    return out


def _write_qtf_file(path: Path, luma_steps: list[int], chroma_steps: list[int]) -> None:
    # Inventory says: "64*2 integers (luma & chroma)".
    all_steps = luma_steps + chroma_steps
    txt = "\n".join(str(int(x)) for x in all_steps) + "\n"
    path.write_text(txt, encoding="utf-8")


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

        progressive = bernoulli(rng, _adjusted_progressive_prob(sampling.progressive_prob, bucket))

        # Among non-progressive runs, bias toward baseline process (web realism).
        baseline_process = False
        if not progressive:
            baseline_process = bernoulli(rng, 0.75)

        subsampling = weighted_choice(rng, _adjusted_subsampling_weights(sampling.subsampling_weights, bucket))

        # Entropy coding knobs
        arithmetic = bernoulli(rng, sampling.arithmetic_prob)
        huffman_opt = bernoulli(rng, 0.60)

        # Quantization: predefined tables 0..8, or rare custom qtf file.
        quant_kind = weighted_choice(rng, sampling.quant_kind_weights)
        quant: dict[str, Any]
        if quant_kind == "annex_k":
            quant = {"kind": "predefined", "id": 0}
        elif quant_kind == "imagemagick":
            quant = {"kind": "predefined", "id": 3}
        elif quant_kind == "perceptual":
            qid = weighted_choice(rng, {2: 0.35, 4: 0.35, 5: 0.10, 6: 0.08, 7: 0.07, 8: 0.05})
            quant = {"kind": "predefined", "id": int(qid)}
        else:
            strength = 0.20 if bucket == "high" else (0.12 if bucket == "mid" else 0.10)
            quant = {"kind": "custom", "generator": "perturbed_annexk", "strength": strength}

        # Progressive scan simplification -qv: uncommon, but exists.
        qv = progressive and bernoulli(rng, 0.20 if bucket == "low" else (0.25 if bucket == "mid" else 0.30))

        # Restart markers
        restart: dict[str, Any] | None = None
        restart_p = sampling.restart_prob
        if bucket == "low":
            restart_p = min(0.10, restart_p + 0.02)
        elif bucket == "high":
            restart_p = max(0.01, restart_p - 0.01)
        if bernoulli(rng, restart_p):
            mcus = triangular_int(rng, 4, 64, 12)
            restart = {"unit": "mcus", "value": mcus}

        # Rare artifact shaping knobs. Keep them sparse.
        dz = oz = dr = False
        if bernoulli(rng, sampling.artifact_knobs_prob):
            dz = bernoulli(rng, 0.45)
            oz = bernoulli(rng, 0.35)
            dr = bernoulli(rng, 0.25)

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
                "luma": _perturb_steps(rng, ANNEXK_LUMA_FLAT, strength),
                "chroma": _perturb_steps(rng, ANNEXK_CHROMA_FLAT, strength),
            }

        return EncoderOptions(normalized=normalized, internal=internal)

    def encode(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> RunResult:
        if self._exe is None:
            raise RuntimeError("jpeg executable not found on PATH")

        base_quality = int(options.normalized.get("quality", 75))

        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            ppm_path = Path(f.name)
        options.temp_paths.append(ppm_path)
        _png_to_ppm_file(input_png, ppm_path)

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
            _write_qtf_file(qtf_path, steps["luma"], steps["chroma"])
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
