"""Configuration and default sampling distributions.

This module defines:
- Default encoder selection weights.
- Global sampling probabilities used by encoder plugins.
- Optional JSON/YAML config overrides.

Defaults are chosen to approximate common JPEG production on the web:
- Baseline (sequential) is more common than progressive overall.
- 4:2:0 subsampling dominates, with 4:4:4 more frequent at higher quality.
- Standard-ish quant tables dominate, with a long tail of other tables.
- Arithmetic coding exists but is extremely rare.

The plugin implementations apply these distributions with correlations driven by
"quality buckets" (low/mid/high).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .encoder_sampling import EncoderSamplingConfig, apply_overrides
from .utils.sampling import clamp

ANNEX_K_LUMA_TABLE: tuple[tuple[int, ...], ...] = (
    # ITU-T Annex K luma quantization table in natural order.
    # Used as a realistic seed for rare custom-table perturbations.
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)

ANNEX_K_CHROMA_TABLE: tuple[tuple[int, ...], ...] = (
    # ITU-T Annex K chroma quantization table in natural order.
    # This is the standard baseline-compatible chroma seed.
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
)

ANNEX_K_LUMA_FLAT: tuple[int, ...] = tuple(v for row in ANNEX_K_LUMA_TABLE for v in row)
ANNEX_K_CHROMA_FLAT: tuple[int, ...] = tuple(v for row in ANNEX_K_CHROMA_TABLE for v in row)


@dataclass(frozen=True)
class EncoderWeights:
    """Encoder selection weights (normalized at runtime)."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "cjpeg": 0.50,
            "jpeg": 0.50,
        }
    )


@dataclass(frozen=True)
class GlobalSampling:
    """High-level sampling controls shared across encoders."""

    # Progressive vs baseline overall target (before quality correlations)
    progressive_prob: float = 0.20

    # Subsampling overall targets
    subsampling_weights: dict[str, float] = field(
        default_factory=lambda: {
            "420": 0.80,
            "444": 0.15,
            "422": 0.05,
        }
    )

    # Quantization table *kind* weights. Encoders map kinds to their knobs.
    quant_kind_weights: dict[str, float] = field(
        default_factory=lambda: {
            "annex_k": 0.50,
            "imagemagick": 0.20,
            "perceptual": 0.25,  # MS-SSIM / PSNR-HVS families
            "custom": 0.05,
        }
    )

    # When "perceptual" is chosen, pick among known tuned table IDs.
    # We favor common mid-frequency tables (2 and 4) and keep the rest as a long tail.
    perceptual_table_weights: dict[int, float] = field(
        default_factory=lambda: {2: 0.32, 4: 0.32, 5: 0.10, 6: 0.09, 7: 0.09, 8: 0.08}
    )

    # Very rare: arithmetic coding (only on encoders that support it)
    arithmetic_prob: float = 0.003  # 0.3%

    # Restart markers: occasional, biased toward off
    restart_prob: float = 0.04

    # Small chance to enable encoder-specific "artifact shaping" knobs
    artifact_knobs_prob: float = 0.08


@dataclass(frozen=True)
class AppConfig:
    encoder_weights: EncoderWeights = field(default_factory=EncoderWeights)
    sampling: GlobalSampling = field(default_factory=GlobalSampling)
    encoder_sampling: EncoderSamplingConfig = field(default_factory=EncoderSamplingConfig)


def load_config(path: Path | None) -> AppConfig:
    """Load optional config overrides.

    Supports JSON by default.
    YAML is supported if PyYAML is installed and the file extension is .yml/.yaml.

    Schema (all keys optional):
    {
      "encoder_weights": {"cjpeg": 0.75, "jpeg": 0.25},
      "sampling": {
        "progressive_prob": 0.30,
        "subsampling_weights": {"420": 0.80, "444": 0.15, "422": 0.05},
        "quant_kind_weights": {"annex_k": 0.50, ...},
        "perceptual_table_weights": {"2": 0.35, "4": 0.35, "5": 0.10, "6": 0.08, "7": 0.07, "8": 0.05},
        "arithmetic_prob": 0.003,
        "restart_prob": 0.04,
        "artifact_knobs_prob": 0.08
      },
      "encoder_sampling": {
        "cjpeg": {
          "dct_weights": {"int": 0.92, "fast": 0.05, "float": 0.03}
        },
        "jpeg": {
          "baseline_process_prob": 0.75
        }
      }
    }
    """

    if path is None:
        return AppConfig()

    path = path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    raw: dict[str, Any]
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. Install with: pip install pyyaml"
            ) from e
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f) or {}

    base = AppConfig()

    # Encoder weights
    ew = dict(base.encoder_weights.weights)
    if isinstance(raw.get("encoder_weights"), dict):
        for k, v in raw["encoder_weights"].items():
            try:
                ew[str(k)] = float(v)
            except Exception:
                continue

    # Sampling overrides
    s = base.sampling
    s_raw = raw.get("sampling") if isinstance(raw.get("sampling"), dict) else {}

    progressive_prob = float(s_raw.get("progressive_prob", s.progressive_prob))
    arithmetic_prob = float(s_raw.get("arithmetic_prob", s.arithmetic_prob))
    restart_prob = float(s_raw.get("restart_prob", s.restart_prob))
    artifact_knobs_prob = float(s_raw.get("artifact_knobs_prob", s.artifact_knobs_prob))

    subsampling_weights = dict(s.subsampling_weights)
    if isinstance(s_raw.get("subsampling_weights"), dict):
        for k, v in s_raw["subsampling_weights"].items():
            try:
                subsampling_weights[str(k)] = float(v)
            except Exception:
                continue

    quant_kind_weights = dict(s.quant_kind_weights)
    if isinstance(s_raw.get("quant_kind_weights"), dict):
        for k, v in s_raw["quant_kind_weights"].items():
            try:
                quant_kind_weights[str(k)] = float(v)
            except Exception:
                continue

    perceptual_table_weights = dict(s.perceptual_table_weights)
    if isinstance(s_raw.get("perceptual_table_weights"), dict):
        for k, v in s_raw["perceptual_table_weights"].items():
            try:
                perceptual_table_weights[int(k)] = float(v)
            except Exception:
                continue

    sampling = GlobalSampling(
        progressive_prob=clamp(progressive_prob, 0.0, 1.0),
        subsampling_weights=subsampling_weights,
        quant_kind_weights=quant_kind_weights,
        perceptual_table_weights=perceptual_table_weights,
        arithmetic_prob=clamp(arithmetic_prob, 0.0, 0.05),
        restart_prob=clamp(restart_prob, 0.0, 0.50),
        artifact_knobs_prob=clamp(artifact_knobs_prob, 0.0, 1.0),
    )

    encoder_sampling = base.encoder_sampling
    if isinstance(raw.get("encoder_sampling"), dict):
        encoder_sampling = apply_overrides(encoder_sampling, raw["encoder_sampling"])

    return AppConfig(
        encoder_weights=EncoderWeights(weights=ew),
        sampling=sampling,
        encoder_sampling=encoder_sampling,
    )
