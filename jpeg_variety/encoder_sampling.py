"""Encoder sampling configuration and adjustment helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any


@dataclass(frozen=True)
class BucketedValues:
    low: float
    mid: float
    high: float

    def for_bucket(self, bucket: str) -> float:
        if bucket == "low":
            return self.low
        if bucket == "high":
            return self.high
        return self.mid


@dataclass(frozen=True)
class BucketedWeightsOverrides:
    low: dict[str, float] = field(default_factory=dict)
    mid: dict[str, float] = field(default_factory=dict)
    high: dict[str, float] = field(default_factory=dict)

    def for_bucket(self, base: dict[str, float], bucket: str) -> dict[str, float]:
        override = {"low": self.low, "mid": self.mid, "high": self.high}.get(bucket, {})
        if not override:
            return dict(base)
        return dict(override)


@dataclass(frozen=True)
class WeightAdjustments:
    delta: dict[str, float] = field(default_factory=dict)
    min_floor: dict[str, float] = field(default_factory=dict)

    def apply(self, base: dict[str, float]) -> dict[str, float]:
        out = dict(base)
        for key, value in self.delta.items():
            out[key] = out.get(key, 0.0) + value
        for key, value in self.min_floor.items():
            out[key] = max(value, out.get(key, 0.0))
        return out


@dataclass(frozen=True)
class ProgressiveProbAdjustments:
    low_multiplier: float = 0.50
    high_multiplier: float = 1.33
    low_min: float = 0.05
    high_max: float = 0.60

    def apply(self, base_prob: float, bucket: str) -> float:
        if bucket == "low":
            return max(self.low_min, base_prob * self.low_multiplier)
        if bucket == "high":
            return min(self.high_max, base_prob * self.high_multiplier)
        return base_prob


@dataclass(frozen=True)
class RestartProbAdjustments:
    low_delta: float = 0.02
    high_delta: float = -0.01
    low_cap: float = 0.10
    high_floor: float = 0.01

    def apply(self, base_prob: float, bucket: str) -> float:
        if bucket == "low":
            return min(self.low_cap, base_prob + self.low_delta)
        if bucket == "high":
            return max(self.high_floor, base_prob + self.high_delta)
        return base_prob


@dataclass(frozen=True)
class TriangularRange:
    low: int
    high: int
    mode: int


@dataclass(frozen=True)
class CjpegSamplingConfig:
    # Bias progressive by quality bucket: low-quality images trend toward baseline,
    # while high-quality images more often use progressive scans.
    progressive_adjustments: ProgressiveProbAdjustments = field(default_factory=ProgressiveProbAdjustments)
    subsampling_overrides: BucketedWeightsOverrides = field(
        default_factory=lambda: BucketedWeightsOverrides(
            low={"420": 0.88, "444": 0.08, "422": 0.04},
            high={"420": 0.68, "444": 0.27, "422": 0.05},
        )
    )
    dct_weights: dict[str, float] = field(default_factory=lambda: {"int": 0.92, "fast": 0.05, "float": 0.03})
    quant_kind_adjustments: dict[str, WeightAdjustments] = field(
        default_factory=lambda: {
            "low": WeightAdjustments(delta={"annex_k": 0.08, "custom": -0.02}, min_floor={"custom": 0.01}),
            "high": WeightAdjustments(
                delta={"perceptual": 0.06, "custom": 0.01, "annex_k": -0.05},
                min_floor={"annex_k": 0.10},
            ),
        }
    )
    custom_quant_strength_by_bucket: BucketedValues = field(
        default_factory=lambda: BucketedValues(low=0.10, mid=0.12, high=0.20)
    )
    # DC scan optimization: keep default (1) most often, let others be rare.
    dc_scan_opt_weights_mid: dict[int, float] = field(default_factory=lambda: {1: 0.80, 2: 0.12, 0: 0.08})
    dc_scan_opt_weights_high: dict[int, float] = field(default_factory=lambda: {1: 0.68, 2: 0.20, 0: 0.12})
    fastcrush_prob_by_bucket: BucketedValues = field(default_factory=lambda: BucketedValues(0.08, 0.12, 0.18))
    trellis_disable_prob_by_bucket: BucketedValues = field(default_factory=lambda: BucketedValues(0.04, 0.02, 0.01))
    trellis_disable_weights: dict[str, float] = field(default_factory=lambda: {"notrellis": 0.40, "notrellis_dc": 0.60})
    tune_mode_weights: dict[str, float] = field(
        default_factory=lambda: {
            "tune_hvs_psnr": 0.62,
            "tune_psnr": 0.25,
            "tune_ssim": 0.08,
            "tune_ms_ssim": 0.05,
        }
    )
    restart_adjustments: RestartProbAdjustments = field(default_factory=RestartProbAdjustments)
    restart_blocks_prob: float = 0.20
    restart_blocks_range: TriangularRange = TriangularRange(8, 128, 16)
    restart_rows_range: TriangularRange = TriangularRange(1, 32, 4)
    quant_baseline_prob_nonprogressive: float = 0.35
    quant_baseline_prob_custom: float = 0.70


@dataclass(frozen=True)
class FraunhoferSamplingConfig:
    # Progressive bias by quality bucket.
    progressive_adjustments: ProgressiveProbAdjustments = field(default_factory=ProgressiveProbAdjustments)
    subsampling_overrides: BucketedWeightsOverrides = field(
        default_factory=lambda: BucketedWeightsOverrides(
            low={"420": 0.88, "444": 0.08, "422": 0.04},
            high={"420": 0.68, "444": 0.27, "422": 0.05},
        )
    )
    baseline_process_prob: float = 0.75
    huffman_opt_prob: float = 0.60
    qv_prob_by_bucket: BucketedValues = field(default_factory=lambda: BucketedValues(0.20, 0.25, 0.30))
    custom_quant_strength_by_bucket: BucketedValues = field(
        default_factory=lambda: BucketedValues(low=0.10, mid=0.12, high=0.20)
    )
    restart_adjustments: RestartProbAdjustments = field(default_factory=RestartProbAdjustments)
    restart_mcus_range: TriangularRange = TriangularRange(4, 64, 12)
    # When artifact knobs are enabled, dz is most common, oz mid, dr rare.
    artifact_knob_probs: dict[str, float] = field(default_factory=lambda: {"dz": 0.45, "oz": 0.35, "dr": 0.25})


@dataclass(frozen=True)
class EncoderSamplingConfig:
    cjpeg: CjpegSamplingConfig = field(default_factory=CjpegSamplingConfig)
    jpeg: FraunhoferSamplingConfig = field(default_factory=FraunhoferSamplingConfig)


def apply_overrides(base: Any, overrides: dict[str, Any]) -> Any:
    """Return a copy of base with overrides applied for matching dataclass fields."""

    if not (is_dataclass(base) and isinstance(overrides, dict)):
        return base

    updates: dict[str, Any] = {}
    for f in fields(base):
        if f.name not in overrides:
            continue
        raw_value = overrides[f.name]
        current = getattr(base, f.name)
        if is_dataclass(current) and isinstance(raw_value, dict):
            updates[f.name] = apply_overrides(current, raw_value)
        elif isinstance(current, dict) and isinstance(raw_value, dict):
            merged = dict(current)
            for key, value in raw_value.items():
                if isinstance(value, (int, float)):
                    merged[str(key)] = float(value)
                else:
                    merged[str(key)] = value
            updates[f.name] = merged
        else:
            if isinstance(current, bool):
                updates[f.name] = bool(raw_value)
            elif isinstance(current, int):
                updates[f.name] = int(raw_value)
            elif isinstance(current, float):
                updates[f.name] = float(raw_value)
            else:
                updates[f.name] = raw_value

    return replace(base, **updates)
