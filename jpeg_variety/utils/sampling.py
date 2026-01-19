"""Weighted sampling helpers.

The core requirement is *non-uniform* selection with realistic skew, plus simple
correlations (e.g., higher quality increases likelihood of progressive + 4:4:4).

These helpers intentionally stay small and dependency-free.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, TypeVar

T = TypeVar("T")


def _normalize(weights: dict[T, float]) -> list[tuple[T, float]]:
    items = [(k, float(v)) for k, v in weights.items() if float(v) > 0.0]
    total = sum(v for _, v in items)
    if total <= 0.0:
        raise ValueError("at least one weight must be > 0")
    return [(k, v / total) for k, v in items]


def weighted_choice(rng: random.Random, weights: dict[T, float]) -> T:
    """Sample a key according to the given non-negative weights."""

    norm = _normalize(weights)
    r = rng.random()
    acc = 0.0
    for k, p in norm:
        acc += p
        if r <= acc:
            return k
    # Numeric edge-case
    return norm[-1][0]


def bernoulli(rng: random.Random, p: float) -> bool:
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return rng.random() < p


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def triangular_int(rng: random.Random, lo: int, hi: int, mode: int) -> int:
    """Triangularly distributed int for more 'human' parameter selection."""

    if hi < lo:
        lo, hi = hi, lo
    mode = max(lo, min(hi, mode))
    x = rng.triangular(lo, hi, mode)
    return int(round(x))


@dataclass(frozen=True)
class QualityBucket:
    name: str  # "low"|"mid"|"high"


def quality_bucket(q: int) -> QualityBucket:
    if q <= 40:
        return QualityBucket("low")
    if q <= 75:
        return QualityBucket("mid")
    return QualityBucket("high")


def mix_weights(*parts: dict[T, float]) -> dict[T, float]:
    """Merge multiple weight dicts by summing matching keys."""

    out: dict[T, float] = {}
    for p in parts:
        for k, v in p.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out
