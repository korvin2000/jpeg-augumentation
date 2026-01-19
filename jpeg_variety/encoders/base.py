"""Encoder interface and shared data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import GlobalSampling
from ..utils.subprocess import RunResult


@dataclass(frozen=True)
class EncodeContext:
    """Context available during option sampling.

    rel_path is the path of the input PNG relative to src_root.
    """

    src_root: Path
    dst_root: Path
    rel_path: Path
    quality_bucket: str  # "low"|"mid"|"high"
    per_file_seed: int
    sampling: GlobalSampling


@dataclass
class EncoderOptions:
    """Holds normalized options and encoder-internal fields.

    normalized must stay JSON-serializable.
    internal can include Paths and non-serializable values for encoder runtime.
    """

    normalized: dict[str, Any]
    internal: dict[str, Any] = field(default_factory=dict)
    temp_paths: list[Path] = field(default_factory=list)


class JPEGEncoder(abc.ABC):
    """Abstract base class for encoder plugins."""

    name: str
    default_weight: float

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if required executable(s) are present."""

    @abc.abstractmethod
    def sample_options(self, base_quality: int, rng, context: EncodeContext) -> EncoderOptions:
        """Sample encoder-specific options."""

    @abc.abstractmethod
    def encode(self, input_png: Path, output_jpg: Path, options: EncoderOptions) -> RunResult:
        """Perform encoding."""

    def cleanup(self, options: EncoderOptions) -> None:
        """Remove any temp files."""

        for p in options.temp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
