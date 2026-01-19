"""Encoder registry and factory.

Encoders register via the @register_encoder decorator.
The factory auto-discovers modules within jpeg_variety.encoders.

Thread-safety note: the factory returns a *new* encoder instance per selection,
so plugins can stay state-free by default.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from ..config import AppConfig
from ..utils.sampling import weighted_choice

from .base import JPEGEncoder


_REGISTRY: dict[str, type[JPEGEncoder]] = {}


def register_encoder(cls: type[JPEGEncoder]) -> type[JPEGEncoder]:
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Encoder class must define a non-empty 'name' attribute")
    if name in _REGISTRY:
        raise ValueError(f"Duplicate encoder registration: {name}")
    _REGISTRY[name] = cls
    return cls


def _auto_import_plugins() -> None:
    # Import all modules in this package except base/__init__.
    pkg_name = __name__
    for m in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if m.ispkg:
            continue
        if m.name in {"base", "__init__"}:
            continue
        importlib.import_module(f"{pkg_name}.{m.name}")


@dataclass(frozen=True)
class AvailableEncoder:
    name: str
    cls: type[JPEGEncoder]
    weight: float


class EncoderFactory:
    """Weighted selection of available encoders."""

    def __init__(self, config: AppConfig):
        _auto_import_plugins()
        self._config = config
        self._available: list[AvailableEncoder] = []
        self._build_available()

    def _build_available(self) -> None:
        weights_cfg = self._config.encoder_weights.weights
        available: list[AvailableEncoder] = []

        for name, cls in _REGISTRY.items():
            enc = cls()  # instantiate to check availability
            if not enc.is_available():
                continue
            weight = float(weights_cfg.get(name, getattr(enc, "default_weight", 1.0)))
            if weight > 0:
                available.append(AvailableEncoder(name=name, cls=cls, weight=weight))

        available.sort(key=lambda a: a.name)  # stable order for reproducibility
        self._available = available

    @property
    def available_names(self) -> list[str]:
        return [a.name for a in self._available]

    def require_any(self) -> None:
        if not self._available:
            raise RuntimeError(
                "No encoders available. Ensure at least one encoder is on PATH, or adjust encoder weights."
            )

    def choose(self, rng) -> JPEGEncoder:
        """Choose an encoder class using weights, then instantiate it."""

        self.require_any()
        name_to_weight = {a.name: a.weight for a in self._available}
        chosen_name = weighted_choice(rng, name_to_weight)
        for a in self._available:
            if a.name == chosen_name:
                return a.cls()
        # Fallback (should be unreachable)
        return self._available[-1].cls()

    def get_encoder(self, name: str) -> JPEGEncoder | None:
        for a in self._available:
            if a.name == name:
                return a.cls()
        return None
