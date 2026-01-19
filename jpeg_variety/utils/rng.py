"""RNG utilities.

Design goals
- Optional global seed for deterministic runs.
- Per-file deterministic RNG derived from (global seed, relative path).
  This ensures stable encoder/options per image across re-runs.

We intentionally avoid Python's built-in hash() (salted per process). Instead we
use BLAKE2b to build a stable 64-bit integer.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeedContext:
    """Holds the effective run seed, if any."""

    run_seed: int | None


def make_run_seed(user_seed: int | None) -> SeedContext:
    """Create a SeedContext.

    If user_seed is None, the run is intentionally non-deterministic.
    """

    if user_seed is None:
        return SeedContext(run_seed=None)
    if user_seed < 0:
        raise ValueError("seed must be >= 0")
    return SeedContext(run_seed=int(user_seed))


def _stable_u64(data: bytes) -> int:
    # 8 bytes = 64-bit. Use a fixed personalization to avoid cross-tool collisions.
    h = hashlib.blake2b(data, digest_size=8, person=b"jpegvar1")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def per_file_seed(seed_ctx: SeedContext, rel_path: Path) -> int:
    """Compute a deterministic 64-bit seed for an item.

    If run_seed is None, uses OS entropy to create an initial seed and still
    returns a deterministic seed *per invocation*, not across invocations.
    """

    norm = rel_path.as_posix().encode("utf-8", errors="surrogatepass")

    if seed_ctx.run_seed is None:
        # Derive a per-run random base seed (not reproducible across runs).
        base = int.from_bytes(os.urandom(8), "little")
    else:
        base = seed_ctx.run_seed

    mixed = base.to_bytes(8, "little", signed=False) + b"\x00" + norm
    return _stable_u64(mixed)


def rng_for_file(seed_ctx: SeedContext, rel_path: Path) -> random.Random:
    """Create a per-file RNG."""

    return random.Random(per_file_seed(seed_ctx, rel_path))
