"""
The strategy added for the reactive family (design addendum 11): P0r, a randomised TWAP.

P0r_twap_random draws, before its first child, which slices to skip and how to weight the others, then tracks that
schedule exactly as the fixed schedules of library.py do. Quantity not sent in a skipped slice is caught up by the
later ones and the harness completes the order at the last slice. Children can only be sent at slice starts (the
engine places them there), so the randomisation acts on sizes and on skipped slices, not on timing inside a slice.

The random plan depends only on (world, seed): it comes from default_rng([crc32(world), seed, P0R_TAG]) and never
touches a market stream. A policy is not told the world or the seed, so reactive_candidate_set takes them.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .library import _schedule_target, candidate_set

P0R_TAG = 0x503072  # "P0r"


@dataclass(frozen=True)
class RandomTWAP:
    key: tuple = (0, 0)  # (crc32 of the world name, seed)
    skip_prob: float = 0.3  # each slice but the last is skipped with this probability
    jitter: float = 0.5  # weights of the other slices are 1 + jitter x uniform(-1, 1)

    def plan(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(keep mask, cumulative schedule) for n slices."""
        rng = np.random.default_rng([int(self.key[0]), int(self.key[1]), P0R_TAG])
        keep = rng.random(n) >= self.skip_prob
        keep[-1] = True
        w = keep * (1.0 + self.jitter * rng.uniform(-1.0, 1.0, n))
        return keep, np.cumsum(w / w.sum())

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        if "cum" not in state:
            state["keep"], state["cum"] = self.plan(obs.n)
        if not state["keep"][obs.k]:
            return 0.0
        return _schedule_target(obs, remaining, state["cum"])


def reactive_candidate_set(size_frac: float, world: str = "", seed: int = 0) -> dict:
    """candidate_set(size_frac) plus P0r_twap_random; used for the reactive family only."""
    out = dict(candidate_set(size_frac))
    out["P0r_twap_random"] = RandomTWAP((zlib.crc32(world.encode()), int(seed)))
    return out
