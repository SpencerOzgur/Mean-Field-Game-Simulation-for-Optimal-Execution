"""
Execution policies shared by every world.

A policy maps a SliceState to a child-order size. All policies are plain
picklable objects so they can be shipped to worker processes. The execution
agent forces the full remainder in the last slice, so every policy completes
the same parent order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def exp_weights(b: float, n: int) -> np.ndarray:
    """Schedule family w_k ∝ exp(-b k / n). b = 0 is TWAP, b > 0 front-loads."""
    w = np.exp(-b * np.arange(n) / n)
    return w / w.sum()


@dataclass(frozen=True)
class Schedule:
    """Open-loop schedule: trade so cumulative fills track cumsum(weights)."""

    weights: tuple

    def __call__(self, s) -> int:
        target_done = s.total * float(np.sum(self.weights[: s.k + 1]))
        return target_done - (s.total - s.remaining)


@dataclass(frozen=True)
class Immediate:
    def __call__(self, s) -> int:
        return s.remaining


@dataclass(frozen=True)
class POV:
    """Trade a fixed fraction of the market volume seen over the last slice."""

    rate: float

    def __call__(self, s) -> int:
        return self.rate * s.mkt_vol_last


@dataclass(frozen=True)
class Urgency:
    """Deadline-paced rule that speeds up when the book is tight and slows when wide."""

    tight_ticks: int = 2
    wide_ticks: int = 6
    fast: float = 2.0
    slow: float = 0.5

    def __call__(self, s) -> int:
        base = s.remaining / (s.n - s.k)
        if s.spread is None:
            return base
        if s.spread <= self.tight_ticks:
            return self.fast * base
        if s.spread >= self.wide_ticks:
            return self.slow * base
        return base


def twap(n: int) -> Schedule:
    return Schedule(tuple(exp_weights(0.0, n)))


def exp_schedule(b: float, n: int) -> Schedule:
    return Schedule(tuple(exp_weights(b, n)))
