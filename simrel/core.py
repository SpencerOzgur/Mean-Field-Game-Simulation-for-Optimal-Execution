"""
Shared schema for every simulator: the execution task, what a policy may see,
and what an episode records.

A policy is called once per decision time as

    qty = policy(observation, remaining_inventory, time_remaining, state)

and returns the number of shares to send as one market order. `state` is a
dict the policy may use as memory; it starts empty in every episode. The
observation holds only quantities known at the decision time. The harness,
not the policy, enforces the constraints: a child order is clipped to
[0, remaining], rounded down to the lot size, and the last slice always sends
the remainder, so every policy completes the same parent order and any forced
terminal liquidation is part of its reported cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Task:
    side: str = "sell"  # "sell" or "buy"
    total_qty: int = 0  # shares; 0 gives the unperturbed twin
    horizon_s: int = 600
    n_slices: int = 20
    post_s: int = 120  # observation window after the deadline

    @property
    def dt(self) -> float:
        return self.horizon_s / self.n_slices


@dataclass(frozen=True)
class Observation:
    k: int  # index of this decision time, 0-based
    n: int  # number of decision times
    t: float  # seconds since the task started
    bid: float
    ask: float
    bid_depth1: float  # shares at the best bid / ask
    ask_depth1: float
    bid_depth5: float  # shares over the best five levels
    ask_depth5: float
    market_volume_last: float  # shares traded in the market during the previous slice (0 at k = 0)
    signed_volume_last: float  # buyer-initiated minus seller-initiated shares in the previous slice
    mid_change_last: float  # mid now minus mid one slice ago, in price units (0 at k = 0)
    arrival_mid: float
    total_qty: int
    side: str

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def near_depth1(self) -> float:
        """Shares at the best level the parent order would hit."""
        return self.bid_depth1 if self.side == "sell" else self.ask_depth1

    @property
    def near_depth5(self) -> float:
        return self.bid_depth5 if self.side == "sell" else self.ask_depth5


Policy = Callable[[Observation, int, float, dict], float]


def child_quantity(policy: Optional[Policy], obs: Observation, remaining: int, state: dict, lot: int = 1) -> int:
    """Apply the harness constraints to a policy's request."""
    if remaining <= 0 or policy is None:
        return 0
    if obs.k == obs.n - 1:
        return remaining  # completion safeguard: the last slice sends everything left
    want = float(policy(obs, remaining, (obs.n - obs.k) / obs.n, state))
    if not np.isfinite(want):
        want = 0.0
    qty = int(np.floor(np.clip(want, 0, remaining) + 1e-6)) // lot * lot  # epsilon guards schedule arithmetic like 499.9999
    return qty


@dataclass
class EpisodeRecord:
    world: str
    seed: int
    policy: str
    task: Task
    arrival_mid: float = np.nan
    slice_t: np.ndarray = None  # decision times, seconds since task start
    slice_mid: np.ndarray = None  # mid observed just before each decision
    slice_spread: np.ndarray = None
    slice_req: np.ndarray = None  # shares requested per slice
    fill_qty: np.ndarray = None  # shares filled per slice
    fill_notional: np.ndarray = None  # sum of price * shares per slice
    grid_t: np.ndarray = None  # common time grid, seconds since task start (includes pre and post)
    grid_mid: np.ndarray = None
    grid_spread: np.ndarray = None
    background_volume: float = np.nan  # market volume in the task window excluding our fills
    wall_s: float = np.nan
    failed: Optional[str] = None  # reason, if the episode could not be completed
    extra: dict = field(default_factory=dict)

    @property
    def filled(self) -> float:
        return float(np.sum(self.fill_qty)) if self.fill_qty is not None else 0.0
