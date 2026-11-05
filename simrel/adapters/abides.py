"""
ABIDES adapter: the common execution task inside an ABIDES-Markets world.

Background configurations and the execution agent come from simbench
(simbench/abides_world.py); this module only translates between that agent's
per-slice state and the shared Observation / EpisodeRecord schema. ABIDES
cannot snapshot a running market, so the twin and each policy are separate
runs with the same seed: they are identical until the first child order.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..core import EpisodeRecord, Observation, Policy, Task, child_quantity

STABLE_WORLDS = ["A1_rmsc04", "A5_deep_mm"]  # the two cleanest configurations in the earlier pilot


class _Bridge:
    """Adapts a simrel policy to the simbench per-slice call, keeping the memory dict and the last mid."""

    def __init__(self, policy: Policy, task: Task) -> None:
        self.policy, self.task = policy, task
        self.state: dict = {}
        self.prev_mid = None
        self.arrival = None

    def __call__(self, s) -> int:
        bids, asks = s.bids or [], s.asks or []
        if not bids or not asks:
            return 0
        bid, ask = float(bids[0][0]), float(asks[0][0])
        mid = 0.5 * (bid + ask)
        self.arrival = mid if self.arrival is None else self.arrival
        # ABIDES labels volume by the resting side: its "sell" list is seller-initiated flow hitting bids
        obs = Observation(
            k=s.k, n=s.n, t=s.k * self.task.dt, bid=bid, ask=ask,
            bid_depth1=float(bids[0][1]), ask_depth1=float(asks[0][1]),
            bid_depth5=float(sum(q for _, q in bids[:5])), ask_depth5=float(sum(q for _, q in asks[:5])),
            market_volume_last=float(s.mkt_vol_last) if s.k else 0.0,
            signed_volume_last=float(s.buy_vol_last - s.sell_vol_last) if s.k else 0.0,
            mid_change_last=0.0 if self.prev_mid is None else mid - self.prev_mid,
            arrival_mid=self.arrival, total_qty=s.total, side=self.task.side,
        )
        self.prev_mid = mid
        return child_quantity(self.policy, obs, s.remaining, self.state, 1)


def run_twin_set(world_name: str, seed: int, task: Task, policies: dict[str, Optional[Policy]], pre_s: int = 300) -> dict[str, EpisodeRecord]:
    from simbench.abides_world import Task as ATask, run_episode

    out = {}
    for name, policy in {"none": None, **policies}.items():
        w0 = time.time()
        active = name != "none" and task.total_qty > 0
        atask = ATask(side=task.side, total_qty=task.total_qty if active else 0, horizon_s=task.horizon_s, n_slices=task.n_slices, post_s=task.post_s)
        rec = EpisodeRecord(world_name, seed, name, task)
        try:
            ep = run_episode(world_name, seed, atask, _Bridge(policy, task) if active else (lambda s: 0), name)
            n = task.n_slices
            s_t, s_mid, s_spr, s_req, f_qty, f_not = (np.zeros(n) for _ in range(6))
            for s in ep.slices:
                k = s["k"]
                s_t[k] = k * task.dt
                s_mid[k] = np.nan if s["mid"] is None else s["mid"]
                s_spr[k] = np.nan if s["spread"] is None else s["spread"]
                s_req[k] = s["qty"]
            if len(ep.fills):
                k = np.clip((ep.fills[:, 0] // task.dt).astype(int), 0, n - 1)
                np.add.at(f_qty, k, ep.fills[:, 2])
                np.add.at(f_not, k, ep.fills[:, 1] * ep.fills[:, 2])
            m = (ep.grid_s >= -pre_s) & (np.mod(ep.grid_s, 5) == 0)
            rec.arrival_mid = float(ep.scalars["arrival_mid"])
            rec.slice_t, rec.slice_mid, rec.slice_spread, rec.slice_req, rec.fill_qty, rec.fill_notional = s_t, s_mid, s_spr, s_req, f_qty, f_not
            rec.grid_t, rec.grid_mid, rec.grid_spread = ep.grid_s[m], ep.mid[m], ep.spread[m]
            rec.background_volume = float(ep.scalars["bg_volume"])
        except Exception as err:
            rec.failed = repr(err)[:200]
        rec.wall_s = time.time() - w0
        out[name] = rec
    return out
