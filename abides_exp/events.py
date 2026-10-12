"""
Market events that modify existing ABIDES components without changing their random
draws, so a treated run and its same-seed baseline stay identical until the event.

- shift_fundamental: add a deterministic offset to the fundamental from a given time
  (a news jump). Implemented on observations, so the underlying OU path and its RNG
  are untouched.
- withdraw_market_makers: market makers stop quoting (and cancel) inside a window.
- set_impact: permanent price impact. Every fill of one of OUR agents (execution
  strategies, scripted traders) shifts the fundamental by lambda bps per 10k shares of
  net flow, permanently. Value agents then carry the price to the new level. This is
  the S = F + lambda * cumulative flow of the MFG model. Background agents' trades do
  not move it. lambda = 0 reproduces the default market exactly.
"""
import types
from typing import Callable, List

import numpy as np

from abides_markets.agents.market_makers.adaptive_market_maker_agent import (
    AdaptiveMarketMakerAgent,
)
from abides_markets.oracles import SparseMeanRevertingOracle

NS = 10**9


class ShiftedOracle(SparseMeanRevertingOracle):
    """
    Sparse mean-reverting oracle plus two deterministic offsets in cents:
      offset_fn(t)                 scheduled shifts (news jumps)
      impact_per_share * cum_flow  permanent impact from our agents' net flow
    Both are applied on top of the untouched OU path, so no random draws change.
    """

    offset_fn: Callable[[int], float] = staticmethod(lambda t: 0.0)
    impact_per_share: float = 0.0   # cents of fundamental per share of net flow

    def _init_impact(self):
        self.cum_flow = 0
        self.flow_log = [(0, 0)]     # (time ns, cumulative signed shares), time-ordered

    def record_flow(self, t: int, signed_qty: int) -> None:
        """Called by our agents on every fill: +qty bought, -qty sold."""
        self.cum_flow += int(signed_qty)
        self.flow_log.append((int(t), self.cum_flow))

    def impact_offset(self, t: int) -> float:
        if not self.impact_per_share:
            return 0.0
        times = [x[0] for x in self.flow_log]
        i = int(np.searchsorted(times, t, side="right")) - 1
        return self.impact_per_share * self.flow_log[max(i, 0)][1]

    def total_offset(self, t: int) -> float:
        return self.offset_fn(t) + self.impact_offset(t)

    def observe_price(self, symbol, current_time, random_state, sigma_n=1000):
        obs = super().observe_price(symbol, current_time, random_state, sigma_n)
        t = min(current_time, self.mkt_close - 1)
        return int(round(obs + self.offset_fn(t) + self.impact_per_share * self.cum_flow))


def shift_fundamental(cfg, offset_fn: Callable[[int], float]):
    """Swap the config's oracle to ShiftedOracle in place (keeps its RNG state)."""
    oracle = cfg["custom_properties"]["oracle"]
    oracle.__class__ = ShiftedOracle
    oracle.offset_fn = offset_fn
    oracle.impact_per_share = 0.0
    oracle._init_impact()
    return oracle


def set_impact(cfg, lambda_bps_per_10k: float, symbol: str = "ABM"):
    """
    Permanent impact: lambda bps of the fundamental per 10,000 shares of our agents'
    net flow. Example: lambda = 5 and a 12,000-share sale lowers the fundamental by
    6 bps for good. Call after shift_fundamental.
    """
    oracle = cfg["custom_properties"]["oracle"]
    r_bar = oracle.symbols[symbol]["r_bar"]
    oracle.impact_per_share = r_bar * (lambda_bps_per_10k / 1e4) / 10_000
    return oracle


def report_fill(agent, order) -> None:
    """Our agents call this on every fill so the oracle can apply permanent impact."""
    oracle = getattr(getattr(agent, "kernel", None), "oracle", None)
    if isinstance(oracle, ShiftedOracle):
        q = order.quantity if order.side.is_bid() else -order.quantity
        oracle.record_flow(agent.current_time, q)


def jump_offset(mkt_open: int, at_s: float, jump_bps: float, r_bar: int = 100_000):
    """Step change of jump_bps (relative to r_bar) at mkt_open + at_s."""
    t0 = mkt_open + int(at_s * NS)
    size = r_bar * jump_bps / 1e4
    return lambda t: size if t >= t0 else 0.0


def fundamental_path(oracle, symbol: str, grid_ns: np.ndarray) -> np.ndarray:
    """True fundamental (cents) on a time grid, from the oracle's log plus any offset."""
    log = oracle.f_log[symbol]
    t = np.array([r["FundamentalTime"] for r in log], dtype="int64")
    v = np.array([r["FundamentalValue"] for r in log], dtype=float)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    idx = np.searchsorted(t, grid_ns, side="right") - 1
    out = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
    if isinstance(oracle, ShiftedOracle):
        out = out + np.array([oracle.total_offset(int(g)) for g in grid_ns])
    return out


def withdraw_market_makers(cfg, mkt_open: int, start_s: float, end_s: float) -> List[int]:
    """
    Market makers place no quotes, and cancel what they have, between start and end.
    Takes effect at each market maker's next event after `start` (they wake about every
    minute and also react to book-imbalance updates), so allow ~1 minute of lag.
    """
    t0, t1 = mkt_open + int(start_s * NS), mkt_open + int(end_s * NS)
    ids = []
    for agent in cfg["agents"]:
        if not isinstance(agent, AdaptiveMarketMakerAgent):
            continue
        ids.append(agent.id)
        orig_place, orig_recv = agent.place_orders, agent.receive_message

        def place_orders(self, mid, _orig=orig_place):
            if t0 <= self.current_time < t1:
                return
            return _orig(mid)

        agent._withdraw_sent = set()

        def receive_message(self, current_time, sender_id, message, _orig=orig_recv):
            if t0 <= current_time < t1:
                for oid, order in list(self.orders.items()):
                    if oid not in self._withdraw_sent:   # cancel each order once
                        self._withdraw_sent.add(oid)
                        self.cancel_order(order)
            return _orig(current_time, sender_id, message)

        agent.place_orders = types.MethodType(place_orders, agent)
        agent.receive_message = types.MethodType(receive_message, agent)
    return ids
