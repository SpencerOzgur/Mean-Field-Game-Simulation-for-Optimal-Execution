"""
Execution policies P0-P7, all with the signature

    qty = policy(observation, remaining_inventory, time_remaining, state)

Fixed policies (P0-P3) depend only on the clock and the inventory. Adaptive
policies (P4-P7) also read the observation; anything they remember about the
past lives in `state`, which starts empty each episode. No policy can see the
future because the observation contains none of it (see tests/test_policies.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _schedule_target(obs, remaining: int, weights_cum: np.ndarray) -> float:
    """Shares needed now so cumulative fills track a pre-set schedule."""
    done = obs.total_qty - remaining
    return obs.total_qty * float(weights_cum[obs.k]) - done


@dataclass(frozen=True)
class ExpSchedule:
    """P0 (b = 0, TWAP), P2 (b > 0, front-loaded), P3 (b < 0, back-loaded): w_k proportional to exp(-b k / n)."""

    b: float = 0.0

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        if "cum" not in state:
            w = np.exp(-self.b * np.arange(obs.n) / obs.n)
            state["cum"] = np.cumsum(w / w.sum())
        return _schedule_target(obs, remaining, state["cum"])


@dataclass(frozen=True)
class Immediate:
    """P1: sell as fast as allowed, at most depth_frac of the visible top-five depth per decision."""

    depth_frac: float = 1.0

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        return min(remaining, self.depth_frac * obs.near_depth5)


@dataclass(frozen=True)
class POV:
    """P4: a fixed fraction of the market volume seen in the previous slice; TWAP-sized first child."""

    rate: float = 0.1

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        if obs.k == 0:
            return remaining / obs.n
        return self.rate * obs.market_volume_last


@dataclass(frozen=True)
class LiquidityAware:
    """P5: deadline-paced, scaled up when the spread is tight and depth ample relative to what this episode has shown so far."""

    lo: float = 0.25
    hi: float = 3.0

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        spreads = state.setdefault("spreads", [])
        depths = state.setdefault("depths", [])
        spreads.append(max(obs.spread, 1e-12))
        depths.append(max(obs.near_depth5, 1.0))
        m = (np.median(spreads) / spreads[-1]) * (depths[-1] / np.mean(depths))
        return float(np.clip(m, self.lo, self.hi)) * remaining / (obs.n - obs.k)


@dataclass(frozen=True)
class SignalAware:
    """P6: deadline-paced, tilted by the previous slice's order-flow imbalance and return.

    signal > 0 means the last slice moved against the parent order (for a sell:
    sellers dominated and the price fell). gamma > 0 trades faster after an
    adverse move (momentum view); gamma < 0 waits (reversal view).
    """

    gamma: float = 1.0
    lo: float = 0.25
    hi: float = 3.0

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        sgn = -1.0 if obs.side == "sell" else 1.0
        imb = -sgn * obs.signed_volume_last / obs.market_volume_last if obs.market_volume_last > 0 else 0.0
        ret = -sgn * np.sign(obs.mid_change_last)
        signal = 0.5 * (imb + ret)
        return float(np.clip(1.0 + self.gamma * signal, self.lo, self.hi)) * remaining / (obs.n - obs.k)


@dataclass(frozen=True)
class RiskAware:
    """P7: Almgren-Chriss feedback rule with an urgency estimated online.

    Trades the fraction 1 - sinh(kappa (m - 1)) / sinh(kappa m) of the remaining
    inventory, m being the number of decisions left. kappa rises with realised
    volatility (risk of waiting) and falls with the spread (cost of trading):
    kappa = risk_aversion * sd(mid changes so far) / spread, per slice.
    """

    risk_aversion: float = 1.0
    kappa_max: float = 3.0

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        moves = state.setdefault("moves", [])
        if obs.k > 0:
            moves.append(obs.mid_change_last)
        m = obs.n - obs.k
        sd = float(np.std(moves)) if len(moves) >= 2 else 0.0
        kappa = min(self.risk_aversion * sd / max(obs.spread, 1e-12), self.kappa_max)
        if kappa < 1e-6:
            return remaining / m
        return remaining * (1.0 - np.sinh(kappa * (m - 1)) / np.sinh(kappa * m))


def candidate_set(size_frac: float) -> dict:
    """The bounded policy class used for selection. POV rates are set relative to the parent size."""
    return {
        "P0_twap": ExpSchedule(0.0),
        "P1_immediate": Immediate(1.0),
        "P2_front_b2": ExpSchedule(2.0),
        "P2_front_b4": ExpSchedule(4.0),
        "P3_back_b-2": ExpSchedule(-2.0),
        "P4_pov_1x": POV(size_frac / (1.0 - size_frac)),
        "P4_pov_2x": POV(2.0 * size_frac / (1.0 - size_frac)),
        "P5_liquidity": LiquidityAware(),
        "P6_signal_momentum": SignalAware(1.0),
        "P6_signal_reversal": SignalAware(-1.0),
        "P7_risk_0.5": RiskAware(0.5),
        "P7_risk_2": RiskAware(2.0),
    }
