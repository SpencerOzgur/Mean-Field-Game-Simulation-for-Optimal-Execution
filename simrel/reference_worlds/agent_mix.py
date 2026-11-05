"""
Reference world R2: a mix of behavioural traders with a hidden regime.

Nothing here is a queue rule or a self-exciting kernel. Order flow comes from
four kinds of trader whose activity depends on prices and on a hidden state:

    noise traders        limit orders, market orders and cancels at constant rates
    fundamentalists      market orders towards a latent fundamental value F_t, with an
                         intensity that grows faster than linearly in the mispricing
    trend followers      market orders in the direction of the recent return; they are
                         several times more active in the hidden "trending" regime
    liquidity providers  cancel resting quotes on a side that has just absorbed
                         one-sided market volume (they step back from toxic flow)

F_t is a random walk and the regime is a two-state Markov chain; both are
exogenous and unaffected by any order. A large sell pushes the price below F
(fundamentalists buy back: resilience), produces a negative return (trend
followers sell: amplification), and makes bids withdraw (impact that is convex
in how fast one trades). Which force wins depends on the regime, which no
market participant observes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lob import Book
from .base import CB, CS, LB, LS, MB, MS, ThinnedWorld, book_state, geometric

LIM_B, LIM_A, CAN_B, CAN_A, NOISE_MB, NOISE_MS, FUND_MB, FUND_MS, TREND_MB, TREND_MS = range(10)


@dataclass(frozen=True)
class AMParams:
    limit: float = 1.4  # noise limit orders per second per side
    cancel: float = 0.7  # cancels per second per q_ref shares of depth
    q_ref: float = 8000.0
    noise_market: float = 0.08
    fund: float = 0.10  # fundamentalist market orders per second at a one-tick mispricing
    fund_power: float = 1.5
    trend: float = 0.05  # trend-follower market orders per second per tick of recent return, calm regime
    trend_boost: float = 5.0  # multiplier in the trending regime
    regime_switch: float = 1.0 / 400.0  # regime changes per second
    return_halflife: float = 30.0  # seconds, for the trend signal
    flow_halflife: float = 20.0  # seconds, for the liquidity providers' toxicity signal
    withdraw: float = 2.0  # extra cancel intensity per v_ref shares of adverse signed flow
    v_ref: float = 2000.0
    fund_vol: float = 0.25  # ticks per sqrt(second)
    inside_prob: float = 0.4
    level_p: float = 0.35
    limit_size_p: float = 0.45
    market_size_p: float = 0.35
    init_levels: int = 10
    init_qty: int = 800


class AgentMix(ThinnedWorld):
    name = "agent_mix"

    def __init__(self, params: AMParams, seed: int) -> None:
        p = params
        lam_max = [2 * p.limit, 2 * p.limit, 8 * p.cancel, 8 * p.cancel, 2 * p.noise_market, 2 * p.noise_market,
                   3.0, 3.0, 3.0, 3.0]
        super().__init__(10, np.maximum(lam_max, 0.5), seed, stream=2)
        self.p = p
        self.exo = np.random.default_rng([seed, 2, 99])  # fundamental and regime: never touched by orders
        self.exo_t = 0.0
        self.F = float(self.p0)
        self.trending = False
        self.sig_t = 0.0
        self.ret = 0.0  # exponentially weighted recent mid change, in ticks
        self.flow = 0.0  # exponentially weighted signed market volume (buys positive)
        self.last_mid = float(self.p0)

    def init_book(self, book: Book) -> None:
        self.seed_book(book, self.p.init_levels, self.p.init_qty)

    # ---- exogenous state and signals
    def tick(self, book: Book, t: float) -> None:
        p = self.p
        while self.exo_t + 1.0 <= t:  # one-second steps, driven only by the exogenous stream
            self.exo_t += 1.0
            self.F += p.fund_vol * self.exo.standard_normal()
            if self.exo.random() < p.regime_switch:
                self.trending = not self.trending
        dt = t - self.sig_t
        if dt > 0:
            mid = self.mid(book, self.last_mid)
            self.ret = self.ret * 0.5 ** (dt / p.return_halflife) + (mid - self.last_mid)
            self.flow *= 0.5 ** (dt / p.flow_halflife)
            self.last_mid, self.sig_t = mid, t

    def _note_trade(self, signed_qty: float) -> None:
        self.flow += signed_qty

    def intensity(self, i: int, book: Book, t: float) -> float:
        p = self.p
        bb, ba = book.best_bid(), book.best_ask()
        if i in (LIM_B, LIM_A):
            return p.limit
        if i in (CAN_B, CAN_A):
            side = "bid" if i == CAN_B else "ask"
            adverse = max(-self.flow, 0.0) if i == CAN_B else max(self.flow, 0.0)  # bids flee sell pressure
            return p.cancel * book.depth(side, 10) / p.q_ref * (1.0 + p.withdraw * adverse / p.v_ref)
        if (ba if i in (NOISE_MB, FUND_MB, TREND_MB) else bb) is None:
            return 0.0
        if i in (NOISE_MB, NOISE_MS):
            return p.noise_market
        mid = self.mid(book, self.last_mid)
        if i in (FUND_MB, FUND_MS):
            gap = (self.F - mid) if i == FUND_MB else (mid - self.F)
            return p.fund * max(gap, 0.0) ** p.fund_power
        k = p.trend * (p.trend_boost if self.trending else 1.0)
        return k * max(self.ret if i == TREND_MB else -self.ret, 0.0)

    def apply(self, i: int, u: np.ndarray, book: Book, t: float, tape: list) -> None:
        p = self.p
        bb, ba = book.best_bid(), book.best_ask()
        ref = book.last_price if book.last_price is not None else self.p0
        if i in (LIM_B, LIM_A):
            is_buy = i == LIM_B
            same = bb if is_buy else ba
            size = self.lot * geometric(u[0], p.limit_size_p)
            level = 0
            if same is None:
                price = (ba - 1 if ba is not None else ref - 1) if is_buy else (bb + 1 if bb is not None else ref + 1)
            else:
                spread = (ba - bb) if (bb is not None and ba is not None) else 1
                if spread >= 2 and u[1] < p.inside_prob:
                    level = -(1 + int(u[2] * (spread - 1)))
                else:
                    level = geometric(u[2], p.level_p) - 1
                price = same - level if is_buy else same + level
            if price > 0:
                book.limit(is_buy, price, size)
                self.record(t, LB if is_buy else LS, size, level, price)
        elif i in (CAN_B, CAN_A):
            side = "bid" if i == CAN_B else "ask"
            prices = book.prices(side, 10)
            if prices:
                vols = np.cumsum([book.level_qty(side, x) for x in prices], dtype=float)
                level = min(int(np.searchsorted(vols, u[1] * vols[-1], side="right")), len(prices) - 1)
                size = self.lot * geometric(u[0], p.limit_size_p)
                book.cancel(side, prices[level], size)
                self.record(t, CB if i == CAN_B else CS, size, level, prices[level])
        else:
            is_buy = i in (NOISE_MB, FUND_MB, TREND_MB)
            size = self.lot * geometric(u[0], p.market_size_p)
            got = sum(f[1] for f in book.market(is_buy, size))
            if got:
                tape.append((t, got if is_buy else -got))
                self._note_trade(got if is_buy else -got)
            self.record(t, MB if is_buy else MS, size, 0)

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        self.tick(book, t)
        self._note_trade(qty if is_buy else -qty)
        if self.log is not None:
            self.log.append((t, MB if is_buy else MS, qty, -1, 0) + book_state(book))


def regimes() -> dict[str, AMParams]:
    return {
        "R2a_am_anchored": AMParams(),
        "R2b_am_trending": AMParams(fund=0.04, trend=0.10, trend_boost=6.0, regime_switch=1.0 / 250.0, withdraw=3.5),
        # added 2 November 2026 (simrel/reports/lob_impact_design_addendum3.md)
        "R2c_am_anchor": AMParams(fund=0.30, fund_power=2.0, trend=0.02, withdraw=1.0),
        "R2d_am_toxic": AMParams(withdraw=5.0, v_ref=1000.0, limit=1.0),
        "R2e_am_calm": AMParams(fund_vol=0.10, trend=0.03, trend_boost=3.0, regime_switch=1.0 / 800.0),
    }
