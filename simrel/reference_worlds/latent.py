"""
Reference world R3: latent liquidity (held-out family of design addendum 8; not used to design any decision rule).

Liquidity is revealed around a latent reference price r_t rather than around the current quotes. r_t follows the
traded mid with relaxation time tau and also drifts exogenously:

    r <- r + (mid - r) (1 - exp(-dt / tau)) + fund_vol * sqrt(dt) * noise

Limit orders are placed a geometric number of ticks behind r on their own side, so after a sell metaorder pushes the
mid below r, new bids arrive at or above the best ask and lift the price back: impact is transient and decays as r
catches up with the mid, as in propagator and latent-order-book models. Market-order signs come in runs of
Pareto-distributed length (persistent order flow); cancellations are proportional to the resting depth. Nothing here is
a queue rule (R1), a fundamental-value agent (R2) or a self-exciting kernel (the Hawkes candidates).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..lob import Book
from .base import CB, CS, LB, LS, MB, MS, ThinnedWorld, book_state, geometric

LIM_B, LIM_A, CAN_B, CAN_A, MKT = range(5)
SEED = 20261106
LISTING = Path(__file__).resolve().parents[2] / "results" / "simrel" / "raw" / "latent_markets.json"
RANGES = [("tau", 20.0, 600.0, "log"), ("limit", 0.8, 2.5, "log"), ("market", 0.06, 0.20, "log"), ("run_alpha", 1.3, 2.5, "uniform"),
          ("level_p", 0.25, 0.6, "uniform"), ("fund_vol", 0.05, 0.3, "log")]


@dataclass(frozen=True)
class LatentParams:
    tau: float = 120.0  # seconds for the reference price to relax towards the mid
    limit: float = 1.4  # limit orders per second per side
    market: float = 0.12  # market orders per second per side
    run_alpha: float = 1.8  # Pareto exponent of sign-run lengths (smaller: longer runs)
    level_p: float = 0.4  # geometric placement behind the reference price
    fund_vol: float = 0.15  # ticks per sqrt(second) of exogenous drift of the reference
    cancel: float = 0.6  # cancels per second per q_ref shares of depth (top ten levels)
    q_ref: float = 8000.0
    limit_size_p: float = 0.45
    market_size_p: float = 0.35
    init_levels: int = 10
    init_qty: int = 800


class LatentLiquidity(ThinnedWorld):
    name = "latent_liquidity"

    def __init__(self, params: LatentParams, seed: int) -> None:
        p = params
        lam_max = [p.limit, p.limit, 8 * p.cancel, 8 * p.cancel, 2 * p.market]
        super().__init__(5, np.maximum(lam_max, 0.5), seed, stream=3)
        self.p = p
        self.exo = np.random.default_rng([seed, 3, 99])  # drift of the reference: never touched by orders
        self.r = float(self.p0)
        self.rel_t, self.exo_t = 0.0, 0.0
        self.run_sign, self.run_left = 1, 0

    def init_book(self, book: Book) -> None:
        self.seed_book(book, self.p.init_levels, self.p.init_qty)

    def tick(self, book: Book, t: float) -> None:
        p = self.p
        dt = t - self.rel_t
        if dt > 0:  # relaxation of the reference towards the traded mid
            self.r += (self.mid(book, self.r) - self.r) * (1.0 - np.exp(-dt / p.tau))
            self.rel_t = t
        while self.exo_t + 1.0 <= t:  # exogenous drift in one-second steps, from its own stream
            self.exo_t += 1.0
            self.r += p.fund_vol * self.exo.standard_normal()

    def intensity(self, i: int, book: Book, t: float) -> float:
        p = self.p
        if i in (LIM_B, LIM_A):
            return p.limit
        if i in (CAN_B, CAN_A):
            return p.cancel * book.depth("bid" if i == CAN_B else "ask", 10) / p.q_ref
        return 2 * p.market

    def apply(self, i: int, u: np.ndarray, book: Book, t: float, tape: list) -> None:
        p = self.p
        if i in (LIM_B, LIM_A):
            is_buy = i == LIM_B
            size = self.lot * geometric(u[0], p.limit_size_p)
            level = geometric(u[2], p.level_p) - 1
            price = int(np.floor(self.r)) - level if is_buy else int(np.ceil(self.r)) + level
            same = book.best_bid() if is_buy else book.best_ask()
            if price > 0:
                book.limit(is_buy, price, size)
                rel = (same - price if is_buy else price - same) if same is not None else 0
                self.record(t, LB if is_buy else LS, size, int(rel), price)
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
            if self.run_left <= 0:  # a new run: sign and Pareto length from this candidate's own uniforms
                self.run_sign = 1 if u[1] < 0.5 else -1
                self.run_left = int(np.ceil((1.0 - min(u[2], 1 - 1e-12)) ** (-1.0 / (p.run_alpha - 1.0))))
            self.run_left -= 1
            is_buy = self.run_sign > 0
            if (book.best_ask() if is_buy else book.best_bid()) is None:
                return
            size = self.lot * geometric(u[0], p.market_size_p)
            got = sum(f[1] for f in book.market(is_buy, size))
            if got:
                tape.append((t, got if is_buy else -got))
            self.record(t, MB if is_buy else MS, size, 0)

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        self.tick(book, t)
        if self.log is not None:
            self.log.append((t, MB if is_buy else MS, qty, -1, 0) + book_state(book))


def draw(rng) -> dict:
    out = {}
    for name, lo, hi, kind in RANGES:
        out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind == "log" else float(rng.uniform(lo, hi))
    return out


def accepted() -> dict:
    if not LISTING.exists():
        return {}
    return {k: v["params"] for k, v in json.loads(LISTING.read_text())["accepted"].items()}
