"""
Reference world R1: a queue-reactive order book.

Order flow has no memory of past events. Instead every intensity is a
function of the current queues, in the spirit of queue-reactive models:

    join the best quote      rate falls as the best queue grows        (long queues deter joiners)
    post deeper in the book  constant rate per side
    improve the quote        only when the spread is two ticks or more  (spreads close quickly)
    cancel                   proportional to the resting depth
    market order             rate rises when the queue it would hit is short relative to the
                             opposite best queue (flow anticipates the next price move)

A large sell therefore has two effects beyond eating bids: the thinned bid
queue attracts further market sells (short-lived momentum), and it attracts
new bids (replenishment). Their balance, set by `theta` and the join rates,
determines how transient impact is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lob import Book
from .base import CB, CS, LB, LS, MB, MS, ThinnedWorld, geometric

JOIN_B, JOIN_A, DEEP_B, DEEP_A, IN_B, IN_A, CAN_B, CAN_A, MKT_B, MKT_S = range(10)  # MKT_B = market buy (hits asks)


@dataclass(frozen=True)
class QRParams:
    join: float = 1.2  # joins per second when the best queue is empty
    q0: float = 2000.0  # queue size at which the join rate halves
    deep: float = 0.5  # limit orders per second behind the best quote, per side
    inside: float = 1.5  # quote improvements per second when the spread allows
    cancel: float = 0.6  # cancels per second per q_ref shares of depth (top ten levels)
    q_ref: float = 8000.0
    market: float = 0.12  # market orders per second per side at balanced queues
    theta: float = 1.2  # sensitivity of market-order flow to queue imbalance
    limit_size_p: float = 0.45
    market_size_p: float = 0.35
    deep_p: float = 0.35
    init_levels: int = 10
    init_qty: int = 800


class QueueReactive(ThinnedWorld):
    name = "queue_reactive"

    def __init__(self, params: QRParams, seed: int) -> None:
        p = params
        lam_max = [3 * p.join, 3 * p.join, 2 * p.deep, 2 * p.deep, 2 * p.inside, 2 * p.inside, 6 * p.cancel, 6 * p.cancel,
                   p.market * np.exp(p.theta) * 1.5, p.market * np.exp(p.theta) * 1.5]
        super().__init__(10, np.maximum(lam_max, 0.5), seed, stream=1)
        self.p = p

    def init_book(self, book: Book) -> None:
        self.seed_book(book, self.p.init_levels, self.p.init_qty)

    def intensity(self, i: int, book: Book, t: float) -> float:
        p = self.p
        bb, ba = book.best_bid(), book.best_ask()
        qb = book.level_qty("bid", bb) if bb is not None else 0
        qa = book.level_qty("ask", ba) if ba is not None else 0
        if i in (JOIN_B, JOIN_A):
            return p.join / (1.0 + (qb if i == JOIN_B else qa) / p.q0)
        if i in (DEEP_B, DEEP_A):
            return p.deep
        if i in (IN_B, IN_A):
            return p.inside if (bb is not None and ba is not None and ba - bb >= 2) else 0.0
        if i in (CAN_B, CAN_A):
            return p.cancel * book.depth("bid" if i == CAN_B else "ask", 10) / p.q_ref
        # market orders: a sell hits the bid queue; more of them when that queue is the shorter one
        if (ba if i == MKT_B else bb) is None:
            return 0.0
        hit, other = (qa, qb) if i == MKT_B else (qb, qa)
        imb = (other - hit) / max(other + hit, 1)
        return p.market * float(np.exp(p.theta * imb))

    def apply(self, i: int, u: np.ndarray, book: Book, t: float, tape: list) -> None:
        p = self.p
        bb, ba = book.best_bid(), book.best_ask()
        ref = book.last_price if book.last_price is not None else self.p0
        if i in (JOIN_B, JOIN_A, DEEP_B, DEEP_A, IN_B, IN_A):
            is_buy = i in (JOIN_B, DEEP_B, IN_B)
            same = bb if is_buy else ba
            size = self.lot * geometric(u[0], p.limit_size_p)
            if same is None:
                level = 0
                price = (ba - 1 if ba is not None else ref - 1) if is_buy else (bb + 1 if bb is not None else ref + 1)
            else:
                if i in (IN_B, IN_A):
                    level = -(1 + int(u[1] * (ba - bb - 1)))
                elif i in (DEEP_B, DEEP_A):
                    level = geometric(u[1], p.deep_p)
                else:
                    level = 0
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
            size = self.lot * geometric(u[0], p.market_size_p)
            got = sum(f[1] for f in book.market(i == MKT_B, size))
            if got:
                tape.append((t, got if i == MKT_B else -got))
            self.record(t, MB if i == MKT_B else MS, size, 0)


def regimes() -> dict[str, QRParams]:
    return {
        "R1a_qr_resilient": QRParams(),
        "R1b_qr_fragile": QRParams(join=0.8, q0=1200.0, deep=0.35, q_ref=5000.0, market=0.14, theta=2.2, init_qty=500),
        # added 2 November 2026 (simrel/reports/lob_impact_design_addendum3.md)
        "R1c_qr_thin": QRParams(join=0.6, q0=1000.0, deep=0.3, init_qty=400),
        "R1d_qr_momentum": QRParams(theta=2.8, market=0.16),
        "R1e_qr_deep": QRParams(join=2.5, q0=4000.0, deep=0.9, cancel=0.4, init_qty=1500),
    }
