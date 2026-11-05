"""
A queue-reactive candidate simulator, fitted to an observable event log (exploratory addition, 4 November 2026).

In the spirit of Huang, Lehalle and Rosenbaum (2015) and of the projection used by Noble, Rosenbaum and Souilmi (2026):
the intensity of every event type is a function of the book's state, here the spread (1, 2, 3+ ticks) and the
imbalance of the ten-level depths (five fixed bins), estimated as count / time spent in the state (the maximum-
likelihood estimator of a piecewise-constant Poisson intensity). Event types: limit orders at the best, inside the
spread and behind the best, per side; cancellations at the best and behind it, per side; market orders per side.
Sizes and placement levels are drawn from their empirical distributions in the log. Unlike the Hawkes candidates, the
flow has no memory of past events; it reacts to a trade only through the book it leaves behind, which is how
queue-reactive models produce transient impact and reversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lob import Book
from ..reference_worlds.base import ThinnedWorld

KINDS = ["LB_best", "LB_in", "LB_deep", "LS_best", "LS_in", "LS_deep", "CB_best", "CB_deep", "CS_best", "CS_deep", "MB", "MS"]
IMB_EDGES = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
N_SPREAD, N_IMB = 3, 5


def state_of(spread: float, bid_depth: float, ask_depth: float) -> int:
    if not np.isfinite(spread) or spread < 1:
        return -1
    s = int(min(spread, 3)) - 1
    tot = bid_depth + ask_depth
    imb = (bid_depth - ask_depth) / tot if tot > 0 else 0.0
    j = int(np.clip(np.searchsorted(IMB_EDGES, imb, side="right") - 1, 0, N_IMB - 1))
    return s * N_IMB + j


def kind_of(etype: int, level: float) -> int:
    side = "B" if etype in (0, 2, 4) else "S"
    if etype in (0, 1):
        return KINDS.index(f"L{side}_" + ("in" if level < 0 else "best" if level == 0 else "deep"))
    if etype in (4, 5):
        return KINDS.index(f"C{side}_" + ("best" if level <= 0 else "deep"))
    return KINDS.index("MB" if etype == 2 else "MS")


@dataclass
class QRCandidateParams:
    lam: np.ndarray  # (n_states, n_kinds) intensities per second
    limit_size_q: np.ndarray
    market_size_q: np.ndarray
    deep_level_q: np.ndarray
    cancel_level_q: np.ndarray
    init_qty: int
    p0: int
    lot: int = 100


def fit(observed: list, p0: int, lot: int = 100) -> tuple[QRCandidateParams, dict]:
    """observed: simrel.reference_worlds.data.Observed episodes (events: t, type, size, level, price, spread, bid depth,
    ask depth before the event)."""
    ns = N_SPREAD * N_IMB
    count, time = np.zeros((ns, len(KINDS))), np.zeros(ns)
    lim, mkt, deep, can = [], [], [], []
    depth = []
    for o in observed:
        e = o.events
        if len(e) < 2:
            continue
        st = np.array([state_of(r[5], r[6], r[7]) for r in e])
        dt = np.diff(e[:, 0])
        for i in range(1, len(e)):  # the state in force between events i-1 and i is the one recorded before event i
            if st[i] >= 0:
                time[st[i]] += dt[i - 1]
                count[st[i], kind_of(int(e[i, 1]), e[i, 3])] += 1
        lim += list(e[np.isin(e[:, 1], (0, 1)), 2])
        mkt += list(e[np.isin(e[:, 1], (2, 3)), 2])
        deep += list(e[np.isin(e[:, 1], (0, 1)) & (e[:, 3] > 0), 3])
        can += list(e[np.isin(e[:, 1], (4, 5)) & (e[:, 3] > 0), 3])
        depth += list(0.5 * (e[:, 6] + e[:, 7]))
    lam = np.where(time[:, None] > 1.0, count / np.maximum(time[:, None], 1e-9), np.nan)
    # states hardly visited take the average over states with the same spread, then the overall average
    overall = count.sum(0) / max(time.sum(), 1e-9)
    for s in range(N_SPREAD):
        rows = slice(s * N_IMB, (s + 1) * N_IMB)
        tm = time[rows].sum()
        avg = count[rows].sum(0) / tm if tm > 1.0 else overall
        block = lam[rows]
        block[np.isnan(block)] = np.broadcast_to(avg, block.shape)[np.isnan(block)]
        lam[rows] = block
    q = lambda x, default: np.quantile(np.asarray(x, float), (np.arange(500) + 0.5) / 500) if len(x) else np.array([default], float)
    params = QRCandidateParams(lam=lam, limit_size_q=q(lim, lot), market_size_q=q(mkt, lot), deep_level_q=np.maximum(q(deep, 1), 1),
                               cancel_level_q=np.maximum(q(can, 1), 1), init_qty=int(max(round(np.mean(depth) / 10 / lot) * lot, lot)) if depth else 4 * lot,
                               p0=int(p0), lot=lot)
    return params, {"time_in_state": time.tolist(), "events": int(count.sum()), "states_visited": int((time > 1.0).sum())}


class QRCandidate(ThinnedWorld):
    """Simulation of the fitted queue-reactive candidate by thinning (shared machinery with the reference worlds)."""

    name = "qr_candidate"

    def __init__(self, params: QRCandidateParams, seed: int) -> None:
        self.p = params
        self.p0 = params.p0
        self.lot = params.lot
        lam_max = np.maximum(np.nanmax(params.lam, axis=0) * 1.05, 1e-3)
        super().__init__(len(KINDS), lam_max, seed, stream=7)

    def init_book(self, book: Book) -> None:
        self.seed_book(book, 10, self.p.init_qty)

    def intensity(self, i: int, book: Book, t: float) -> float:
        bb, ba = book.best_bid(), book.best_ask()
        spread = float(ba - bb) if (bb is not None and ba is not None) else 3.0
        if KINDS[i].endswith("_in") and spread < 2:
            return 0.0
        s = state_of(spread, float(book.depth("bid", 10)), float(book.depth("ask", 10)))
        return float(self.p.lam[max(s, 0), i])

    @staticmethod
    def _q(u: float, q: np.ndarray) -> float:
        return float(q[min(int(u * len(q)), len(q) - 1)])

    def apply(self, i: int, u: np.ndarray, book: Book, t: float, tape: list) -> None:
        k = KINDS[i]
        bb, ba = book.best_bid(), book.best_ask()
        mid = self.mid(book, self.p0)
        if k[0] == "L":
            is_buy = k[1] == "B"
            size = max(int(round(self._q(u[0], self.p.limit_size_q) / self.lot)) * self.lot, self.lot)
            same, opp = (bb, ba) if is_buy else (ba, bb)
            if same is None:
                same = int(np.floor(mid)) if is_buy else int(np.ceil(mid))
            if k.endswith("_best"):
                price = same
            elif k.endswith("_in"):
                if opp is None or abs(opp - same) < 2:
                    return
                step = 1 + int(u[1] * (abs(opp - same) - 1))
                price = same + step if is_buy else same - step
            else:
                lvl = int(round(self._q(u[1], self.p.deep_level_q)))
                price = same - lvl if is_buy else same + lvl
            if price > 0:
                book.limit(is_buy, int(price), size)
        elif k[0] == "C":
            side = "bid" if k[1] == "B" else "ask"
            prices = book.prices(side, 10)
            if not prices:
                return
            if k.endswith("_best"):
                target = prices[0]
            else:
                lvl = int(round(self._q(u[1], self.p.cancel_level_q)))
                want = prices[0] - lvl if side == "bid" else prices[0] + lvl
                target = min(prices, key=lambda x: abs(x - want))
            book.cancel_order(side, target, u[2])
        else:
            is_buy = k == "MB"
            if (ba if is_buy else bb) is None:
                return
            size = max(int(round(self._q(u[0], self.p.market_size_q) / self.lot)) * self.lot, self.lot)
            fills = book.market(is_buy, size)
            got = sum(f[1] for f in fills)
            if got:
                tape.append((t, got if is_buy else -got))
