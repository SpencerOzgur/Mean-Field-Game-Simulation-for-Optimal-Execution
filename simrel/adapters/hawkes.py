"""
Multivariate marked Hawkes limit-order-book simulator.

Six event types arrive with mutually exciting intensities

    lambda_i(t) = mu_i * g_i(book) + sum_j alpha_ij * sum_{k: t_k^j < t} w_k * beta * exp(-beta (t - t_k^j))

    0 LB limit buy     2 MB market buy     4 CB cancel on the bid side
    1 LS limit sell    3 MS market sell    5 CS cancel on the ask side

alpha_ij is the expected number of type-i events triggered by one type-j
event of unit weight; w_k is a mark weight: 1 for limit orders and cancels,
and (size / mean size) ** weight_exponent, capped, for market orders. g_i makes the baseline state-dependent: cancellations scale
with the resting depth on their side (which keeps depth stationary) and
market orders thin out when the spread is wide.

Every event carries marks (size; price level for limit orders and cancels)
and is applied to a price-time-priority book (simrel.lob.Book), so prices,
spread and queues are outcomes, not inputs. An injected market order acts on
the book directly and also enters the intensities as an MB/MS event with its
own mark weight, so the background flow reacts to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eventsim import FlowModel
from ..lob import Book

LB, LS, MB, MS, CB, CS = range(6)


def default_alpha() -> np.ndarray:
    a = np.zeros((6, 6))
    #            source:  LB    LS    MB    MS    CB    CS
    a[LB] = [0.20, 0.05, 0.05, 0.15, 0.10, 0.00]
    a[LS] = [0.05, 0.20, 0.15, 0.05, 0.00, 0.10]
    a[MB] = [0.00, 0.02, 0.35, 0.00, 0.00, 0.05]
    a[MS] = [0.02, 0.00, 0.00, 0.35, 0.05, 0.00]
    a[CB] = [0.10, 0.00, 0.00, 0.20, 0.15, 0.00]
    a[CS] = [0.00, 0.10, 0.20, 0.00, 0.00, 0.15]
    return a


@dataclass
class HawkesParams:
    mu: np.ndarray = field(default_factory=lambda: np.array([0.60, 0.60, 0.05, 0.05, 0.50, 0.50]))  # events per second
    alpha: np.ndarray = field(default_factory=default_alpha)
    beta: float = 0.5  # 1/s; excitation half-life is ln(2)/beta
    lot: int = 100
    p0: int = 1000  # initial mid in ticks (tick = 1/p0 of the price)
    ref_depth: float = 20000.0  # depth (top 10 levels, one side) at which the cancel baseline equals mu
    limit_size_p: float = 0.45  # limit and cancel sizes are lot * Geometric(p)
    market_size_p: float = 0.35
    level_p: float = 0.35  # limit price is Geometric(p) - 1 ticks behind the best quote
    inside_prob: float = 0.35  # chance a limit order improves the quote when the spread allows
    init_levels: int = 10
    init_level_qty: int = 2000
    weight_exponent: float = 1.0  # excitation of a market order scales as (size / mean size) ** exponent ...
    weight_cap: float = 10.0  # ... up to this cap
    # Optional empirical marks (set when fitting to real order data). Each *_q is a table of equally spaced quantiles,
    # sampled by inverse CDF from one uniform, which replaces the geometric distribution of the same name.
    limit_size_q: np.ndarray | None = None
    market_size_q: np.ndarray | None = None
    level_q: np.ndarray | None = None  # ticks behind the best quote for limit orders that do not improve it
    cancel_level_q: np.ndarray | None = None  # ticks behind the best quote of the order that is cancelled
    whole_order_cancels: bool = False  # a cancel removes one whole resting order (as deletions do in exchange data)
    init_orders: tuple | None = None  # starting book as (is_buy, price in ticks, shares) rows, instead of the flat default
    mean_market: float | None = None
    lam_max: np.ndarray | None = None  # per-type cap on the intensity (candidate rate); default max(8 mu, 3)
    # Multiple timescales: the kernel is a sum of exponentials with these decays; alpha_stack[m] is component m's
    # branching matrix (alpha is then their sum). None = the single kernel (alpha, beta).
    betas: np.ndarray | None = None
    alpha_stack: np.ndarray | None = None
    # Optional realism fixes, each off by default so earlier versions stay reproducible.
    # Spread: while the spread is two ticks or more, limit orders that improve the quote arrive at this extra rate per
    # side (per second), replacing the fixed inside_prob.
    close_rate: float = 0.0
    # Queue: the cancel baseline is split into the best queue and the levels behind it, each proportional to its own
    # resting size: g = queue_w * q1 / ref_q1 + (1 - queue_w) * rest / ref_rest. queue_w is the share of cancels at the best.
    queue_w: float | None = None
    ref_q1: float = 1.0
    ref_rest: float = 1.0
    deep_cancel_level_q: np.ndarray | None = None  # ticks behind the best (>= 1) for cancels not at the best
    # Size: market orders are sized relative to the queue they hit (size = ratio * best opposite queue), with the
    # ratio drawn from its distribution in the data, replacing the size distribution that ignores the queue.
    market_ratio_q: np.ndarray | None = None
    # Gap filling: while the second price level on a side is more than one tick behind the best, limit orders arrive
    # one tick behind the best on that side at this extra rate (per second). In the INTC data that gap is never seen.
    fill_rate: float = 0.0

    def mean_market_size(self) -> float:
        return self.mean_market if self.mean_market else self.lot / self.market_size_p


class HawkesLOB(FlowModel):
    """Simulates the process by thinning per-type streams of candidate events.

    Each event type owns a random stream that produces candidate times at a
    constant rate lam_max_i, and for every candidate a fixed set of uniforms
    (one for acceptance, the rest for marks). A candidate becomes an event when
    its acceptance uniform falls below lambda_i(t) / lam_max_i. Because the
    candidate streams do not depend on the state of the market, two copies of
    the simulator that share a history (a twin and an intervention) keep
    drawing the same candidates and the same marks, and differ only where their
    intensities or books differ. This is what makes twin comparisons precise.
    Intensities are capped at lam_max_i.
    """

    name = "hawkes"
    N_UNIFORMS = 5

    def __init__(self, params: HawkesParams, seed: int) -> None:
        self.p = params
        self.lot = params.lot
        self.lam_max = np.maximum(8.0 * params.mu, 3.0) if params.lam_max is None else np.asarray(params.lam_max, dtype=float)
        self.rngs = [np.random.default_rng([seed, i]) for i in range(6)]
        self.next_t = np.array([r.exponential(1.0 / lm) for r, lm in zip(self.rngs, self.lam_max)])
        self._betas = np.atleast_1d(np.asarray(params.betas if params.betas is not None else [params.beta], dtype=float))
        self._alphas = np.asarray(params.alpha_stack) if params.alpha_stack is not None else np.asarray(params.alpha)[None]
        self.e = np.zeros((len(self._betas), 6))  # excited part of each intensity, per kernel component
        self.e_t = 0.0  # time at which self.e is valid
        self.n_events = np.zeros(6, dtype=np.int64)
        self.n_capped = 0
        self.log: list | None = None  # optional event log in the reference-world format, for fitting
        self._pre = (1.0, 0.0, 0.0)
        self._p_inside, self._p_best, self._p_fill = 0.0, 0.0, 0.0

    # ---- book set-up
    def init_book(self, book: Book) -> None:
        p = self.p
        if p.init_orders is not None:
            for is_buy, price, qty in p.init_orders:
                book.limit(bool(is_buy), int(price), int(qty))
            book.last_price = p.p0
            return
        for k in range(p.init_levels):
            book.limit(True, p.p0 - 1 - k, p.init_level_qty)
            book.limit(False, p.p0 + 1 + k, p.init_level_qty)
        book.last_price = p.p0

    # ---- intensities
    def _g(self, i: int, book: Book) -> float:
        bb, ba = book.best_bid(), book.best_ask()
        if i in (MB, MS):
            if (ba if i == MB else bb) is None:
                return 0.0
            spread = (ba - bb) if (bb is not None and ba is not None) else 5
            return 1.0 / max(spread, 1)
        if i in (CB, CS):
            side = "bid" if i == CB else "ask"
            depth = book.depth(side, 10)
            if self.p.queue_w is None:
                return depth / self.p.ref_depth
            best = bb if i == CB else ba
            q1 = book.level_qty(side, best) if best is not None else 0
            at_best = self.p.queue_w * q1 / self.p.ref_q1
            total = at_best + (1.0 - self.p.queue_w) * (depth - q1) / self.p.ref_rest
            self._p_best = at_best / total if total > 0 else 0.0
            return total
        return 1.0

    def _decay_to(self, t: float) -> None:
        if t > self.e_t:
            self.e *= np.exp(-self._betas * (t - self.e_t))[:, None]
            self.e_t = t

    def _excite(self, j: int, weight: float = 1.0) -> None:
        self.e += self._alphas[:, :, j] * self._betas[:, None] * weight

    def _weight(self, size: int) -> float:
        return float(min((size / self.p.mean_market_size()) ** self.p.weight_exponent, self.p.weight_cap))

    @staticmethod
    def _table(u: float, q: np.ndarray) -> int:
        return int(q[min(int(u * len(q)), len(q) - 1)])

    @staticmethod
    def _geometric(u: float, prob: float) -> int:
        """Inverse-CDF geometric on {1, 2, ...} from one uniform."""
        return int(np.floor(np.log1p(-min(u, 1 - 1e-12)) / np.log1p(-prob))) + 1

    # ---- one event, with marks taken from the candidate's uniforms
    def _apply(self, i: int, u: np.ndarray, book: Book, t: float, tape: list) -> None:
        p = self.p
        bb, ba = book.best_bid(), book.best_ask()
        ref = book.last_price if book.last_price is not None else p.p0
        weight, size, level = 1.0, 0, 0
        if i in (LB, LS):
            size = self._table(u[0], p.limit_size_q) if p.limit_size_q is not None else p.lot * self._geometric(u[0], p.limit_size_p)
            is_buy = i == LB
            same = bb if is_buy else ba
            if same is None:
                price = (ba - 1 if ba is not None else ref - 1) if is_buy else (bb + 1 if bb is not None else ref + 1)
            elif self._p_fill > 0 and u[3] < self._p_fill:  # a gap-filling order, one tick behind the best
                level = 1
                price = same - 1 if is_buy else same + 1
            else:
                spread = (ba - bb) if (bb is not None and ba is not None) else 1
                if spread >= 2 and u[1] < (self._p_inside if p.close_rate > 0 else p.inside_prob):
                    level = -(1 + int(u[2] * (spread - 1)))
                else:
                    level = self._table(u[2], p.level_q) if p.level_q is not None else self._geometric(u[2], p.level_p) - 1
                price = same - level if is_buy else same + level
            if price > 0:
                book.limit(is_buy, price, size)
        elif i in (MB, MS):
            if p.market_ratio_q is not None:
                opp = ba if i == MB else bb
                q1 = book.level_qty("ask" if i == MB else "bid", opp) if opp is not None else 0
                ratio = float(p.market_ratio_q[min(int(u[0] * len(p.market_ratio_q)), len(p.market_ratio_q) - 1)])
                size = max(int(round(ratio * q1)), 1)
                # excitation keeps the weight distribution the fit used: the weight of the size at the same quantile
                weight_size = self._table(u[0], p.market_size_q)
            else:
                size = self._table(u[0], p.market_size_q) if p.market_size_q is not None else p.lot * self._geometric(u[0], p.market_size_p)
                weight_size = size
            fills = book.market(i == MB, size)
            got = sum(f[1] for f in fills)
            if got:
                tape.append((t, got if i == MB else -got))
            weight = self._weight(weight_size)
        else:
            side = "bid" if i == CB else "ask"
            prices = book.prices(side, 10)
            if prices and p.whole_order_cancels:
                if p.queue_w is None:
                    level = self._table(u[1], p.cancel_level_q)
                else:  # best queue or behind it, in proportion to the two parts of the baseline
                    level = 0 if u[3] < self._p_best else self._table(u[1], p.deep_cancel_level_q)
                target = prices[0] - level if i == CB else prices[0] + level
                if target not in (book.bids if i == CB else book.asks):  # nothing rests there: take the nearest visible level
                    target = min(prices, key=lambda x: abs(x - target))
                size = book.cancel_order(side, target, u[2])
            elif prices:
                vols = np.cumsum([book.level_qty(side, x) for x in prices], dtype=float)
                level = int(np.searchsorted(vols, u[1] * vols[-1], side="right"))
                level = min(level, len(prices) - 1)
                size = p.lot * self._geometric(u[0], p.limit_size_p)
                book.cancel(side, prices[level], size)
        self.n_events[i] += 1
        if self.log is not None:
            self.log.append((t, i, size, level, 0) + self._pre)
        self._excite(i, weight)

    # ---- FlowModel interface
    def advance(self, book: Book, t: float, t_until: float, tape: list) -> None:
        p = self.p
        while True:
            i = int(np.argmin(self.next_t))
            tc = float(self.next_t[i])
            if tc >= t_until:
                break
            rng = self.rngs[i]
            u = rng.random(self.N_UNIFORMS)  # always the same number of draws per candidate
            self.next_t[i] = tc + rng.exponential(1.0 / self.lam_max[i])
            self._decay_to(tc)
            lam = p.mu[i] * self._g(i, book) + self.e[:, i].sum()
            if p.close_rate > 0 and i in (LB, LS):
                bb, ba = book.best_bid(), book.best_ask()
                wide = bb is not None and ba is not None and ba - bb >= 2
                self._p_inside = p.close_rate / (lam + p.close_rate) if wide else 0.0
                lam += p.close_rate if wide else 0.0
            self._p_fill = 0.0
            if p.fill_rate > 0 and i in (LB, LS):
                lv = book.prices("bid" if i == LB else "ask", 2)
                if len(lv) == 2 and abs(lv[0] - lv[1]) > 1:
                    self._p_fill = p.fill_rate / (lam + p.fill_rate)
                    lam += p.fill_rate
            if lam > self.lam_max[i]:
                self.n_capped += 1
                lam = self.lam_max[i]
            if u[4] * self.lam_max[i] < lam:
                if self.log is not None:
                    from ..reference_worlds.base import book_state

                    self._pre = book_state(book)
                book.now = tc
                self._apply(i, u, book, tc, tape)
        self._decay_to(t_until)

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        if qty > 0:
            self._decay_to(t)
            self._excite(MB if is_buy else MS, self._weight(qty))
            if self.log is not None:
                self.log.append((t, MB if is_buy else MS, qty, -1, 0) + self._pre)


def regimes() -> dict[str, HawkesParams]:
    """Two hand-set regimes used in the pilot (tick = 10 bps of the price)."""
    return {
        "H1_liquid": HawkesParams(ref_depth=8000.0, init_level_qty=800, mu=np.array([0.60, 0.60, 0.10, 0.10, 0.50, 0.50])),
        "H2_thin": HawkesParams(ref_depth=3000.0, init_level_qty=300, mu=np.array([0.45, 0.45, 0.14, 0.14, 0.40, 0.40]), level_p=0.30),
    }


def branching_ratio(alpha: np.ndarray) -> float:
    """Spectral radius of alpha; the unconditioned process is stable when it is below 1."""
    return float(np.max(np.abs(np.linalg.eigvals(alpha))))
