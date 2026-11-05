"""
Base class for the reference worlds.

A reference world is a market whose rules we wrote and can therefore
interrogate with unlimited counterfactual runs. Each world defines a set of
event types with state-dependent intensities and applies them to the shared
matching engine. Simulation is by thinning state-independent candidate
streams (one per event type), exactly as in the Hawkes simulator, so an
unperturbed twin and an intervention stay on the same random path wherever
their states agree.

Every applied event is also written to an event log in a neutral six-type
vocabulary (limit / market / cancel, by side) with its size and price level.
That log is the only thing a candidate simulator is allowed to learn from.
"""

from __future__ import annotations

import numpy as np

from ..eventsim import FlowModel
from ..lob import Book

LB, LS, MB, MS, CB, CS = range(6)  # neutral event vocabulary of the observable log


def geometric(u: float, prob: float) -> int:
    """Inverse-CDF geometric on {1, 2, ...} from one uniform."""
    return int(np.floor(np.log1p(-min(u, 1 - 1e-12)) / np.log1p(-prob))) + 1


def book_state(book: Book) -> tuple[float, float, float]:
    """(spread in ticks, bid depth over ten levels, ask depth over ten levels); spread is nan if a side is empty."""
    bb, ba = book.best_bid(), book.best_ask()
    spread = float(ba - bb) if (bb is not None and ba is not None) else float("nan")
    return (spread, float(book.depth("bid", 10)), float(book.depth("ask", 10)))


class ThinnedWorld(FlowModel):
    N_UNIFORMS = 5
    lot = 100
    p0 = 1000  # initial mid in ticks; one tick is 10 bps of the price

    def __init__(self, n_types: int, lam_max: np.ndarray, seed: int, stream: int = 0) -> None:
        self.lam_max = np.asarray(lam_max, dtype=float)
        self.rngs = [np.random.default_rng([seed, stream, i]) for i in range(n_types)]
        self.next_t = np.array([r.exponential(1.0 / lm) for r, lm in zip(self.rngs, self.lam_max)])
        self.n_capped = 0
        self.n_events = 0
        self.log: list | None = None
        self._pre = (1.0, 0.0, 0.0)  # book state just before the current event (only tracked while logging)

    # ---- to be provided by each world
    def intensity(self, i: int, book: Book, t: float) -> float:
        raise NotImplementedError

    def apply(self, i: int, u: np.ndarray, book: Book, t: float, tape: list) -> None:
        raise NotImplementedError

    def tick(self, book: Book, t: float) -> None:
        """Hook called before each candidate, for exogenous state such as a latent fundamental."""

    # ---- helpers for subclasses
    def record(self, t: float, kind: int, size: int, level: int, price: int = 0) -> None:
        """Log one event: time, neutral type, shares, price level relative to the best quote, absolute price
        (limits and cancels), and the spread and ten-level depths just before the event."""
        self.n_events += 1
        if self.log is not None:
            self.log.append((t, kind, size, level, price) + self._pre)

    def seed_book(self, book: Book, levels: int, qty: int) -> None:
        for k in range(levels):
            book.limit(True, self.p0 - 1 - k, qty)
            book.limit(False, self.p0 + 1 + k, qty)
        book.last_price = self.p0

    @staticmethod
    def mid(book: Book, fallback: float) -> float:
        bb, ba = book.best_bid(), book.best_ask()
        if bb is not None and ba is not None:
            return 0.5 * (bb + ba)
        return float(bb if bb is not None else (ba if ba is not None else fallback))

    # ---- FlowModel interface
    def advance(self, book: Book, t: float, t_until: float, tape: list) -> None:
        while True:
            i = int(np.argmin(self.next_t))
            tc = float(self.next_t[i])
            if tc >= t_until:
                break
            rng = self.rngs[i]
            u = rng.random(self.N_UNIFORMS)
            self.next_t[i] = tc + rng.exponential(1.0 / self.lam_max[i])
            self.tick(book, tc)
            lam = self.intensity(i, book, tc)
            if lam > self.lam_max[i]:
                self.n_capped += 1
                lam = self.lam_max[i]
            if u[4] * self.lam_max[i] < lam:
                if self.log is not None:
                    self._pre = book_state(book)
                book.now = tc  # time stamp for the optional order-level record (simrel.lob.Book.tape)
                self.apply(i, u, book, tc, tape)
        self.tick(book, t_until)
