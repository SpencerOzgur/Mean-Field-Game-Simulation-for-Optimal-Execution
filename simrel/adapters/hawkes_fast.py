"""
A faster version of the Hawkes simulator for kernels with fast decays.

The simulator thins state-independent candidate streams. With a fast kernel the cap on each intensity is high, so
almost all candidates are rejected, and the pure-Python loop spends its time rejecting. Here the candidate times and
their uniforms are drawn in blocks, and a compiled loop (numba) runs through candidates until one is accepted. Between
two accepted events the book does not change, so each type's baseline (mu * g(book) plus the spread-closing and
gap-filling terms) is fixed and only the excitation decays, which the compiled loop handles. The accepted event is
applied by the same Python code as in HawkesLOB.

The candidate streams are still independent of the market state, so twins keep sharing candidates and marks; the
draws come in a different order than in HawkesLOB, so the two classes give different (equally distributed) paths for
the same seed.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..lob import Book
from .hawkes import CB, CS, LB, LS, MB, MS, HawkesLOB

BLOCK = 4096


@njit(cache=True)
def _next_accept(t_until, nxt, T, U, e, e_t, betas, base, lam_max):
    """Run candidates until one is accepted. Returns (code, type, time, index, excited intensity of that type, e_t, capped).
    code 0 = accepted, 1 = reached t_until (e decayed to t_until), 2 = stream `type` needs more candidates."""
    capped = 0
    n_types, block = T.shape
    while True:
        for j in range(n_types):  # keep one unused candidate in every stream, so T[j, nxt[j]] is always valid
            if nxt[j] >= block - 1:
                return 2, j, e_t, -1, 0.0, e_t, capped
        i = 0
        for j in range(1, n_types):
            if T[j, nxt[j]] < T[i, nxt[i]]:
                i = j
        tc = T[i, nxt[i]]
        if tc >= t_until:
            if t_until > e_t:
                for m in range(betas.shape[0]):
                    f = np.exp(-betas[m] * (t_until - e_t))
                    for j in range(n_types):
                        e[m, j] *= f
                e_t = t_until
            return 1, -1, t_until, -1, 0.0, e_t, capped
        if tc > e_t:
            for m in range(betas.shape[0]):
                f = np.exp(-betas[m] * (tc - e_t))
                for j in range(n_types):
                    e[m, j] *= f
            e_t = tc
        ex = 0.0
        for m in range(betas.shape[0]):
            ex += e[m, i]
        lam = base[i] + ex
        if lam > lam_max[i]:
            capped += 1
            lam = lam_max[i]
        k = nxt[i]
        nxt[i] = k + 1
        if U[i, k, 4] * lam_max[i] < lam:
            return 0, i, tc, k, ex, e_t, capped


class FastHawkesLOB(HawkesLOB):
    name = "hawkes_fast"

    def __init__(self, params, seed: int) -> None:
        super().__init__(params, seed)
        n = 6
        self.T = np.empty((n, BLOCK))
        self.U = np.empty((n, BLOCK, self.N_UNIFORMS))
        self.nxt = np.full(n, BLOCK, dtype=np.int64)  # nothing drawn yet: the first refill draws a whole block
        self._last_t = np.zeros(n)
        for i in range(n):
            self._refill(i, 0.0)

    def _refill(self, i: int, t0: float) -> None:
        """Draw the next block of candidates of type i, continuing after the last unused one."""
        rng = self.rngs[i]
        k = self.nxt[i]
        keep_T, keep_U = self.T[i, k:].copy(), self.U[i, k:].copy()
        start = keep_T[-1] if len(keep_T) else self._last_t[i]
        fresh = BLOCK - len(keep_T)
        gaps = rng.exponential(1.0 / self.lam_max[i], size=fresh)
        self.T[i, : len(keep_T)], self.U[i, : len(keep_U)] = keep_T, keep_U
        self.T[i, len(keep_T):] = start + np.cumsum(gaps)
        self.U[i, len(keep_U):] = rng.random((fresh, self.N_UNIFORMS))
        self._last_t[i] = self.T[i, -1]
        self.nxt[i] = 0

    def _baselines(self, book: Book) -> tuple[np.ndarray, dict]:
        """Each type's intensity apart from excitation, and the extra terms that decide marks, for the current book."""
        p = self.p
        base = np.empty(6)
        extra = {}
        for i in range(6):
            g = self._g(i, book)
            if i in (CB, CS):
                extra[i] = self._p_best
            base[i] = p.mu[i] * g
        bb, ba = book.best_bid(), book.best_ask()
        for i in (LB, LS):
            close = fill = 0.0
            if p.close_rate > 0 and bb is not None and ba is not None and ba - bb >= 2:
                close = p.close_rate
            if p.fill_rate > 0:
                lv = book.prices("bid" if i == LB else "ask", 2)
                if len(lv) == 2 and abs(lv[0] - lv[1]) > 1:
                    fill = p.fill_rate
            extra[i] = (base[i], close, fill)
            base[i] += close + fill
        return base, extra

    def advance(self, book: Book, t: float, t_until: float, tape: list) -> None:
        p = self.p
        base, extra = self._baselines(book)
        while True:
            code, i, tc, k, ex, self.e_t, capped = _next_accept(t_until, self.nxt, self.T, self.U, self.e, self.e_t,
                                                                 self._betas, base, self.lam_max)
            self.n_capped += capped
            if code == 1:
                return
            if code == 2:
                self._refill(i, tc)
                continue
            if i in (LB, LS):  # which of the arriving processes this limit order came from
                b0, close, fill = extra[i]
                lam = b0 + ex + close + fill
                self._p_inside = close / lam if close > 0 else 0.0
                self._p_fill = fill / lam if fill > 0 else 0.0
            elif i in (CB, CS):
                self._p_best = extra[i]
            book.now = tc
            self._apply(i, self.U[i, k], book, tc, tape)
            base, extra = self._baselines(book)
