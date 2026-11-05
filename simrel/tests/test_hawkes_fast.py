"""The compiled loop must make exactly the decisions a plain-Python loop makes on the same candidate streams."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simrel.adapters.hawkes import CB, CS, LB, LS, HawkesParams, regimes  # noqa: E402
from simrel.adapters.hawkes_fast import FastHawkesLOB  # noqa: E402
from simrel.lob import Book  # noqa: E402


class PythonLoop(FastHawkesLOB):
    """Same candidate blocks, acceptance decided in Python exactly as HawkesLOB.advance does."""

    def advance(self, book, t, t_until, tape):
        p = self.p
        while True:
            for j in range(6):
                if self.nxt[j] >= self.T.shape[1] - 1:
                    self._refill(j, 0.0)
            i = int(np.argmin(self.T[np.arange(6), self.nxt]))
            tc = float(self.T[i, self.nxt[i]])
            if tc >= t_until:
                break
            k = self.nxt[i]
            self.nxt[i] += 1
            u = self.U[i, k]
            self._decay_to(tc)
            lam = p.mu[i] * self._g(i, book) + self.e[:, i].sum()
            close = fill = 0.0
            if i in (LB, LS):
                bb, ba = book.best_bid(), book.best_ask()
                if p.close_rate > 0 and bb is not None and ba is not None and ba - bb >= 2:
                    close = p.close_rate
                if p.fill_rate > 0:
                    lv = book.prices("bid" if i == LB else "ask", 2)
                    if len(lv) == 2 and abs(lv[0] - lv[1]) > 1:
                        fill = p.fill_rate
                tot = lam + close + fill
                self._p_inside = close / tot if close > 0 else 0.0
                self._p_fill = fill / tot if fill > 0 else 0.0
                lam = tot
            if lam > self.lam_max[i]:
                self.n_capped += 1
                lam = self.lam_max[i]
            if u[4] * self.lam_max[i] < lam:
                book.now = tc
                self._apply(i, u, book, tc, tape)
        self._decay_to(t_until)


def _run(cls, params, seed, horizon=300.0):
    m, b = cls(params, seed), Book()
    b.tape = []
    m.init_book(b)
    tape = []
    m.advance(b, 0.0, horizon, tape)
    return b.tape, m.n_events.copy()


def test_fast_loop_matches_python_loop():
    import dataclasses

    base = regimes()["H1_liquid"]
    for params in (base, dataclasses.replace(base, close_rate=5.0, fill_rate=20.0, lam_max=np.maximum(8 * base.mu, 3) + 25.0),
                   dataclasses.replace(base, betas=np.array([0.5, 5.0]), alpha_stack=np.stack([base.alpha * 0.5, base.alpha * 0.5]))):
        for seed in (1, 2):
            tape_fast, n_fast = _run(FastHawkesLOB, params, seed)
            tape_py, n_py = _run(PythonLoop, params, seed)
            assert np.array_equal(n_fast, n_py)
            assert len(tape_fast) == len(tape_py) and all(a[:2] == b[:2] and a[2:] == b[2:] for a, b in zip(tape_fast, tape_py))
