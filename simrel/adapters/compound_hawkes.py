"""
Compound Hawkes limit-order-book candidate after Jain, Firoozye, Kochems and Treleaven (arXiv 2312.08927): the modern
Hawkes baseline of design addendum 10 (simrel/reports/lob_impact_design_addendum10.md, section 6).

Twelve event types, by side and position relative to the best quotes (the paper's six queues):

    0 LS_deep  sell limit behind the best ask      6 LB_deep  buy limit behind the best bid
    1 CS_deep  cancel behind the best ask          7 LB_best  buy limit at the best bid
    2 LS_best  sell limit at the best ask          8 CB_best  cancel at the best bid
    3 CS_best  cancel at the best ask              9 MS       market sell (hits the best bid)
    4 MB       market buy (hits the best ask)      10 LB_in   buy limit inside the spread (new best bid)
    5 LS_in    sell limit inside the spread        11 CB_deep cancel behind the best bid

    lambda_i(t) = mu_i g_i(book) + sum_m sum_j alpha_m[i, j] sum_{t_k^j < t} beta_m exp(-beta_m (t - t_k^j))

As in the paper: in-spread orders have g = (spread - 1)^s_exp (zero at a one-tick spread); event sizes are drawn per
type from a distribution with point masses at round lots (100, 200, 500 shares) and a geometric remainder; cancels
remove whole resting orders; excitation is unmarked (sizes do not change intensities). Departures, all forced by the
setting: the paper's non-parametric power-law kernels are approximated by a sum of exponentials with fixed decays
(BETAS, fitted by EM); the time-of-day profile is omitted (30-minute windows); our book keeps every level, so deep
limit orders and cancels are placed by the observed distribution of ticks behind the best rather than by the paper's
two-queue resampling, and cancel baselines scale with the resting depth on their side (keeping depth stationary).

Simulation keeps the study's design: each type owns a candidate stream with fixed uniforms, accepted by thinning, so a
twin and an intervention share randomness (simrel.adapters.hawkes.HawkesLOB).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eventsim import FlowModel
from ..lob import Book

D = 12
LS_DEEP, CS_DEEP, LS_BEST, CS_BEST, MB, LS_IN, LB_DEEP, LB_BEST, CB_BEST, MS, LB_IN, CB_DEEP = range(D)
BUY_SIDE = {LB_DEEP, LB_BEST, CB_BEST, LB_IN, CB_DEEP}  # events on the bid side of the book (MS hits it too)
LIMITS = {LS_DEEP: (False, "deep"), LS_BEST: (False, "best"), LS_IN: (False, "in"), LB_DEEP: (True, "deep"), LB_BEST: (True, "best"), LB_IN: (True, "in")}
CANCELS = {CS_DEEP: ("ask", "deep"), CS_BEST: ("ask", "best"), CB_BEST: ("bid", "best"), CB_DEEP: ("bid", "deep")}
BETAS = (0.1, 1.0, 10.0)
ROUND = (100, 200, 500)
N_U = 4


@dataclass
class CHParams:
    mu: np.ndarray  # (D,)
    alpha: np.ndarray  # (M, D target, D source)
    betas: np.ndarray  # (M,)
    s_exp: float  # in-spread baseline exponent
    ref_depth: float  # ten-level depth per side at which a cancel baseline equals mu
    size_round: np.ndarray  # (D, len(ROUND)) point-mass probabilities
    size_geo: np.ndarray  # (D,) geometric parameter (in lots) of the remainder
    deep_q: np.ndarray  # quantiles of ticks behind the best (>= 1) for deep limit orders
    cancel_deep_q: np.ndarray  # quantiles of price-level index (>= 1) for deep cancels
    lam_max: np.ndarray  # (D,) thinning bounds
    lot: int = 100
    p0: int = 1000
    init_levels: int = 10
    init_level_qty: int = 800


# ------------------------------------------------------------------ classification of observed events
def classify(ev: np.ndarray) -> np.ndarray:
    """Map reference-market events (simrel.reference_worlds.data.Observed.events) to the twelve types; -1 if unusable."""
    k, lev = ev[:, 1].astype(int), ev[:, 3]
    out = np.full(len(ev), -1)
    out[(k == 0) & (lev < 0)] = LB_IN
    out[(k == 0) & (lev == 0)] = LB_BEST
    out[(k == 0) & (lev >= 1)] = LB_DEEP
    out[(k == 1) & (lev < 0)] = LS_IN
    out[(k == 1) & (lev == 0)] = LS_BEST
    out[(k == 1) & (lev >= 1)] = LS_DEEP
    out[k == 2] = MB
    out[k == 3] = MS
    out[(k == 4) & (lev <= 0)] = CB_BEST
    out[(k == 4) & (lev >= 1)] = CB_DEEP
    out[(k == 5) & (lev <= 0)] = CS_BEST
    out[(k == 5) & (lev >= 1)] = CS_DEEP
    return out


def _g_events(ty, spread, bdep, adep, s_exp, ref_depth):
    g = np.ones(len(ty))
    inside = (ty == LS_IN) | (ty == LB_IN)
    g[inside] = np.maximum(spread[inside] - 1.0, 0.0) ** s_exp
    bid_c = (ty == CB_BEST) | (ty == CB_DEEP)
    ask_c = (ty == CS_BEST) | (ty == CS_DEEP)
    g[bid_c] = bdep[bid_c] / ref_depth
    g[ask_c] = adep[ask_c] / ref_depth
    return np.maximum(g, 1e-6)


def _G_episode(o, s_exp, ref_depth):
    """Time integral of each type's state factor over one episode (from the 5-second grid)."""
    dt = np.diff(np.append(o.grid_t, o.duration))
    spr = np.maximum(o.ask - o.bid, 1.0)
    ins = float((np.maximum(spr - 1.0, 0.0) ** s_exp * dt).sum())
    bd, ad = float((o.bid_depth10 / ref_depth * dt).sum()), float((o.ask_depth10 / ref_depth * dt).sum())
    G = np.full(D, o.duration)
    G[[LS_IN, LB_IN]] = ins
    G[[CB_BEST, CB_DEEP]] = bd
    G[[CS_BEST, CS_DEEP]] = ad
    return G


# ------------------------------------------------------------------ EM for a multi-exponential kernel
def _features(t, ty, betas):
    """A[n, m, j] = sum over earlier events k of type j of beta_m exp(-beta_m (t_n - t_k))."""
    betas = np.asarray(betas, float)
    M = len(betas)
    A = np.zeros((len(t), M, D))
    a = np.zeros((M, D))
    last = 0.0
    for n in range(len(t)):
        a *= np.exp(-betas * (t[n] - last))[:, None]
        A[n] = a
        a[:, ty[n]] += betas
        last = t[n]
    return A


def em(seqs, betas, n_iter: int = 80):
    """seqs: (t, types, g, G, T) per episode. Returns mu (D,), alpha (M, D, D), log-likelihood."""
    betas = np.asarray(betas, float)
    M = len(betas)
    feats, tys, gs = [], [], []
    G_tot = np.zeros(D)
    E = np.zeros((M, D))  # integrated kernel mass of each source type
    for t, ty, g, G, T in seqs:
        feats.append(_features(t, ty, betas))
        tys.append(ty)
        gs.append(g)
        G_tot += G
        for m in range(M):
            np.add.at(E[m], ty, 1.0 - np.exp(-betas[m] * (T - t)))
    A, ty, g = np.concatenate(feats), np.concatenate(tys), np.concatenate(gs)
    counts = np.bincount(ty, minlength=D).astype(float)
    mu = np.maximum(0.5 * counts / np.maximum(G_tot, 1e-9), 1e-5)
    alpha = np.full((M, D, D), 0.02 / M)
    ll = -np.inf
    for _ in range(n_iter):
        base = mu[ty] * g
        exc = np.einsum("nmj,mnj->nmj", A, alpha[:, ty, :])  # (n, M, D) contributions alpha_m[ty_n, j] A[n, m, j]
        lam = np.maximum(base + exc.sum(axis=(1, 2)), 1e-12)
        ll = float(np.log(lam).sum() - (mu * G_tot).sum() - np.einsum("mij,mj->", alpha, E))
        r0 = base / lam
        r = exc / lam[:, None, None]
        mu = np.maximum(np.bincount(ty, weights=r0, minlength=D) / np.maximum(G_tot, 1e-9), 1e-6)
        num = np.zeros((M, D, D))
        for m in range(M):
            np.add.at(num[m], ty, r[:, m, :])
        alpha = num / np.maximum(E[:, None, :], 1e-9)
    return mu, alpha, ll


# ------------------------------------------------------------------ fit
def _quant(x, n=500):
    x = np.asarray(x, float)
    return np.quantile(x, (np.arange(n) + 0.5) / n, method="inverted_cdf") if len(x) else np.array([1.0])


NMAX = 400  # largest size, in lots, of the geometric remainder


def _trunc_pmf(q: float, lot: int) -> np.ndarray:
    """pmf over n = 1..NMAX lots of a geometric(q) with the round sizes removed and renormalised."""
    n = np.arange(1, NMAX + 1)
    pmf = q * (1 - q) ** (n - 1)
    pmf[np.isin(n * lot, ROUND)] = 0.0
    return pmf / pmf.sum()


def _trunc_ll(q: float, n: np.ndarray, lot: int) -> float:
    pmf = _trunc_pmf(q, lot)
    return float(np.log(np.maximum(pmf[np.minimum(n, NMAX) - 1], 1e-300)).sum())


def fit(observed: list, lot: int = 100, max_branching: float = 0.9, s_grid=(0.5, 1.0, 2.0), n_iter: int = 80) -> tuple[CHParams, dict]:
    ev = np.vstack([o.events for o in observed])
    ref_depth = float(0.5 * (np.concatenate([o.bid_depth10 for o in observed]).mean() + np.concatenate([o.ask_depth10 for o in observed]).mean()))
    best = None
    for s_exp in s_grid:
        seqs = []
        for o in observed:
            e = o.events
            ty = classify(e)
            keep = ty >= 0
            spr = np.nan_to_num(e[keep, 5], nan=5.0)
            g = _g_events(ty[keep], spr, e[keep, 6], e[keep, 7], s_exp, ref_depth)
            seqs.append((e[keep, 0], ty[keep], g, _G_episode(o, s_exp, ref_depth), o.duration))
        mu, alpha, ll = em(seqs, BETAS, n_iter)
        if best is None or ll > best[3]:
            best = (s_exp, mu, alpha, ll, seqs)
    s_exp, mu, alpha, ll, seqs = best
    eig0 = float(np.max(np.abs(np.linalg.eigvals(alpha.sum(axis=0)))))
    if eig0 >= max_branching:  # keep the process stable and the stationary event rates equal to the observed ones
        alpha = alpha * (max_branching / eig0)
        T = sum(s[4] for s in seqs)
        ty_all = np.concatenate([s[1] for s in seqs])
        rate = np.bincount(ty_all, minlength=D) / T
        g_mean = sum(s[3] for s in seqs) / T
        mu = np.maximum(rate - alpha.sum(axis=0) @ rate, 0.02 * rate) / np.maximum(g_mean, 1e-9)
    ty_all = classify(ev)
    sizes = ev[:, 2]
    size_round = np.zeros((D, len(ROUND)))
    size_geo = np.full(D, 0.5)
    for i in range(D):
        s = sizes[ty_all == i]
        if not len(s):
            continue
        for r, v in enumerate(ROUND):
            size_round[i, r] = float(np.mean(s == v))
        rest = s[~np.isin(s, ROUND)]
        if len(rest):  # maximum likelihood for a geometric (in lots) with the round sizes excluded, as it is sampled
            n = np.maximum(np.round(rest / lot), 1).astype(int)
            grid = np.linspace(0.02, 0.98, 97)
            size_geo[i] = float(grid[int(np.argmax([_trunc_ll(q, n, lot) for q in grid]))])
    lev = ev[:, 3]
    deep_q = _quant(lev[np.isin(ty_all, [LS_DEEP, LB_DEEP])])
    cancel_deep_q = _quant(lev[np.isin(ty_all, [CS_DEEP, CB_DEEP])])
    # thinning bounds: a high quantile of the intensities just after each event in the data
    peaks = []
    for t, ty, g, G, T in seqs:
        A = _features(t, ty, BETAS)
        A[np.arange(len(t)), :, ty] += np.asarray(BETAS)[None, :]
        peaks.append(np.einsum("nmj,mij->ni", A, alpha) + mu[None, :] * 4.0)
    lam_max = np.maximum(np.quantile(np.vstack(peaks), 0.995, axis=0) + 4.0 * mu * np.maximum(g.max(), 1.0), 3.0)
    mean_price = float(np.mean([0.5 * (o.bid + o.ask).mean() for o in observed]))
    p = CHParams(mu=mu, alpha=alpha, betas=np.asarray(BETAS, float), s_exp=s_exp, ref_depth=ref_depth, size_round=size_round,
                 size_geo=size_geo, deep_q=deep_q, cancel_deep_q=cancel_deep_q, lam_max=lam_max, lot=lot,
                 p0=int(round(mean_price)), init_level_qty=max(int(ref_depth / 10) // lot * lot, lot))
    eig = float(np.max(np.abs(np.linalg.eigvals(alpha.sum(axis=0)))))
    info = {"loglik": ll, "s_exp": s_exp, "branching_ratio": eig, "branching_before_cap": eig0, "n_events": int(len(ev)),
            "type_counts": np.bincount(ty_all[ty_all >= 0], minlength=D).tolist()}
    return p, info


# ------------------------------------------------------------------ simulator
class CompoundHawkesLOB(FlowModel):
    name = "chawkes"

    def __init__(self, params: CHParams, seed: int) -> None:
        self.p = params
        self.lot = params.lot
        self.p0 = params.p0
        self.rngs = [np.random.default_rng([seed, 100 + i]) for i in range(D)]
        self.next_t = np.array([r.exponential(1.0 / lm) for r, lm in zip(self.rngs, params.lam_max)])
        self.e = np.zeros((len(params.betas), D))  # excited intensity of each target type, per kernel component
        self.e_t = 0.0
        self.n_capped = 0
        self.log: list | None = None
        self._cdf = [None] * D

    def init_book(self, book: Book) -> None:
        p = self.p
        for k in range(p.init_levels):
            book.limit(True, p.p0 - 1 - k, p.init_level_qty)
            book.limit(False, p.p0 + 1 + k, p.init_level_qty)
        book.last_price = p.p0

    def _decay_to(self, t: float) -> None:
        if t > self.e_t:
            self.e *= np.exp(-self.p.betas * (t - self.e_t))[:, None]
            self.e_t = t

    def _excite(self, j: int) -> None:
        self.e += self.p.alpha[:, :, j] * self.p.betas[:, None]

    def _g(self, i: int, book: Book) -> float:
        bb, ba = book.best_bid(), book.best_ask()
        if i in (LS_IN, LB_IN):
            if bb is None or ba is None:
                return 0.0
            return float(max(ba - bb - 1, 0)) ** self.p.s_exp
        if i in CANCELS:
            return book.depth(CANCELS[i][0], 10) / self.p.ref_depth
        if i == MB and ba is None or i == MS and bb is None:
            return 0.0
        return 1.0

    def _size(self, i: int, u: float) -> int:
        cum = np.cumsum(self.p.size_round[i])
        for r, c in enumerate(cum):
            if u < c:
                return ROUND[r]
        v = (u - cum[-1]) / max(1.0 - cum[-1], 1e-12)
        if self._cdf[i] is None:
            self._cdf[i] = np.cumsum(_trunc_pmf(float(self.p.size_geo[i]), self.lot))
        n = int(np.searchsorted(self._cdf[i], v, side="right")) + 1
        return min(n, NMAX) * self.lot

    @staticmethod
    def _table(u: float, q: np.ndarray) -> int:
        return max(int(q[min(int(u * len(q)), len(q) - 1)]), 1)

    def _record(self, t, kind, size, level, price, pre):
        if self.log is not None:
            self.log.append((t, kind, size, level, price) + pre)

    def _apply(self, i: int, u: np.ndarray, book: Book, t: float, tape: list) -> None:
        bb, ba = book.best_bid(), book.best_ask()
        pre = ((ba - bb) if (bb is not None and ba is not None) else np.nan, book.depth("bid", 10), book.depth("ask", 10)) if self.log is not None else None
        ref = book.last_price if book.last_price is not None else self.p.p0
        if i in LIMITS:
            is_buy, where = LIMITS[i]
            same = bb if is_buy else ba
            size = self._size(i, u[0])
            if same is None:
                price = (ba - 1 if ba is not None else ref - 1) if is_buy else (bb + 1 if bb is not None else ref + 1)
            elif where == "best":
                price = same
            elif where == "deep":
                lv = self._table(u[1], self.p.deep_q)
                price = same - lv if is_buy else same + lv
            else:  # inside the spread
                spread = (ba - bb) if (bb is not None and ba is not None) else 1
                if spread < 2:
                    return
                step = 1 + int(u[1] * (spread - 1))
                price = same + step if is_buy else same - step
            if price > 0:
                book.limit(is_buy, int(price), size)
                lvl = {"best": 0, "in": -1}.get(where, abs(price - same) if same is not None else 0)
                self._record(t, 0 if is_buy else 1, size, lvl, int(price), pre)
        elif i in (MB, MS):
            size = self._size(i, u[0])
            fills = book.market(i == MB, size)
            got = sum(f[1] for f in fills)
            if got:
                tape.append((t, got if i == MB else -got))
            self._record(t, 2 if i == MB else 3, size, 0, 0, pre)
        else:
            side, where = CANCELS[i]
            prices = book.prices(side, 10)
            if not prices:
                return
            if where == "best":
                target = prices[0]
            else:
                if len(prices) < 2:
                    return
                lv = min(self._table(u[1], self.p.cancel_deep_q), len(prices) - 1)
                target = prices[lv]
            got = book.cancel_order(side, target, u[2])
            self._record(t, 4 if side == "bid" else 5, got, 0 if where == "best" else prices.index(target), int(target), pre)
        self._excite(i)

    def advance(self, book: Book, t: float, t_until: float, tape: list) -> None:
        lam_max = self.p.lam_max
        while True:
            i = int(np.argmin(self.next_t))
            tc = float(self.next_t[i])
            if tc >= t_until:
                break
            rng = self.rngs[i]
            u = rng.random(N_U)  # always the same number of draws per candidate
            self.next_t[i] = tc + rng.exponential(1.0 / lam_max[i])
            self._decay_to(tc)
            lam = self.p.mu[i] * self._g(i, book) + self.e[:, i].sum()
            if lam > lam_max[i]:
                self.n_capped += 1
                lam = lam_max[i]
            if u[3] * lam_max[i] < lam:
                book.now = tc
                self._apply(i, u, book, tc, tape)
        self._decay_to(t_until)

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        if qty > 0:
            self._decay_to(t)
            self._excite(MB if is_buy else MS)


def observe_model(model, seed: int, duration: float = 1800.0, grid_dt: float = 5.0):
    """The observable record of a compound Hawkes model, in the reference-market format (for parameter recovery)."""
    from ..reference_worlds.data import Observed

    model.log = []
    book, tape = Book(), []
    model.init_book(book)
    rows, t = [], 0.0
    last = (float(model.p0 - 1), float(model.p0 + 1))
    while t < duration - 1e-9:
        bb, ba = book.best_bid(), book.best_ask()
        last = (float(bb) if bb is not None else last[0], float(ba) if ba is not None else last[1])
        rows.append((t, last[0], last[1], book.depth("bid", 10), book.depth("ask", 10)))
        model.advance(book, t, t + grid_dt, tape)
        t += grid_dt
    g = np.array(rows, dtype=float)
    return Observed(seed, duration, np.array(model.log, dtype=float).reshape(-1, 8), g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4],
                    np.array(tape, dtype=float).reshape(-1, 2))
