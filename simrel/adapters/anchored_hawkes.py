"""
Mechanism-corrected Hawkes candidate of design addendum 10, section 6 ("corrected").

The fitted Hawkes design h_c_f_m_k1 plus two mechanisms the Hawkes family lacks, added on a fixed clock of DT seconds:

  anchoring   a latent fundamental V follows a random walk with volatility sigma_f (ticks per sqrt(second)); market
              orders towards it arrive at rate f * |V - mid|^p_exp (per second, mispricing in ticks)
  withdrawal  resting orders on the side facing recent one-sided flow are cancelled at extra rate
              w * max(0, adverse flow) / V_REF, where the flow signal is the signed traded volume with half-life FLOW_HL

f, p_exp, sigma_f and w are estimated from the six observed episodes by simulated method of moments on the diagnostic
features of goal2_reliability (fit_smm). The Hawkes part keeps its own fit. Each clock step draws a fixed set of
random numbers, so a twin and an intervention share them (the study's common-random-numbers design).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eventsim import FlowModel
from ..lob import Book

DT = 0.5
FLOW_HL = 30.0  # seconds; fixed, not estimated (the diagnostic window), not taken from any reference market
V_REF = 1000.0
MAX_PER_STEP = 2  # at most this many events of each mechanism per clock step (rates are small: lambda * DT << 1)


@dataclass(frozen=True)
class AnchorParams:
    f: float = 0.1
    p_exp: float = 1.5
    sigma_f: float = 0.2
    w: float = 1.0


class AnchoredHawkes(FlowModel):
    name = "h_anchor"

    def __init__(self, base, ap: AnchorParams, seed: int) -> None:
        self.base, self.ap = base, ap
        self.lot = base.lot
        self.p = base.p  # observe() and the runner read p0 from here
        self.p0 = base.p.p0
        self.rng = np.random.default_rng([seed, 777])
        self.V = None
        self.flow = 0.0
        self.clock = 0.0
        self.decay = float(np.exp(-np.log(2.0) * DT / FLOW_HL))
        self.market_q = float(base.p.market_size_p)
        self.log = None

    def init_book(self, book: Book) -> None:
        self.base.init_book(book)
        self.V = float(self.p0)

    @staticmethod
    def _mid(book: Book, ref: float) -> float:
        bb, ba = book.best_bid(), book.best_ask()
        return 0.5 * (bb + ba) if bb is not None and ba is not None else ref

    def _step(self, book: Book, t: float, tape: list, n0: int) -> None:
        ap = self.ap
        z = self.rng.standard_normal()
        u = self.rng.random(4 * MAX_PER_STEP)  # fixed draws per step
        for _, q in tape[n0:]:  # everything traded in this step, background and injected
            self.flow += q
        self.flow *= self.decay
        self.V += ap.sigma_f * np.sqrt(DT) * z
        mid = self._mid(book, self.V)
        mis = self.V - mid
        lam_f = ap.f * abs(mis) ** ap.p_exp
        book.now = t
        for k in range(MAX_PER_STEP):
            if u[k] < lam_f * DT / MAX_PER_STEP:
                is_buy = mis > 0
                size = self.lot * (int(np.floor(np.log1p(-min(u[MAX_PER_STEP + k], 1 - 1e-12)) / np.log1p(-self.market_q))) + 1)
                fills = book.market(is_buy, size)
                got = sum(f_[1] for f_ in fills)
                if got:
                    tape.append((t, got if is_buy else -got))
                    self.base.on_external(is_buy, got, t, book)
        side = "bid" if self.flow < 0 else "ask"  # sell pressure thins the bid, buy pressure the ask
        lam_w = ap.w * abs(self.flow) / V_REF
        for k in range(MAX_PER_STEP):
            if u[2 * MAX_PER_STEP + k] < lam_w * DT / MAX_PER_STEP:
                prices = book.prices(side, 3)
                if prices:
                    lv = min(int(u[3 * MAX_PER_STEP + k] * len(prices)), len(prices) - 1)
                    book.cancel_order(side, prices[lv], u[3 * MAX_PER_STEP + k])

    def advance(self, book: Book, t: float, t_until: float, tape: list) -> None:
        while self.clock + DT <= t_until + 1e-12:
            nxt = self.clock + DT
            if nxt > t:
                n0 = len(tape)
                self.base.advance(book, max(t, self.clock), nxt, tape)
                self._step(book, nxt, tape, n0)
            self.clock = nxt
        if t_until > max(t, self.clock):
            self.base.advance(book, max(t, self.clock), t_until, tape)

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        self.base.on_external(is_buy, qty, t, book)


# ------------------------------------------------------------------ simulated method of moments
MOMENTS = ["vr30", "vr120", "vr600", "R100_over_R1", "block_ac25", "depth_flow", "H", "vol5", "spread"]  # vol5, spread: levels


def episode_window(model, duration: float = 1800.0):
    """One observable episode (from t = 0, as the fitting episodes are recorded) as a ten-level LOBSTER window."""
    from ..data import lobster as L
    from ..experiments import goal2_run as G

    book = Book()
    book.tape = []
    model.init_book(book)
    model.advance(book, 0.0, duration, [])
    w = L.tape_to_lobster(book.tape, 1.0, duration - 1.0, price_unit=100)  # the starting book is entered at t = 0
    return G.R.two_sided(G.R.S.visible_window(w))[0]


def moments(windows) -> np.ndarray:
    from ..experiments.goal2_reliability import features

    F = []
    for w in windows:
        f = features(w)
        b = w.book
        ask, bid = b[:, 0].astype(float), b[:, 2].astype(float)
        ok = (ask > 0) & (bid > 0) & (ask < 9e9) & (bid < 9e9)
        t = w.message[:, 0] / 1e9
        g = t[0] + np.arange(0, t[-1] - t[0], 5.0)
        i = np.maximum(np.searchsorted(t, g, side="right") - 1, 0)
        mid = np.where(ok, 0.5 * (ask + bid), np.nan)[i] / 100.0  # ticks
        f["vol5"] = float(np.nanstd(np.diff(mid)))
        f["spread"] = float(np.nanmean(np.where(ok, ask - bid, np.nan)) / 100.0)
        F.append(f)
    X = np.array([[f.get(m, np.nan) for m in MOMENTS] for f in F], float)
    X[:, MOMENTS.index("R100_over_R1")] = np.log(np.clip(X[:, MOMENTS.index("R100_over_R1")], 0.1, 1e4))
    return X


def _theta(z) -> AnchorParams:
    return AnchorParams(f=float(np.exp(z[0])), p_exp=float(1.0 + 1.5 / (1.0 + np.exp(-z[1]))), sigma_f=float(np.exp(z[2])), w=float(np.exp(z[3])))


def fit_smm(base_factory, data_windows, n_rep: int = 2, seed0: int = 70_000, maxfev: int = 80) -> tuple[AnchorParams, dict]:
    """Minimise the weighted distance between the data's mean features and the corrected model's (common random numbers)."""
    from scipy.optimize import minimize

    Xd = moments(data_windows)
    md = np.nanmean(Xd, axis=0)
    sd = np.nanstd(Xd, axis=0, ddof=1) / np.sqrt(np.isfinite(Xd).sum(axis=0))
    n_ep = len(data_windows) * n_rep
    seeds = [seed0 + k for k in range(n_ep)]

    def sim(ap):
        return moments([episode_window(AnchoredHawkes(base_factory(s), ap, s)) for s in seeds])

    Xs0 = sim(AnchorParams())
    s_sim = np.nanstd(Xs0, axis=0, ddof=1) / np.sqrt(np.isfinite(Xs0).sum(axis=0))
    wvec = 1.0 / np.maximum(sd ** 2 + s_sim ** 2, 1e-8)
    hist = []

    def obj(z):
        ms = np.nanmean(sim(_theta(z)), axis=0)
        d = np.nan_to_num((ms - md) ** 2 * wvec, nan=0.0)
        val = float(d.sum())
        hist.append(val)
        return val

    z0 = np.array([np.log(0.1), 0.0, np.log(0.2), 0.0])
    res = minimize(obj, z0, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 0.05, "fatol": 0.05, "initial_simplex": z0 + np.vstack([np.zeros(4), np.eye(4) * 1.5])})
    ap = _theta(res.x)
    return ap, {"objective": float(res.fun), "objective_start": hist[0] if hist else np.nan, "n_eval": len(hist), "data_moments": md.tolist(),
                "moments": MOMENTS, "params": ap.__dict__}
