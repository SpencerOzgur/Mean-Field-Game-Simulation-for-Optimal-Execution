"""
Permanent-impact (lambda) agent-mix family (development pilot; simrel/reports/lambda_pilot_rule.md).

In AgentMix the fundamental F is exogenous, so every price move caused by trading fades. LambdaWorld is AgentMix with
one change: executed market orders move F permanently,

    F += lam * signed shares        (buys positive; lam in ticks per share)

Two modes:

    all      every executed market order, background or injected, moves F. This is the family. The world does not use
             identity: the trader's order moves F exactly as a background order of the same size does.
    trader   COMPARISON ARM ONLY. Only injected (external) orders move F. This uses identity, so it is unrealistic,
             and passive data are bit-identical to lam = 0 by construction.

The hook is AgentMix._note_trade, which AgentMix.apply calls once for every background market order that fills and
AgentMix.on_external calls once for every injected order. No random draw is added or changed: the exogenous stream
and the agent streams are those of AgentMix. With lam = 0 the world is bit-identical to AgentMix.

Units. lam is set through lam_bp: the permanent move, in bp of the price, caused by trading 1% of the market's
10-minute reference volume (ticks_per_share converts it for a market).

Volatility matching. In mode "all" the background flow adds variance to F. match_fund_vol lowers fund_vol so that the
variance of 600 s mid returns in passive episodes equals that of lam = 0 (same calibration seeds), within a tolerance.
Windows that move by more than 300 bp are left out of that variance and counted separately (return_var).

Pilot markets: parameter stream numpy default_rng(20261130), names XLP_pilot_*; used once, never reused.
"""

from __future__ import annotations

import copy

import numpy as np

from ..eventsim import GRID_DT
from ..lob import Book
from . import latent, sampled
from .agent_mix import AgentMix

PILOT_SEED = 20261130  # parameter stream of the pilot markets; used once, never reused
PILOT_PREFIX = "XLP_pilot_"
MODES = ("all", "trader")
VOL_HORIZON_S = 600.0
VOL_TOL = 0.05
WILD_TICKS = 30.0  # 300 bp, the wild-move threshold of the viability rule (configs/reference.py)


class Registry:
    """Collects the worlds of one run_twin_set call: the snapshot copy first, then one copy per run, in run order."""

    def __init__(self) -> None:
        self.items: list = []

    def __deepcopy__(self, memo):
        return self


class LambdaWorld(AgentMix):
    name = "lambda_agent_mix"

    def __init__(self, params, seed: int, lam: float = 0.0, mode: str = "all", registry: Registry | None = None) -> None:
        assert mode in MODES
        super().__init__(params, seed)
        self.lam = float(lam)  # ticks per share
        self.mode = mode
        self.n_moves = 0  # times F was moved by a trade
        self.moved_shares = 0.0  # signed shares that moved F
        self.registry = registry

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        if self.registry is not None:
            self.registry.items.append(new)
        return new

    def _move(self, signed_qty: float) -> None:
        if self.lam != 0.0 and signed_qty != 0:
            self.F += self.lam * signed_qty
            self.n_moves += 1
            self.moved_shares += signed_qty

    def _note_trade(self, signed_qty: float) -> None:
        super()._note_trade(signed_qty)
        if self.mode == "all":
            self._move(signed_qty)

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        super().on_external(is_buy, qty, t, book)  # reaches _note_trade once (mode "all" moves F there)
        if self.mode == "trader":
            self._move(qty if is_buy else -qty)


def ticks_per_share(lam_bp: float, reference_volume: float) -> float:
    """lam_bp (bp of price per 1% of the 10-minute reference volume) in ticks per share. One tick is 10 bp."""
    if lam_bp == 0.0:
        return 0.0
    return lam_bp * (AgentMix.p0 / 1e4) / (0.01 * reference_volume)


def make_lambda(values: dict, lam_bp: float = 0.0, reference_volume: float = 1.0, mode: str = "all", fund_vol: float | None = None,
                registry: Registry | None = None):
    """Factory for run_twin_set: seed -> LambdaWorld. fund_vol replaces the market's own value (volatility matching)."""
    vals = dict(values) if fund_vol is None else {**values, "fund_vol": float(fund_vol)}
    lam = ticks_per_share(lam_bp, reference_volume)
    return lambda seed: LambdaWorld(sampled.params("am", vals), seed, lam, mode, registry)


def make_world(family: str, values: dict, **kw):
    """Factory for either family; latent markets have no lambda (they are the reference point)."""
    if family == "am":
        return make_lambda(values, **kw)
    return lambda seed: latent.LatentLiquidity(latent.LatentParams(**values), seed)


def pilot_draws(n_am: int = 12, n_latent: int = 4, n_extra: int = 0) -> dict:
    """The draws of the pilot stream default_rng(PILOT_SEED), in order: a block of n_am agent-mix draws, n_latent
    latent draws, then n_extra further agent-mix draws (the stream continued after the block, numbered on from n_am).

    Names are assigned to every draw, in draw order. Which draws are pilot markets is decided by the selection rule of
    simrel/reports/lambda_pilot_rule.md (simrel/experiments/lambda_pilot.py, lambda_pilot_calib.py).
    """
    rng = np.random.default_rng(PILOT_SEED)
    out = {}
    for k in range(1, n_am + 1):
        out[f"{PILOT_PREFIX}a{k:02d}"] = ("am", sampled.draw(rng, sampled.AM_RANGES))
    for k in range(1, n_latent + 1):
        out[f"{PILOT_PREFIX}l{k:02d}"] = ("latent", latent.draw(rng))
    for k in range(n_am + 1, n_am + n_extra + 1):
        out[f"{PILOT_PREFIX}a{k:02d}"] = ("am", sampled.draw(rng, sampled.AM_RANGES))
    return out


# ---- passive episodes and volatility matching
def passive_mid(factory, seed: int, seconds: float, warmup_s: float = 600.0) -> dict:
    """One passive episode: the mid on the 5 s grid after the warm-up, and the tape and volume of the sample."""
    model = factory(seed)
    book, tape, t = Book(), [], 0.0
    model.init_book(book)
    mids, last = [], float(model.p0)
    n_warm = 0
    while t < warmup_s + seconds - 1e-9:
        if abs(t - warmup_s) < 1e-9:
            n_warm = len(tape)
        if t >= warmup_s - 1e-9:
            last = model.mid(book, last)
            mids.append(last)
        model.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    mids.append(model.mid(book, last))
    return {"mid": np.asarray(mids), "tape": tape[n_warm:], "model": model, "book": book}


def window_moves(mids: list, horizon_s: float = VOL_HORIZON_S, step_s: float = 60.0) -> np.ndarray:
    """Mid changes (ticks) over horizon_s, on overlapping windows step_s apart, pooled over episodes."""
    k, s = int(round(horizon_s / GRID_DT)), int(round(step_s / GRID_DT))
    return np.concatenate([(np.asarray(m, float)[k:] - np.asarray(m, float)[:-k])[::s] for m in mids])


def return_var(mids: list, wild_ticks: float | None = WILD_TICKS) -> float:
    """Mean squared 600 s mid change (ticks squared) over the windows that are not wild.

    A wild window moves by more than 300 bp (30 ticks), the threshold of the existing viability rule. The agent-mix
    markets have rare episodes in which the book empties and the price runs hundreds of ticks; one such episode in 48
    multiplies the plain variance by ten, so the plain variance cannot be matched within 5%. Wild windows are left out
    of the matched variance and their share is reported next to it (wild_share). wild_ticks = None gives the plain
    mean square.
    """
    d = window_moves(mids)
    if wild_ticks is not None:
        d = d[np.abs(d) <= wild_ticks]
    return float((d**2).mean())


def wild_share(mids: list, wild_ticks: float = WILD_TICKS) -> float:
    return float((np.abs(window_moves(mids)) > wild_ticks).mean())


def match_fund_vol(var_of, base_vol: float, target: float, tol: float = VOL_TOL, max_iter: int = 6) -> dict:
    """Find the fund_vol in [0, base_vol] at which var_of(fund_vol) is within tol of target.

    var_of must be deterministic (fixed calibration seeds), so the search is reproducible. The variance is close to
    linear in fund_vol squared, so the search is a secant iteration in x = fund_vol ** 2 between the two bracketing
    points, starting from x = base_vol ** 2 and x = 0. Returns the matched value, the achieved ratio, the evaluations,
    and feasible = False if even fund_vol = 0 leaves the variance above target * (1 + tol).
    """
    evals = []

    def f(x: float) -> float:
        v = var_of(float(np.sqrt(max(x, 0.0))))
        evals.append((float(np.sqrt(max(x, 0.0))), v / target))
        return v / target - 1.0

    def result(x, feasible=True, matched=True):
        ratio = dict(evals)[float(np.sqrt(max(x, 0.0)))]
        return {"fund_vol": float(np.sqrt(max(x, 0.0))), "ratio": ratio, "feasible": feasible, "matched": matched, "evals": list(evals)}

    hi = base_vol**2
    f_hi = f(hi)
    if abs(f_hi) <= tol or f_hi < 0:  # already within tolerance, or below target: fund_vol is not raised
        return result(hi, matched=abs(f_hi) <= tol)
    lo, f_lo = 0.0, f(0.0)
    if f_lo > tol:
        return result(lo, feasible=False, matched=False)
    if abs(f_lo) <= tol:
        return result(lo)
    best = (abs(f_hi), hi) if abs(f_hi) < abs(f_lo) else (abs(f_lo), lo)
    for _ in range(max_iter):
        x = lo - f_lo * (hi - lo) / (f_hi - f_lo)
        x = min(max(x, lo + 0.02 * (hi - lo)), hi - 0.02 * (hi - lo))
        fx = f(x)
        if abs(fx) < best[0]:
            best = (abs(fx), x)
        if abs(fx) <= tol:
            return result(x)
        if fx > 0:
            hi, f_hi = x, fx
        else:
            lo, f_lo = x, fx
    return result(best[1], matched=False)
