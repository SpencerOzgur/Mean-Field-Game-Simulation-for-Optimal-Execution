"""
Statistics of the reversal check (simrel/reports/reversal_check_plan.md), identical for real and simulated data.

A segment is one real day (09:45 to 15:45) or one simulated passive episode. Its inputs are

    bid, ask   the quote in force on a 5 s grid (grid point i is at 5 i seconds from the segment start); NaN where a
               side of the book is missing
    trades     array (n, 2): time in seconds from the segment start, signed shares (buyer-initiated positive)
    tick       the tick size in the price units of bid and ask

segment_summary() reduces a segment to sums and per-window arrays, so that bootstraps over days, seeds and markets
only recombine summaries. The strategies are the study's own objects (policies.library.candidate_set) passed through
core.child_quantity; nothing about them is re-implemented here.
"""

from __future__ import annotations

import numpy as np

from ..core import Observation, child_quantity
from ..policies.library import candidate_set

GRID_S, BIN_S, WINDOW_S, N_SLICES, LOT = 5.0, 30.0, 600.0, 20, 100
SIZE_FRAC = 0.15
K_BIN, K_WIN = int(BIN_S / GRID_S), int(WINDOW_S / GRID_S)
POLICIES = {"twap": "P0_twap", "p6": "P6_signal_reversal"}


def _acf_sums(x: np.ndarray) -> tuple[float, float]:
    """Numerator and denominator of realism._acf1 for one series (demeaned within the series); NaN pairs are skipped."""
    x = x - np.nanmean(x)
    ok = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    return float((x[1:][ok] * x[:-1][ok]).sum()), float(np.nansum(x * x))


def bin_flows(trades: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Total and signed traded shares in each 30 s bin [30 j, 30 (j + 1))."""
    vol, signed = np.zeros(n_bins), np.zeros(n_bins)
    if len(trades):
        j = np.floor(trades[:, 0] / BIN_S).astype(int)
        ok = (j >= 0) & (j < n_bins)
        np.add.at(vol, j[ok], np.abs(trades[ok, 1]))
        np.add.at(signed, j[ok], trades[ok, 1])
    return vol, signed


def sell_signal(vol: np.ndarray, signed: np.ndarray, dmid: np.ndarray) -> np.ndarray:
    """P6's signal for a sell parent from one slice's flows and mid change (SignalAware.__call__ with side = sell)."""
    imb = np.divide(signed, vol, out=np.zeros_like(signed, dtype=float), where=vol > 0)
    return 0.5 * (imb + np.sign(dmid))


def backtest(mid: np.ndarray, vol: np.ndarray, signed: np.ndarray, qty: int, side: str = "sell", own_flow: bool = False,
             bid: np.ndarray | None = None, ask: np.ndarray | None = None) -> dict:
    """Impact-free timing cost (bp of the arrival mid) of TWAP and P6_signal_reversal on one window.

    mid[k] is the mid at decision k (k = 0..19); vol[k] and signed[k] are the public flows of the 30 s before
    decision k (index 0 is ignored: the engine shows zeros at the first decision). own_flow adds the trader's own
    previous child to those flows, as the engine does, with no effect on prices.
    """
    pol = candidate_set(SIZE_FRAC)
    n, sgn = N_SLICES, (1.0 if side == "sell" else -1.0)
    bid = mid if bid is None else bid
    ask = mid if ask is None else ask
    out = {}
    for tag, name in POLICIES.items():
        remaining, own_last, cost, state, sched = qty, 0.0, 0.0, {}, []
        for k in range(n):
            mv = 0.0 if k == 0 else float(vol[k])
            sv = 0.0 if k == 0 else float(signed[k])
            if own_flow:
                mv, sv = mv + own_last, sv + (-own_last if side == "sell" else own_last)
            obs = Observation(k=k, n=n, t=k * BIN_S, bid=float(bid[k]), ask=float(ask[k]), bid_depth1=0.0, ask_depth1=0.0, bid_depth5=0.0,
                              ask_depth5=0.0, market_volume_last=mv, signed_volume_last=sv,
                              mid_change_last=0.0 if k == 0 else float(mid[k] - mid[k - 1]), arrival_mid=float(mid[0]), total_qty=qty, side=side)
            q = child_quantity(pol[name], obs, remaining, state, LOT)
            remaining -= q
            own_last = float(q)
            cost += q * sgn * (mid[0] - mid[k])
            sched.append(q)
        out[tag] = cost / (qty * mid[0]) * 1e4
        out[tag + "_sched"] = sched
    return out


def segment_summary(bid: np.ndarray, ask: np.ndarray, trades: np.ndarray, tick: float) -> dict:
    """Sums for S1 to S3, the per-window inputs of S4, and the tick and spread context of one segment."""
    bid, ask = np.asarray(bid, float), np.asarray(ask, float)
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    ok = np.isfinite(mid) & (spread > 0)
    mid = np.where(ok, mid, np.nan)
    n5 = len(mid)
    out = {"n_grid": int(n5), "n_missing": int((~ok).sum())}
    out["s1_num"], out["s1_den"] = _acf_sums(np.diff(mid))
    m30 = mid[::K_BIN]
    d30 = np.diff(m30)
    n_bins = len(d30)
    out["s2_num"], out["s2_den"] = _acf_sums(d30)
    vol, signed = bin_flows(trades, n_bins)
    # S3: signal from bin j - 1, return over bin j in bp of the mid at the start of bin j
    x = sell_signal(vol[:-1], signed[:-1], d30[:-1])
    y = d30[1:] / m30[1:-1] * 1e4
    good = np.isfinite(x) & np.isfinite(y)
    out["s3_x"], out["s3_y"] = x[good].astype(np.float32), y[good].astype(np.float32)
    # S4 inputs: for each 600 s window, the mids at the 20 decisions and the flows of the 30 s before each
    wins = []
    for w in range(n_bins // N_SLICES):
        a = w * N_SLICES
        mk = m30[a:a + N_SLICES]
        if not np.isfinite(mk).all():
            continue
        v = np.r_[0.0, vol[a:a + N_SLICES - 1]]
        s = np.r_[0.0, signed[a:a + N_SLICES - 1]]
        wins.append({"mid": mk, "vol": v, "signed": s, "bid": bid[::K_BIN][a:a + N_SLICES], "ask": ask[::K_BIN][a:a + N_SLICES],
                     "window_volume": float(vol[a:a + N_SLICES].sum())})
    out["windows"] = wins
    out["volume"] = float(vol.sum())
    out["n_bins"] = int(n_bins)
    sp = spread[ok]
    out.update({"mean_mid": float(np.nanmean(mid)), "tick": float(tick), "spread_sum": float(sp.sum()), "spread_n": int(len(sp)),
                "one_tick_n": int((np.abs(sp - tick) < 1e-9 * max(tick, 1.0)).sum())})
    return out


# ---- combining summaries
def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")  # a mid that never moves has no autocorrelation


def s1(segs: list) -> float:
    return _ratio(sum(s["s1_num"] for s in segs), sum(s["s1_den"] for s in segs))


def s2(segs: list) -> float:
    return _ratio(sum(s["s2_num"] for s in segs), sum(s["s2_den"] for s in segs))


def s3(segs: list) -> tuple[float, float]:
    """Pooled OLS slope of the next 30 s return (bp) on the signal, and its t-statistic clustered by segment."""
    x = np.concatenate([s["s3_x"] for s in segs]).astype(float)
    y = np.concatenate([s["s3_y"] for s in segs]).astype(float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return float("nan"), float("nan")
    b = ((x - xm) * (y - ym)).sum() / sxx
    resid = (y - ym) - b * (x - xm)
    g, i = [], 0
    for s in segs:
        n = len(s["s3_x"])
        g.append(((x[i:i + n] - xm) * resid[i:i + n]).sum())
        i += n
    g = np.array(g)
    G = len(g)
    var = (g**2).sum() / sxx**2 * (G / (G - 1) if G > 1 else 1.0)
    return float(b), float(b / np.sqrt(var)) if var > 0 else float("nan")


def s3_slope(segs: list) -> float:
    x = np.concatenate([s["s3_x"] for s in segs]).astype(float)
    y = np.concatenate([s["s3_y"] for s in segs]).astype(float)
    return _ratio(float(((x - x.mean()) * (y - y.mean())).sum()), float(((x - x.mean()) ** 2).sum()))


def parent_qty(segs: list) -> int:
    """15% of the mean traded volume per 600 s window, rounded down to 100 shares."""
    v = [w["window_volume"] for s in segs for w in s["windows"]]
    return max(int(SIZE_FRAC * float(np.mean(v))) // LOT * LOT, LOT)


def window_gaps(seg: dict, qty: int) -> dict:
    """Per window of one segment: P6 minus TWAP timing cost (bp), sell, sell with own flow in the signal, and buy."""
    sell, sell_own, buy = [], [], []
    for w in seg["windows"]:
        a = backtest(w["mid"], w["vol"], w["signed"], qty, "sell", False, w["bid"], w["ask"])
        b = backtest(w["mid"], w["vol"], w["signed"], qty, "sell", True, w["bid"], w["ask"])
        c = backtest(w["mid"], w["vol"], w["signed"], qty, "buy", False, w["bid"], w["ask"])
        sell.append(a["p6"] - a["twap"])
        sell_own.append(b["p6"] - b["twap"])
        buy.append(c["p6"] - c["twap"])
    return {"s4": np.array(sell), "s4b": np.array(sell_own), "s4_buy": np.array(buy)}


def context(segs: list) -> dict:
    mid = float(np.mean([s["mean_mid"] for s in segs]))
    tick = segs[0]["tick"]
    sp = sum(s["spread_sum"] for s in segs) / sum(s["spread_n"] for s in segs)
    return {"tick_bp": tick / mid * 1e4, "spread_ticks": sp / tick, "spread_bp": sp / mid * 1e4,
            "share_one_tick": sum(s["one_tick_n"] for s in segs) / sum(s["spread_n"] for s in segs), "mean_mid": mid,
            "volume_per_600s": float(np.mean([w["window_volume"] for s in segs for w in s["windows"]]))}
