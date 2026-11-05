"""
Metaorder impact measured on our own ITCH data with synthetic metaorders (exploratory, 4 November 2026).

Method of Maitrier, Loeper and Bouchaud (arXiv:2503.18199, Algorithms 1-2): every aggressive order of the day (visible
and hidden executions with the same time stamp and resting side form one order) is assigned at random to one of
N_TRADERS synthetic traders with equal probability; a metaorder is a run of same-sign orders of one trader within the
session (at least two children). Impact I = sign * (p_e - p_s) / p_s, with p_s the mid just before the first child and
p_e the mid just before the next aggressive order after the last child; it is normalised by the day's range
sigma_D = (max p - min p) / p_0 and the size by the day's volume V_D: I / sigma_D = Y (Q / V_D)^delta.

The fit gives a measured impact law for each stock, against which the simulators' TWAP impact (Goal 1, 10% and 30%
of the slice's ten-minute volume) is compared instead of the literature band used in slice_14.

    python simrel/experiments/slice_17_synthetic_metaorders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402

N_TRADERS = 20
N_ASSIGN = 5  # independent random assignments, pooled
SESSION = (9.75 * 3600, 15.75 * 3600)
BINS = np.logspace(-6, -2, 17)


def aggressive_orders(m: np.ndarray, b: np.ndarray):
    """Per aggressive order: row of its first fill, sign (+1 buyer-initiated), shares; and the mid after each row."""
    ex = np.flatnonzero((m[:, 1] == 4) | (m[:, 1] == 5))
    key = np.r_[True, (np.diff(m[ex, 0]) != 0) | (np.diff(m[ex, 5]) != 0)]
    grp = np.cumsum(key) - 1
    first = ex[key]
    size = np.bincount(grp, weights=m[ex, 3])
    sign = -m[first, 5]  # resting side +1 (bid) is hit by a seller
    ask, bid = b[:, 0].astype(float), b[:, 2].astype(float)
    mid = 0.5 * (ask + bid)
    mid[(np.abs(ask) >= L.EMPTY) | (np.abs(bid) >= L.EMPTY)] = np.nan
    return first, sign, size, mid


def metaorders(first, sign, size, mid, rng) -> pd.DataFrame:
    trader = rng.integers(0, N_TRADERS, len(first))
    rows = []
    for t in range(N_TRADERS):
        idx = np.flatnonzero(trader == t)
        if len(idx) < 2:
            continue
        s = sign[idx]
        cut = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1, len(idx)]
        for a, z in zip(cut[:-1], cut[1:]):
            if z - a < 2:
                continue
            j0, j1 = idx[a], idx[z - 1]
            if j1 + 1 >= len(first):
                continue
            ps, pe = mid[first[j0] - 1], mid[first[j1 + 1] - 1]  # mid before the first child; before the next order after the last
            if np.isfinite(ps) and np.isfinite(pe) and ps > 0:
                rows.append({"Q": size[idx[a:z]].sum(), "n": z - a, "I": s[a] * (pe - ps) / ps})
    return pd.DataFrame(rows)


def main() -> None:
    out, fits = [], []
    for sym in ("INTC", "CSCO", "MSFT"):
        rng = np.random.default_rng(20261104)
        for d in C.FIT_DAYS + C.TEST_DAYS:
            m, b = L.load_day(C.DATA_DIR, sym, d)
            t = m[:, 0] / L.NS
            k = (t >= SESSION[0]) & (t < SESSION[1])
            m, b = m[k], b[k]
            first, sign, size, mid = aggressive_orders(m, b)
            px = m[(m[:, 1] == 4) | (m[:, 1] == 5), 4]
            sigma, V = (px.max() - px.min()) / px[0], size.sum()
            for _ in range(N_ASSIGN):
                mo = metaorders(first, sign, size, mid, rng)
                mo["q"], mo["y"], mo["stock"], mo["day"] = mo["Q"] / V, mo["I"] / sigma, sym, d
                mo["sigma_D"], mo["V_D"] = sigma, V
                out.append(mo)
        print(f"{sym}: done", flush=True)
    a = pd.concat(out)
    for sym, g in a.groupby("stock"):
        g = g[g["q"] > 5e-6]  # the paper's noisy region below 5e-6 is left out
        cat = pd.cut(g["q"], BINS)
        bm = g.groupby(cat, observed=True).agg(q=("q", "mean"), y=("y", "mean"), n=("y", "size"))
        bm = bm[(bm["n"] >= 200) & (bm["y"] > 0)]
        slope, icpt = np.polyfit(np.log(bm["q"]), np.log(bm["y"]), 1)
        y_sqrt = float(np.median(bm["y"] / np.sqrt(bm["q"])))  # prefactor if the exponent is fixed at one half
        fits.append({"stock": sym, "n_metaorders": len(g), "delta": slope, "Y_at_fitted_delta_q1e-3": float(np.exp(icpt) * 1e-3 ** slope / np.sqrt(1e-3)),
                     "Y_sqrt_law": y_sqrt, "q_range": f"{bm['q'].min():.1e}-{bm['q'].max():.1e}"})
    f = pd.DataFrame(fits)
    dst = ROOT / "results" / "simrel" / "slices"
    f.to_csv(dst / "synthetic_metaorders_fit.csv", index=False)
    a.groupby(["stock", "day"])[["sigma_D", "V_D"]].first().to_csv(dst / "synthetic_metaorders_daily.csv")
    print(f.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
