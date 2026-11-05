"""
Is each simulator's cost of selling at once realistic? Measured directly on real data.

Selling Q shares immediately walks the bid side of the book. On real held-out windows we take the book at many random
moments and compute the cost of selling Q into it (volume-weighted fill price against the mid, in bps), with Q equal
to the parent order of the slice. This is the mechanical cost of the immediate strategy before any reaction by other
traders, which is exactly what each simulator's "immediate" strategy pays at its first decision. Simulated values are
the immediate strategy's mean shortfall over seeds. Hidden liquidity and other venues are ignored, so the real figure
is an upper bound on what a real trader would pay on Nasdaq alone.

    SIMREL_SYMBOL=CSCO python simrel/experiments/slice_15_immediate_cost.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402

VERSIONS = ["hawkes", "hawkes_sd_q_c", "hawkes_sd_c_f_m", "abides", "abides_quoting"]


def walk_cost(row: np.ndarray, q: float) -> float:
    """Cost in bps of selling q shares into a ten-level book row (bid prices at 2, 6, ...; sizes at 3, 7, ...)."""
    mid = 0.5 * (row[0] + row[2])
    left, notional = q, 0.0
    for lvl in range(10):
        p, s = row[4 * lvl + 2], row[4 * lvl + 3]
        if p < -L.EMPTY or s <= 0:
            break
        take = min(left, s)
        notional += take * p
        left -= take
        if left <= 0:
            break
    if left > 0:  # deeper than ten levels: charge the remainder at the tenth level (a lower bound)
        notional += left * row[4 * 9 + 2]
    return float((mid - notional / q) / mid * 1e4)


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    info = R.slice_info()["counts"]
    e = pd.read_csv(R.OUT / "policy_episodes.csv")
    e["size_frac"] = e.get("size_frac", 0.1)
    e["size_frac"] = e["size_frac"].fillna(0.1).round(2)
    e = e[e["sim"].isin(VERSIONS) & (e["policy"] == "P1_immediate") & (e["seed"] < 40020)]
    rows = []
    for name in R.usable_slices():
        ws = R.real_windows(name, "test")
        for size in (0.10, 0.30):
            q = max(int(round(size * info[name]["volume_visible"] / 100.0)) * 100, 100)
            costs = []
            for w in ws:
                ok = np.nonzero((w.book[:, 0] < L.EMPTY) & (w.book[:, 2] > -L.EMPTY))[0]
                for i in rng.choice(ok, size=min(200, len(ok)), replace=False):
                    costs.append(walk_cost(w.book[i], q))
            row = {"slice": name, "size_frac": size, "Q": q, "real_immediate_bps": float(np.mean(costs))}
            for v in VERSIONS:
                d = e[(e["sim"] == v) & (e["slice"] == name) & (e["size_frac"] == size)]
                row[v] = d["is_bps"].mean() if len(d) else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(R.OUT / "immediate_cost.csv", index=False)
    s = t.groupby("size_frac")[["real_immediate_bps"] + VERSIONS].mean()
    print(f"{C.SYMBOL}: cost of selling the whole order at once (bps), mean over slices")
    print(s.round(2).to_string())
    ratio = t[VERSIONS].div(t["real_immediate_bps"], axis=0).groupby(t["size_frac"]).median()
    print("simulated / real, median over slices:")
    print(ratio.round(2).to_string())
