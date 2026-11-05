"""
Bounds on the real cost of selling at once (exploratory, 4 November 2026).

slice_15 walks the visible Nasdaq book, an upper bound: hidden orders also absorb sells (fills from aggregate books
are only partially identified). For a lower bound, every visible level is enlarged by the share of bid-side
executions that hit hidden orders on that stock (hidden_liquidity_share.csv), as if all hidden liquidity sat in the
visible levels in proportion to their size. Same book moments as slice_15 (same random seed).

    SIMREL_SYMBOL=CSCO python simrel/experiments/slice_16_immediate_bounds.py
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
from simrel.experiments.slice_15_immediate_cost import VERSIONS, walk_cost  # noqa: E402

if __name__ == "__main__":
    h = pd.read_csv(ROOT / "results" / "simrel" / "slices" / "hidden_liquidity_share.csv")
    share = float(h[h["stock"] == C.SYMBOL]["hidden_share_bid_side"].mean())
    rng = np.random.default_rng(5)
    info = R.slice_info()["counts"]
    sim = pd.read_csv(R.OUT / "immediate_cost.csv")
    rows = []
    for name in R.usable_slices():
        ws = R.real_windows(name, "test")
        for size in (0.10, 0.30):
            q = max(int(round(size * info[name]["volume_visible"] / 100.0)) * 100, 100)
            up, lo = [], []
            for w in ws:
                ok = np.nonzero((w.book[:, 0] < L.EMPTY) & (w.book[:, 2] > -L.EMPTY))[0]
                for i in rng.choice(ok, size=min(200, len(ok)), replace=False):
                    row = w.book[i].astype(float)
                    up.append(walk_cost(row, q))
                    deep = row.copy()
                    deep[3::4] = deep[3::4] / (1.0 - share)
                    lo.append(walk_cost(deep, q))
            s = sim[(sim["slice"] == name) & (sim["size_frac"].round(2) == size)].iloc[0]
            rows.append({"slice": name, "size_frac": size, "real_upper_bps": np.mean(up), "real_lower_bps": np.mean(lo), **{v: s[v] for v in VERSIONS}})
    t = pd.DataFrame(rows)
    t.to_csv(R.OUT / "immediate_cost_bounds.csv", index=False)
    m = t.groupby("size_frac")[["real_lower_bps", "real_upper_bps"] + VERSIONS].mean()
    print(f"{C.SYMBOL} (bid-side hidden share {share:.0%}): cost of selling at once, bps, mean over slices")
    print(m.round(2).to_string())
    where = {v: np.where(m[v] < m["real_lower_bps"], "below", np.where(m[v] > m["real_upper_bps"], "above", "inside")) for v in VERSIONS}
    print("simulator against the real band:", {v: dict(zip(m.index, w)) for v, w in where.items()})
