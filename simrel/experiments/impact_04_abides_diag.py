"""
Why ABIDES shows no LOB-Impact response (simrel/reports/lob_impact_design_addendum2.md, section 2).

Per simulator and build-up arm (INTC, rollouts 0-23 per side): peak impact after 100 children and its standard error,
the mechanical part (mid jump across each child, summed over the metaorder) and the part between children.

    python simrel/experiments/impact_04_abides_diag.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "results" / "simrel" / "lob_impact" / "INTC" / "rec"
ARMS = {"beta": "own clock, median child", "beta_t": "real-time spacing, median child", "beta_x5": "own clock, 5x child",
        "beta_t_x5": "real-time spacing, 5x child"}
N = 24


def main() -> None:
    rows = []
    for sim in ("abides", "abides_quoting", "hawkes"):
        for arm, label in ARMS.items():
            peaks, mech, betw, gap_s = [], [], [], []
            for side in ("buy", "sell"):
                for k in range(N):
                    p = REC / sim / f"{arm}_{side}_{k:03d}.pkl"
                    if not p.exists():
                        continue
                    r = pickle.load(open(p, "rb"))
                    if "mid_child" not in r or len(r["mid_child"]) < 100:
                        continue
                    s, ref = r["sign"], r["mid_ref"]
                    peaks.append(s * (r["mid_child"][99] - ref) / ref * 1e4)
                    mech.append(s * np.nansum(r["mid_child"] - r["mid_pre"]) / ref * 1e4)
                    betw.append(s * np.nansum(r["mid_pre"][1:] - r["mid_child"][:-1]) / ref * 1e4)
            if not peaks:
                continue
            pk = np.asarray(peaks)
            empty = int(np.sum(~np.isfinite(pk)))  # one side of the book empty at the last child
            pk = pk[np.isfinite(pk)]
            se = pk.std(ddof=1) / np.sqrt(len(pk))
            rows.append({"simulator": sim, "arm": label, "n": len(pk), "empty_side_at_end": empty, "peak_bps": pk.mean(), "se": se,
                         "detected": abs(pk.mean()) >= 3 * se, "mechanical_bps": np.nanmean(mech), "between_children_bps": np.nanmean(betw)})
    t = pd.DataFrame(rows)
    out = ROOT / "results" / "simrel" / "lob_impact" / "abides_diagnostic.csv"
    t.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
