"""
Does improving a simulator's realism change its execution decisions? Analysis of the Hawkes version ladder.

    python simrel/experiments/slice_05_ladder.py

All versions are compared on the same four policies and the same 20 seeds (40000-40019) per slice.
Decision statistic: cost of TWAP minus cost of immediate execution (positive = immediate is cheaper), paired by seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel import slices as S  # noqa: E402
from simrel.configs import slices as C  # noqa: E402
from simrel.metrics.cost import boot_ci  # noqa: E402

OUT = S.out_dir()
POL = C.VERSION_POLICIES
FLAGS = ["same_day", "queue", "close"]

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    e = pd.read_csv(OUT / "policy_episodes.csv")
    e = e[e["sim"].str.startswith("hawkes") & e["policy"].isin(POL) & (e["seed"] < 40000 + C.N_VERSION_SEEDS)]
    e = e[e["failed"].isna() | (e["failed"] == False)]  # noqa: E712
    real = pd.read_csv(OUT / "realism.csv")
    stat = pd.read_csv(OUT / "realism_by_statistic.csv")
    rows, per_slice = [], []
    for v, flags in C.HAWKES_VERSIONS.items():
        d = e[e["sim"] == v]
        w = d.pivot_table(index=["slice", "seed"], columns="policy", values="is_bps")
        diff = (w["P0_twap"] - w["P1_immediate"]).to_numpy()
        lo, hi = boot_ci(diff)
        means = d.groupby(["slice", "policy"])["is_bps"].mean().unstack()
        best = means.idxmin(axis=1)
        for name, b in best.items():
            dd = (w.loc[name, "P0_twap"] - w.loc[name, "P1_immediate"]).to_numpy()
            l2, h2 = boot_ci(dd)
            per_slice.append({"version": v, "slice": name, "best": b, "twap_minus_immediate": dd.mean(), "lo": l2, "hi": h2,
                              "l1": float(real[(real["source"] == v) & (real["slice"] == name)]["l1_mean"].iloc[0])})
        r = real[real["source"] == v]
        st = stat[stat["source"] == v].groupby("statistic")["l1"].mean()
        imp = d.groupby("policy")["impact_bps"].mean()
        rows.append({"version": v, **{f: bool(flags.get(f)) for f in FLAGS}, "l1_mean": r["l1_mean"].mean(),
                     "twap_minus_immediate": diff.mean(), "lo": lo, "hi": hi,
                     "slices_best_immediate": int((best == "P1_immediate").sum()), "n_slices": len(best),
                     "best_counts": best.value_counts().to_dict(),
                     **{f"is_{p}": d[d["policy"] == p]["is_bps"].mean() for p in POL},
                     "impact_twap": imp["P0_twap"], "impact_immediate": imp["P1_immediate"],
                     "l1_touch": (st["ask_volume_touch"] + st["bid_volume_touch"]) / 2, "l1_spread": st["spread"],
                     "l1_interarrival": st["log_inter_arrival_time"], "l1_depth": (st["ask_volume"] + st["bid_volume"]) / 2})
    t = pd.DataFrame(rows).sort_values("l1_mean", ascending=False)
    t.to_csv(OUT / "ladder_summary.csv", index=False)
    ps = pd.DataFrame(per_slice)
    ps.to_csv(OUT / "ladder_by_slice.csv", index=False)
    print(t.drop(columns=["best_counts"]).round(2).to_string(index=False))
    print(t[["version", "best_counts"]].to_string(index=False))
    # main effects of each fix on the decision statistic and on realism (2^3 factorial, averaged over slices)
    m = ps.merge(t[["version"] + FLAGS], on="version")
    for f in FLAGS:
        on, off = m[m[f]], m[~m[f]]
        print(f"effect of {f:8s}: realism L1 {on['l1'].mean() - off['l1'].mean():+.3f}   TWAP-minus-immediate {on['twap_minus_immediate'].mean() - off['twap_minus_immediate'].mean():+.2f} bps")
    print("correlation across versions x slices between realism L1 and TWAP-minus-immediate:", round(m[["l1", "twap_minus_immediate"]].corr().iloc[0, 1], 2))
    print(ps.pivot(index="slice", columns="version", values="twap_minus_immediate").round(1).to_string())
