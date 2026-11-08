"""
Design addendum 9: regret against simulation budget, and the true generator with one mechanism removed.

budget   Mean regret of the oracle, single candidates, the ensemble's cheapest strategy and always TWAP at n = 5, 10, 20,
         30, 50, 100 decision seeds (mean over the disjoint n-seed blocks of seeds 0-99), by family. Exploratory.
removed  Four agent-mix markets; the oracle and the oracle with withdraw, fund or trend switched off, at n = 10, 30, 100,
         300 seeds: regret, whether the true best is recommended, rank correlation of predicted and true costs, and the
         error in each strategy's predicted gain over TWAP. Hypotheses H9a and H9b of the addendum.

    SIMREL_INCLUDE_HELDOUT=1 python simrel/experiments/goal2_budget.py budget
    python simrel/experiments/goal2_budget.py removed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_addendum8 as A  # noqa: E402

G = A.G
OUT = G.BASE / "addendum9"
REMOVED_MARKETS = ["R2a_am_anchored", "R2b_am_trending", "R2d_am_toxic", "R2s01"]
REMOVED = ["oracle", "oracle-no-withdraw", "oracle-no-fund", "oracle-no-trend"]


def matrix(w, name, size, n_max):
    rows = []
    for s in range(n_max):
        p = G.RAW / "policy" / f"{w}_{name}_{G.SEED_POL + s}.pkl"
        if p.exists():
            rows += [dict(r, seed_index=s) for r in G.load(p) if not r.get("failed") and round(r["size_frac"], 2) == size]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    cost = d.pivot_table(index="seed_index", columns="policy", values="is_bps").reindex(range(n_max)).sort_index(axis=1)
    return list(cost.columns), cost.to_numpy()


def budget() -> None:
    worlds = [w for w in G.worlds_with_results() if not w.endswith("_long")]
    truth = A.truth_tables(worlds)
    names = list(G.family())
    rows = []
    for w in worlds:
        for size in (0.05, 0.15):
            key = (w, size)
            if key not in truth:
                continue
            tw = A.regret_of(truth, key, "P0_twap")
            mats = {n: matrix(w, n, size, 100) for n in names + ["oracle"]}
            for n_seeds in (5, 10, 20, 30, 50, 100):
                blocks = [range(b * n_seeds, (b + 1) * n_seeds) for b in range(100 // n_seeds)]
                orc = [A.regret_of(truth, key, A.pick(*mats["oracle"], blk)) for blk in blocks] if mats["oracle"] else [np.nan]
                single, ens = [], []
                for blk in blocks:
                    means = []
                    for n in names:
                        if mats[n] is None:
                            continue
                        pols, cost = mats[n]
                        single.append(A.regret_of(truth, key, A.pick(pols, cost, blk)))
                        means.append(pd.Series(np.nanmean(cost[list(blk)], axis=0), index=pols))
                    if means:
                        m = pd.concat(means, axis=1).mean(axis=1).round(9).sort_index()
                        ens.append(A.regret_of(truth, key, m.idxmin()))
                rows.append({"world": w, "group": A.group(w), "size": size, "seeds": n_seeds, "always TWAP": tw, "oracle": np.nanmean(orc),
                             "single candidate": np.nanmean(single), "ensemble": np.nanmean(ens)})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "regret_by_budget.csv", index=False)
    s = t.groupby(["group", "seeds"])[["always TWAP", "oracle", "single candidate", "ensemble"]].mean()
    s.to_csv(OUT / "regret_by_budget_summary.csv")
    pd.set_option("display.width", 200)
    print("mean regret (bps) against the number of decision seeds")
    print(s.round(2).to_string())


def removed() -> None:
    from scipy.stats import spearmanr

    truth = A.truth_tables(REMOVED_MARKETS)
    rows, gains = [], []
    for w in REMOVED_MARKETS:
        for size in (0.05, 0.15):
            key = (w, size)
            tp, tmean, _, _, _ = truth[key]
            tc = pd.Series(tmean, index=tp)
            best = tc.idxmin()
            for name in REMOVED:
                m = matrix(w, name, size, 300)
                if m is None:
                    continue
                pols, cost = m
                for n_seeds in (10, 30, 100, 300):
                    blocks = [range(b * n_seeds, (b + 1) * n_seeds) for b in range(300 // n_seeds)]
                    regs, hits, rhos = [], [], []
                    for blk in blocks:
                        p = A.pick(pols, cost, blk)
                        regs.append(A.regret_of(truth, key, p)); hits.append(p == best)
                        pm = pd.Series(np.nanmean(cost[list(blk)], axis=0), index=pols)
                        rhos.append(spearmanr(pm[tc.index], tc).correlation)
                    rows.append({"world": w, "size": size, "simulator": name, "seeds": n_seeds, "regret": float(np.nanmean(regs)),
                                 "picks true best": float(np.mean(hits)), "rank corr": float(np.nanmean(rhos)), "n_blocks": len(blocks),
                                 "seeds found": int(np.isfinite(cost[:, 0]).sum())})
                pm = pd.Series(np.nanmean(cost, axis=0), index=pols)
                for a in tc.index:
                    gains.append({"world": w, "size": size, "simulator": name, "strategy": a,
                                  "predicted gain over TWAP": pm["P0_twap"] - pm[a], "true gain": tc["P0_twap"] - tc[a]})
    t, g = pd.DataFrame(rows), pd.DataFrame(gains)
    g["error"] = g["predicted gain over TWAP"] - g["true gain"]
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "removed_mechanism.csv", index=False)
    g.to_csv(OUT / "removed_mechanism_gains.csv", index=False)
    pd.set_option("display.width", 220)
    print("true generator with one mechanism removed: regret (bps) by decision seeds, mean over 4 markets x 2 sizes")
    print(t.pivot_table(index="simulator", columns="seeds", values="regret").round(2).to_string())
    print("\nshare of blocks recommending the true best")
    print(t.pivot_table(index="simulator", columns="seeds", values="picks true best").round(2).to_string())
    print("\nrank correlation of predicted and true costs")
    print(t.pivot_table(index="simulator", columns="seeds", values="rank corr").round(2).to_string())
    print("\nper market, regret at 100 and 300 seeds")
    print(t[t.seeds.isin([100, 300])].pivot_table(index=["world", "size"], columns=["simulator", "seeds"], values="regret").round(2).to_string())
    h9a = t[t.simulator == "oracle"].groupby("seeds").regret.mean()
    print(f"\nH9a (oracle regret falls towards zero): {h9a.round(2).to_dict()}")
    t3 = t[t.seeds == 300].set_index(["world", "size", "simulator"]).regret
    t1 = t[t.seeds == 100].set_index(["world", "size", "simulator"]).regret
    ok = []
    for w in REMOVED_MARKETS:
        hit = any(t3.get((w, s, n), np.nan) > 1.0 and t3.get((w, s, n), np.nan) >= t1.get((w, s, n), np.nan) - 1e-9
                  for s in (0.05, 0.15) for n in REMOVED[1:])
        ok.append(hit)
    print(f"H9b (some removed mechanism keeps regret above 1 bp at 300 seeds without falling from 100), per market: {dict(zip(REMOVED_MARKETS, ok))}")
    print("\nmean error in predicted gain over TWAP, 300 seeds (positive: over-promised)")
    print(g.pivot_table(index="strategy", columns="simulator", values="error").round(2).to_string())


if __name__ == "__main__":
    {"budget": budget, "removed": removed}[sys.argv[1]]()
