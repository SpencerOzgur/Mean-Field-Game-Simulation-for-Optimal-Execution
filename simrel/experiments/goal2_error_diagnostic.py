"""
Why does TWAP beat simulator-based selection? Exploratory diagnostic on the first ten Goal 2 markets (addendum 5).

For every strategy a, simulator s and market-size cell c: predicted gain over TWAP, D_hat = pred(TWAP) - pred(a), against
the true gain D = true(TWAP) - true(a). Three candidate explanations are separated:

  systematic bias    the error D_hat - D is shared by the simulators: its mean over simulators per (cell, strategy)
                     explains most of its variance, so averaging simulators cannot remove it
  ranking errors     simulators order the strategies wrongly (rank correlation of predicted and true costs over the
                     12 strategies), independently of any selection
  optimizer's curse  the chosen strategy's error is larger than the typical strategy's error: selection picks the
                     most flattering mistakes

    python simrel/experiments/goal2_error_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_run as G  # noqa: E402

DISCOVERY = G.WORLDS + G.NEW_WORLDS  # the first ten markets; the sampled markets are not read here
OUT = G.OUT / "diagnostic"


def long_table() -> pd.DataFrame:
    t = pd.read_csv(G.OUT / "goal2_simulators.csv")
    t = t[t["world"].isin(DISCOVERY)]
    rows = []
    for (w, s), g in t.groupby(["world", "size_frac"]):
        tr = G.truth(w)
        tc = tr[tr["size_frac"].round(2) == round(s, 2)].set_index("policy")["true_cost"]
        for _, r in g.iterrows():
            for a in tc.index:
                p = r.get(f"pred_{a}", np.nan)
                rows.append({"world": w, "size": s, "cell": f"{w}|{s}", "sim": r["sim"], "strategy": a, "pick": r["pick"] == a,
                             "pred_cost": p, "true_cost": tc[a], "pred_gain": r["pred_P0_twap"] - p, "true_gain": tc["P0_twap"] - tc[a],
                             "regret": tc[a] - tc.min()})
    d = pd.DataFrame(rows).dropna(subset=["pred_cost"])
    d["error"] = d["pred_gain"] - d["true_gain"]  # positive: the simulator over-promises the strategy's gain over TWAP
    return d


def main() -> None:
    from scipy.stats import spearmanr

    d = long_table()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "gain_errors.csv", index=False)
    pd.set_option("display.width", 220)
    nt = d[d["strategy"] != "P0_twap"]

    # 1. per strategy: bias, spread, and how often simulators pick it against how often it is truly best
    best = d.loc[d.groupby(["cell", "sim"])["true_cost"].idxmin()].drop_duplicates("cell")["strategy"].value_counts()
    per = nt.groupby("strategy").agg(mean_pred_gain=("pred_gain", "mean"), mean_true_gain=("true_gain", "mean"), bias=("error", "mean"),
                                     rmse=("error", lambda e: float(np.sqrt(np.mean(e ** 2)))), picked=("pick", "sum"))
    per["truly_best_in_cells"] = best.reindex(per.index).fillna(0).astype(int)
    per = per.sort_values("bias", ascending=False)
    per.to_csv(OUT / "per_strategy.csv")
    print("1. predicted against true gain over TWAP, by strategy (bps; error = predicted - true)")
    print(per.round(2).to_string())

    # 2. how much of the error is shared by the simulators (a (cell, strategy) effect that averaging cannot remove)
    shared = nt.groupby(["cell", "strategy"])["error"].transform("mean")
    total_var = float(nt["error"].var())
    share = float(shared.var() / total_var)
    print(f"\n2. error variance explained by the (market, strategy) mean over simulators: {share:.0%} (shared); idiosyncratic {1 - share:.0%}")
    fam = nt.assign(fam=np.where(nt["world"].str.startswith("R1"), "queue-reactive", "agent-mix"))
    fam["shared"] = shared
    for f, g in fam.groupby("fam"):
        print(f"   {f}: shared {float(g['shared'].var() / g['error'].var()):.0%}, mean |error| {g['error'].abs().mean():.2f} bps")

    # 3. ranking accuracy, before any selection
    rho = d.groupby(["cell", "sim"]).apply(lambda g: spearmanr(g["pred_cost"], g["true_cost"]).correlation, include_groups=False)
    print(f"\n3. rank correlation of predicted and true costs over the 12 strategies: median {rho.median():.2f}, "
          f"share above 0.5: {(rho > 0.5).mean():.0%}, share below 0: {(rho < 0).mean():.0%}")
    print("   by family:", rho.groupby(lambda k: "queue-reactive" if k[0].startswith("R1") else "agent-mix").median().round(2).to_dict())

    # 4. selection: error of the chosen strategy against the typical non-TWAP strategy of the same simulator and cell
    chosen = nt[nt["pick"]]
    typ = nt.groupby(["cell", "sim"])["error"].mean()
    amp = chosen.set_index(["cell", "sim"])["error"] - typ.reindex(chosen.set_index(["cell", "sim"]).index)
    print(f"\n4. error of the chosen strategy {chosen['error'].mean():+.2f} bps against {typ.mean():+.2f} for the typical strategy: "
          f"selection adds {amp.mean():+.2f} bps of over-promise")

    # 5. counterfactual picks: remove the shared bias (oracle), or rank correctly but keep each simulator's choice set
    d2 = d.copy()
    d2["pred_debiased"] = d2["pred_cost"] - d2.groupby(["cell", "strategy"])["pred_cost"].transform("mean") + d2.groupby(["cell", "strategy"])["true_cost"].transform("mean")
    pick_db = d2.loc[d2.groupby(["cell", "sim"])["pred_debiased"].idxmin()]
    pick_raw = d2[d2["pick"]]
    twap = d2[d2["strategy"] == "P0_twap"].drop_duplicates("cell")["regret"]
    print(f"\n5. mean regret: simulators as they are {pick_raw['regret'].mean():.2f}; with the shared bias removed (oracle) {pick_db['regret'].mean():.2f}; "
          f"always TWAP {twap.mean():.2f}")


if __name__ == "__main__":
    main()
