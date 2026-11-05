"""
Goal 2 intervals that respect the design's dependence (simrel/reports/lob_impact_design_addendum2.md, section 3).

Three-level bootstrap: markets (4) with replacement; simulator designs (29) with replacement, one draw shared by every
market; within each drawn (market, design) its policy seeds with replacement, from which its predicted costs, its
recommendation, its regret and its metaorder measure are recomputed. Realism and LOB-Impact scores are held fixed.

    python simrel/experiments/goal2_cluster_bootstrap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_run as G  # noqa: E402

N_BOOT = 2000
SEED = 20261102
FIXED = ["order_book", "price", "impact", "trade_response_error", "li_flow_error", "li_response_error", "li_peak_error",
         "li_law_error_literature"]
MEASURES = FIXED + ["metaorder_impact_error"]
OUT = G.OUT / "lob_impact"


def seed_tables():
    """(world, design, size) -> (policies, seeds x policies cost matrix, seeds TWAP peak vector)."""
    names = list(G.family())
    out = {}
    for w in G.worlds_with_results():
        for n in names:
            rows = []
            for s in range(G.POLICIES_SEEDS):
                p = G.RAW / "policy" / f"{w}_{n}_{G.SEED_POL + s}.pkl"
                if p.exists():
                    rows += [r for r in G.load(p) if not r.get("failed")]
            if not rows:
                continue
            d = pd.DataFrame(rows)
            for size, g in d.groupby("size_frac"):
                cost = g.pivot_table(index="seed", columns="policy", values="is_bps")
                peak = g[g["policy"] == "P0_twap"].set_index("seed")["disp_end_bps"].reindex(cost.index)
                out[(w, n, round(size, 2))] = (list(cost.columns), cost.to_numpy(), peak.to_numpy())
    return names, out


def spearman_within(df: pd.DataFrame, m: str) -> float:
    d = df.dropna(subset=[m])
    a = d.groupby("cell")[m].rank()
    b = d.groupby("cell")["regret"].rank()
    return float(np.corrcoef(a, b)[0, 1]) if len(d) > 3 else np.nan


def one(rng, names, tabs, truth, fixed, boot: bool) -> dict:
    all_w = sorted({k[0] for k in tabs})
    worlds = list(rng.choice(all_w, len(all_w))) if boot else all_w
    designs = list(rng.choice(names, len(names))) if boot else names
    rows = []
    for wi, w in enumerate(worlds):
        for size in (0.05, 0.15):
            tc = truth[(w, size)]
            for n in designs:
                key = (w, n, size)
                if key not in tabs or (w, n) not in fixed.index:
                    continue
                pols, cost, peak = tabs[key]
                idx = rng.integers(0, len(cost), len(cost)) if boot else np.arange(len(cost))
                means = np.round(np.nanmean(cost[idx], axis=0), 9)  # columns are sorted by name: exact ties go to the first, as in goal2_run
                pick = pols[int(np.nanargmin(means))]
                rows.append({"cell": f"{wi}_{size}", "world": w, "size": size, "sim": n, "regret": tc["cost"][pick] - tc["best"],
                             "metaorder_impact_error": abs(np.nanmean(peak[idx]) - tc["twap_peak"]), "twap_regret": tc["cost"]["P0_twap"] - tc["best"],
                             **fixed.loc[(w, n)].to_dict()})
    df = pd.DataFrame(rows)
    res = {f"rho_{m}": spearman_within(df, m) for m in MEASURES}
    sel = {m: [] for m in MEASURES}
    twap = []
    for _, g in df.groupby("cell"):
        twap.append(g["twap_regret"].iloc[0])
        for m in MEASURES:
            ok = g.dropna(subset=[m])
            sel[m].append(ok.loc[ok[m].idxmin(), "regret"] if len(ok) else np.nan)
    res["regret_always_twap"] = float(np.mean(twap))
    for m in MEASURES:
        res[f"regret_best_{m}"] = float(np.nanmean(sel[m]))
    res["diff_metaorder_minus_twap"] = res["regret_best_metaorder_impact_error"] - res["regret_always_twap"]
    res["diff_metaorder_minus_order_book"] = res["regret_best_metaorder_impact_error"] - res["regret_best_order_book"]
    res["diff_li_peak_minus_order_book"] = res["regret_best_li_peak_error"] - res["regret_best_order_book"]
    return res


def main() -> None:
    names, tabs = seed_tables()
    if "--worlds" in sys.argv:  # a subset of markets, e.g. the original four (reported separately, addendum 3)
        keep = set(sys.argv[sys.argv.index("--worlds") + 1].split(","))
        tabs = {k: v for k, v in tabs.items() if k[0] in keep}
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    truth = {}
    for w in G.worlds_with_results():
        t = G.truth(w)
        for size, g in t.groupby("size_frac"):
            c = g.set_index("policy")["true_cost"]
            truth[(w, round(size, 2))] = {"cost": c, "best": c.min(), "twap_peak": g.set_index("policy").loc["P0_twap", "true_twap_peak"]}
    li = pd.read_csv(OUT / "goal2_with_lob_impact.csv")
    fixed = li.groupby(["world", "sim"])[FIXED].first()
    rng = np.random.default_rng(SEED)
    est = one(rng, names, tabs, truth, fixed, boot=False)
    bs = pd.DataFrame([one(rng, names, tabs, truth, fixed, boot=True) for _ in range(N_BOOT)])
    out = pd.DataFrame({"estimate": pd.Series(est), "lo": bs.quantile(0.025), "hi": bs.quantile(0.975),
                        "share_above_0": (bs > 0).mean()})
    out.to_csv(OUT / f"goal2_cluster_bootstrap{tag}.csv")
    pd.set_option("display.width", 200)
    print(f"three-level bootstrap (markets, simulator designs, seeds), {N_BOOT} resamples")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
