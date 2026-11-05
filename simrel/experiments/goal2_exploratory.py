"""
Exploratory analyses prompted by the literature check (4 November 2026; not pre-registered, run after every Goal 2
result had been seen).

1. split   Separate impact management from alpha capture: the signal strategies (P6_signal_momentum,
           P6_signal_reversal) trade on predicted price moves rather than on the execution schedule. Every analysis is
           repeated with the strategy set restricted to schedules (all strategies except P6), and for comparison with
           the full set.
2. shrink  A Bayesian alternative to choosing the cheapest predicted strategy (Smith and Winkler 2006): each strategy's
           predicted gain over TWAP is shrunk towards a prior learned on the first ten markets,
               posterior = m_a + w_a (D_a - m_a),   w_a = s_a^2 / (s_a^2 + e_a^2),
           m_a, s_a^2 = mean and variance of the strategy's true gain over the training cells, e_a^2 = the ensemble's
           error variance for it (between-simulator variance + seed variance + historical error^2). The strategy with
           the largest positive posterior gain is chosen, else TWAP. Evaluated on the sampled and the well-specified
           markets, which were not used to fit the prior.

    python simrel/experiments/goal2_exploratory.py split
    python simrel/experiments/goal2_exploratory.py shrink
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_cluster_bootstrap as B  # noqa: E402
from simrel.experiments import goal2_rules as RU  # noqa: E402
from simrel.experiments import goal2_run as G  # noqa: E402

OUT = G.OUT / "exploratory"
SIGNAL = ("P6_signal_momentum", "P6_signal_reversal")
N_BOOT = 1000
MIS = G.WORLDS + G.NEW_WORLDS + G.SAMPLED_WORLDS


def restrict(tabs: dict, keep) -> dict:
    out = {}
    for k, (pols, cost, peak) in tabs.items():
        idx = [i for i, p in enumerate(pols) if keep(p)]
        out[k] = ([pols[i] for i in idx], cost[:, idx], peak)
    return out


def truth_for(worlds, keep) -> dict:
    out = {}
    for w in worlds:
        t = G.truth(w)
        for size, g in t.groupby("size_frac"):
            c = g.set_index("policy")["true_cost"]
            c = c[[p for p in c.index if keep(p)]]
            out[(w, round(size, 2))] = {"cost": c, "best": c.min(), "twap_peak": g.set_index("policy").loc["P0_twap", "true_twap_peak"]}
    return out


def split() -> None:
    names, tabs0 = B.seed_tables()
    li = pd.read_csv(B.OUT / "goal2_with_lob_impact.csv")
    fixed = li.groupby(["world", "sim"])[B.FIXED].first()
    rows = []
    for label, keep in [("all strategies", lambda p: True), ("schedules only", lambda p: p not in SIGNAL)]:
        for group, worlds in [("misspecified (20)", MIS), ("well-specified (6)", G.WELL_WORLDS)]:
            tabs = {k: v for k, v in restrict(tabs0, keep).items() if k[0] in worlds}
            truth = truth_for(worlds, keep)
            rng = np.random.default_rng(1)
            est = B.one(rng, names, tabs, truth, fixed, False)
            bs = pd.DataFrame([B.one(rng, names, tabs, truth, fixed, True) for _ in range(N_BOOT)])
            for stat in ["regret_always_twap", "regret_best_metaorder_impact_error", "regret_best_order_book", "diff_metaorder_minus_twap",
                         "rho_metaorder_impact_error", "rho_order_book", "rho_li_peak_error"]:
                rows.append({"strategy set": label, "markets": group, "statistic": stat, "estimate": est[stat],
                             "lo": bs[stat].quantile(0.025), "hi": bs[stat].quantile(0.975)})
            # the ensemble's cheapest strategy against TWAP, paired
            cells = RU.cell_tables(worlds, names, tabs)
            tr = {k: v["cost"] for k, v in truth.items()}
            hist = json.loads((RU.OUT / "rules_frozen.json").read_text())["history"]
            ens = np.mean(list(RU.regret_of("ensemble_raw", 0, cells, tr, hist).values()))
            rows.append({"strategy set": label, "markets": group, "statistic": "ensemble_raw_regret", "estimate": ens, "lo": np.nan, "hi": np.nan})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "split_schedules_signals.csv", index=False)
    t["value"] = t.apply(lambda r: f"{r.estimate:+.2f}" + ("" if np.isnan(r.lo) else f" [{r.lo:+.2f}, {r.hi:+.2f}]"), axis=1)
    pd.set_option("display.width", 220)
    print(t.pivot_table(index="statistic", columns=["markets", "strategy set"], values="value", aggfunc="first").to_string())


def shrink() -> None:
    names, tabs = B.seed_tables()
    train_cells = RU.cell_tables(RU.TRAIN, names, tabs)
    train_truth = RU.truth_costs(RU.TRAIN)
    hist = json.loads((RU.OUT / "rules_frozen.json").read_text())["history"]
    gains = {}
    for key, tc in train_truth.items():
        for a in tc.index:
            gains.setdefault(a, []).append(tc["P0_twap"] - tc[a])
    prior = {a: (float(np.mean(v)), float(np.var(v, ddof=1))) for a, v in gains.items()}
    out = []
    for group, worlds in [("sampled (10, untouched by the prior)", G.SAMPLED_WORLDS), ("well-specified (6)", G.WELL_WORLDS), ("training (10, in-sample)", RU.TRAIN)]:
        cells = RU.cell_tables(worlds, names, tabs)
        truth = RU.truth_costs(worlds)
        reg = {"always TWAP": [], "ensemble, cheapest predicted": [], "ensemble, Bayesian shrinkage": [], "prior only (no simulator)": []}
        picks = []
        for key, cell in cells.items():
            tc = truth[key]
            pols, g, var_b, var_seed = RU.ensemble_gain(cell)
            d = np.nanmean(g, axis=0)
            m = np.array([prior[a][0] for a in pols]); s2 = np.array([prior[a][1] for a in pols])
            e2 = var_b + var_seed + np.array([hist[a]["h"] for a in pols]) ** 2
            post = m + s2 / (s2 + e2) * (d - m)
            post[np.array(pols) == "P0_twap"] = 0.0
            m0 = np.where(np.array(pols) == "P0_twap", 0.0, m)
            choice = lambda v: pols[int(np.nanargmax(v))] if np.nanmax(v) > 0 else "P0_twap"
            r = lambda a: float(tc[a] - tc.min())
            raw = np.where(np.array(pols) == "P0_twap", 0.0, d)
            reg["always TWAP"].append(r("P0_twap"))
            reg["ensemble, cheapest predicted"].append(r(choice(raw)))
            reg["ensemble, Bayesian shrinkage"].append(r(choice(post)))
            reg["prior only (no simulator)"].append(r(choice(m0)))
            picks.append(choice(post))
        out.append({"markets": group, **{k: float(np.mean(v)) for k, v in reg.items()},
                    "shrinkage picks other than TWAP": f"{sum(p != 'P0_twap' for p in picks)} of {len(picks)}"})
    t = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "bayesian_shrinkage.csv", index=False)
    pd.set_option("display.width", 220)
    print("mean regret (bps)")
    print(t.round(2).to_string(index=False))
    print("\nprior (training markets): mean true gain over TWAP and its s.d., by strategy")
    print(pd.DataFrame(prior, index=["mean", "var"]).T.assign(sd=lambda x: np.sqrt(x["var"]))[["mean", "sd"]].round(2).to_string())


if __name__ == "__main__":
    {"split": split, "shrink": shrink}[sys.argv[1]]()
