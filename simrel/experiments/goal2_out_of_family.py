"""
Out-of-family validation of decision rules (simrel/reports/lob_impact_design_addendum6.md, experiment 3).

Families: queue-reactive (R1*), agent-mix (R2*), Hawkes-built (W*). For each ordered (train, test) pair the gated
ensemble rule (historical errors, biases and k chosen on the training family, as in goal2_rules.train) and the Bayesian
shrinkage rule (prior from the training family) are applied to the test family without retuning. Selection by
metaorder impact accuracy needs no training and is reported per test family. Intervals: three-level bootstrap.

    python simrel/experiments/goal2_out_of_family.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_cluster_bootstrap as B  # noqa: E402
from simrel.experiments import goal2_rules as RU  # noqa: E402
from simrel.experiments import goal2_run as G  # noqa: E402

N_BOOT = 1000
TWAP = "P0_twap"


def families() -> dict:
    ws = G.worlds_with_results()
    return {"queue-reactive": [w for w in ws if w.startswith("R1")], "agent-mix": [w for w in ws if w.startswith("R2")], "Hawkes-built": [w for w in ws if w.startswith("W")]}


def train(worlds, names, tabs) -> dict:
    cells, truth = RU.cell_tables(worlds, names, tabs), RU.truth_costs(worlds)
    hist = RU.history(cells, truth)
    res = {k: float(np.mean(list(RU.regret_of("ensemble", k, cells, truth, hist).values()))) for k in RU.K_GRID}
    best = min(res.values())
    k_star = max(k for k, v in res.items() if v <= best + 1e-12)
    gains = {}
    for tc in truth.values():
        for a in tc.index:
            gains.setdefault(a, []).append(tc[TWAP] - tc[a])
    prior = {a: (float(np.mean(v)), float(np.var(v, ddof=1))) for a, v in gains.items()}
    return {"hist": hist, "k": k_star, "prior": prior}


def shrink_regret(cells, truth, hist, prior) -> float:
    out = []
    for key, cell in cells.items():
        tc = truth[key]
        pols, g, var_b, var_seed = RU.ensemble_gain(cell)
        d = np.nanmean(g, axis=0)
        m = np.array([prior[a][0] for a in pols]); s2 = np.array([prior[a][1] for a in pols])
        e2 = var_b + var_seed + np.array([hist[a]["h"] for a in pols]) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            post = np.where(s2 + e2 > 0, m + s2 / (s2 + e2) * (d - m), m)
        post[np.array(pols) == TWAP] = 0.0
        pick = pols[int(np.nanargmax(post))] if np.nanmax(post) > 0 else TWAP
        out.append(float(tc[pick] - tc.min()))
    return float(np.mean(out))


def evaluate(test, names, tabs0, model, rng, boot: bool) -> dict:
    tabs = {k: v for k, v in tabs0.items() if k[0] in test}
    truth = RU.truth_costs(test)
    ws = sorted({k[0] for k in tabs}); ws = list(rng.choice(ws, len(ws))) if boot else ws
    ds = list(rng.choice(names, len(names))) if boot else names
    cells, tr = {}, {}
    for wi, w in enumerate(ws):
        for size in (0.05, 0.15):
            cell = {}
            for j, n in enumerate(ds):
                if (w, n, size) not in tabs:
                    continue
                pols, cost, _ = tabs[(w, n, size)]
                c = cost[rng.integers(0, len(cost), len(cost))] if boot else cost
                gain = c[:, [pols.index(TWAP)]] - c
                cell[f"{n}#{j}"] = (pols, np.round(np.nanmean(c, 0), 9), np.nanstd(gain, 0, ddof=1) / np.sqrt(len(c)), len(c))
            cells[(str(wi), size)] = cell
            tr[(str(wi), size)] = truth[(w, size)]
    twap = float(np.mean([v[TWAP] - v.min() for v in tr.values()]))
    gated = float(np.mean(list(RU.regret_of("ensemble", model["k"], cells, tr, model["hist"]).values())))
    shr = shrink_regret(cells, tr, model["hist"], model["prior"])
    return {"always_twap": twap, "gated": gated, "shrinkage": shr, "gated_minus_twap": gated - twap, "shrinkage_minus_twap": shr - twap}


def main() -> None:
    names, tabs = B.seed_tables()
    fam = families()
    models = {f: train(ws, names, tabs) for f, ws in fam.items()}
    rows = []
    for ftr, m in models.items():
        print(f"trained on {ftr}: k = {m['k']}", flush=True)
        for fte, test in fam.items():
            if fte == ftr:
                continue
            rng = np.random.default_rng(11)
            est = evaluate(test, names, tabs, m, rng, False)
            bs = pd.DataFrame([evaluate(test, names, tabs, m, rng, True) for _ in range(N_BOOT)])
            for stat in est:
                rows.append({"train": ftr, "test": fte, "k": m["k"], "statistic": stat, "estimate": est[stat], "lo": bs[stat].quantile(0.025), "hi": bs[stat].quantile(0.975)})
    # selection by metaorder accuracy: no training
    li = pd.read_csv(B.OUT / "goal2_with_lob_impact.csv")
    fixed = li.groupby(["world", "sim"])[B.FIXED].first()
    for fte, test in fam.items():
        t2 = {k: v for k, v in tabs.items() if k[0] in test}
        truth = {}
        for w in test:
            t = G.truth(w)
            for size, g in t.groupby("size_frac"):
                c = g.set_index("policy")["true_cost"]
                truth[(w, round(size, 2))] = {"cost": c, "best": c.min(), "twap_peak": g.set_index("policy").loc[TWAP, "true_twap_peak"]}
        rng = np.random.default_rng(12)
        est = B.one(rng, names, t2, truth, fixed, False)
        bs = pd.DataFrame([B.one(rng, names, t2, truth, fixed, True) for _ in range(N_BOOT)])
        for stat in ("regret_best_metaorder_impact_error", "diff_metaorder_minus_twap"):
            rows.append({"train": "none", "test": fte, "k": np.nan, "statistic": stat, "estimate": est[stat], "lo": bs[stat].quantile(0.025), "hi": bs[stat].quantile(0.975)})
    t = pd.DataFrame(rows)
    out = G.OUT / "rules"
    t.to_csv(out / "out_of_family.csv", index=False)
    t["value"] = t.apply(lambda r: f"{r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", axis=1)
    pd.set_option("display.width", 220)
    print(t.pivot_table(index=["train", "test"], columns="statistic", values="value", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
