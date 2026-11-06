"""
Addendum 7, analyses 1 and 5: paired ensemble and single-simulator regret minus always TWAP by market family (three-level
bootstrap), and the share of single-simulator regret beyond the oracle's (misspecification).

    SIMREL_POLICY_SEEDS=100 SIMREL_ALLOW_MISSING_SEEDS=1 SIMREL_GOAL2_TAG=seeds86 python simrel/experiments/goal2_interim_paired.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simrel.experiments import goal2_rules as R  # noqa: E402

G = R.G
hist = json.loads((R.OUT / "rules_frozen.json").read_text())["history"]
names, tabs = R.B.seed_tables()
fams = {"queue-reactive": [w for w in G.worlds_with_results() if w.startswith("R1")], "agent-mix": [w for w in G.worlds_with_results() if w.startswith("R2")],
        "well-specified": G.WELL_WORLDS, "misspecified": G.WORLDS + G.NEW_WORLDS + G.SAMPLED_WORLDS}
o = pd.read_csv(G.BASE / "addendum6" / "oracle.csv")
rows = []
for label, worlds in fams.items():
    tb = {k: v for k, v in tabs.items() if k[0] in worlds}
    truth = R.truth_costs(worlds)
    rng = np.random.default_rng(7)

    def once(boot):
        ws = sorted({k[0] for k in tb}); ws = list(rng.choice(ws, len(ws))) if boot else ws
        ds = list(rng.choice(names, len(names))) if boot else names
        cells, tr = {}, {}
        for wi, w in enumerate(ws):
            for size in (0.05, 0.15):
                cell = {}
                for j, n in enumerate(ds):
                    if (w, n, size) not in tb:
                        continue
                    pols, cost, _ = tb[(w, n, size)]
                    c = cost[rng.integers(0, len(cost), len(cost))] if boot else cost
                    gain = c[:, [pols.index("P0_twap")]] - c
                    cell[f"{n}#{j}"] = (pols, np.round(np.nanmean(c, 0), 9), np.nanstd(gain, 0, ddof=1) / np.sqrt(len(c)), len(c))
                cells[(str(wi), size)] = cell; tr[(str(wi), size)] = truth[(w, size)]
        tw = np.mean([v["P0_twap"] - v.min() for v in tr.values()])
        en = np.mean(list(R.regret_of("ensemble_raw", 0, cells, tr, hist).values()))
        ps = np.mean(list(R.regret_of("per_sim_raw", 0, cells, tr, hist).values()))
        return tw, en, ps, en - tw, ps - tw

    est = once(False); bs = np.array([once(True) for _ in range(1000)])
    q = lambda i: f"{est[i]:+.2f} [{np.percentile(bs[:, i], 2.5):+.2f}, {np.percentile(bs[:, i], 97.5):+.2f}]"
    ow = o[(o.seeds == 100) & o.world.isin(worlds)]
    rows.append({"markets": label, "always TWAP": f"{est[0]:.2f}", "oracle (100)": f"{ow.regret.mean():.2f}", "single simulator": f"{est[2]:.2f}",
                 "ensemble minus TWAP": q(3), "single simulator minus TWAP": q(4),
                 "misspecification share": f"{(est[2] - ow.regret.mean()) / est[2]:.0%}"})
    print(rows[-1], flush=True)
t = pd.DataFrame(rows); t.to_csv(G.OUT / "extras_paired_and_oracle.csv", index=False)
pd.set_option("display.width", 220); print(t.to_string(index=False))
