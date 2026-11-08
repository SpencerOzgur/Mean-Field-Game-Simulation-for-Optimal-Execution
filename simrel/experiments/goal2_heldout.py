"""
Design addendum 8, step 5: the held-out latent-liquidity family, evaluated without any retuning.

Reported on the six markets R3s01-R3s06 at 100 decision seeds: always TWAP, the oracle, ensemble and single-simulator
regret (and their paired differences from TWAP), the gated rule with its values frozen on the first ten markets at
100 seeds (results/simrel/goal2/seeds100/rules/rules_frozen.json), and the validation profile. Selection by each
measure, the correlations and the step-2 decomposition come from goal2_cluster_bootstrap.py and goal2_addendum8.py run
with SIMREL_INCLUDE_HELDOUT=1 (see the command block at the end of the docstring of run_all).

    SIMREL_POLICY_SEEDS=100 SIMREL_INCLUDE_HELDOUT=1 SIMREL_GOAL2_TAG=heldout python simrel/experiments/goal2_heldout.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SIMREL_POLICY_SEEDS", "100")
os.environ.setdefault("SIMREL_INCLUDE_HELDOUT", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_rules as RU  # noqa: E402

G = RU.G
FROZEN = G.BASE / "seeds100" / "rules" / "rules_frozen.json"
N_BOOT = int(os.environ.get("SIMREL_N_BOOT", 1000))


def paired(worlds, names, tabs, hist) -> pd.DataFrame:
    tb = {k: v for k, v in tabs.items() if k[0] in worlds}
    truth = RU.truth_costs(worlds)
    rng = np.random.default_rng(8)

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
                cells[(str(wi), size)] = cell
                tr[(str(wi), size)] = truth[(w, size)]
        tw = float(np.mean([v["P0_twap"] - v.min() for v in tr.values()]))
        out = {"always TWAP": tw}
        for rule, k in [("ensemble_raw", 0.0), ("per_sim_raw", 0.0), ("ensemble", 3.0), ("ensemble_bc", 3.0), ("per_sim", 3.0), ("ensemble", 1.0), ("per_sim", 1.0)]:
            v = float(np.mean(list(RU.regret_of(rule, k, cells, tr, hist).values())))
            out[f"{rule}{'' if rule.endswith('raw') else f' k={k:g}'}"] = v
            out[f"{rule}{'' if rule.endswith('raw') else f' k={k:g}'} minus TWAP"] = v - tw
        return out

    est = once(False)
    bs = pd.DataFrame([once(True) for _ in range(N_BOOT)])
    return pd.DataFrame({"estimate": pd.Series(est), "lo": bs.quantile(0.025), "hi": bs.quantile(0.975)})


def main() -> None:
    frozen = json.loads(FROZEN.read_text())
    assert frozen["k"] == {"ensemble": 3.0, "ensemble_bc": 3.0, "per_sim": 3.0}, frozen["k"]
    names, tabs = RU.B.seed_tables()
    t = paired(G.LATENT_WORLDS, names, tabs, frozen["history"])
    o = pd.read_csv(G.BASE / "addendum6" / "oracle.csv") if (G.BASE / "addendum6" / "oracle.csv").exists() else None
    out = G.OUT / "heldout"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "heldout_rules_and_ensembles.csv")
    pd.set_option("display.width", 200)
    print("held-out latent-liquidity family (6 markets, 100 seeds), rules frozen on the first ten markets")
    print(t.round(2).to_string())
    # validation profile and the frozen tests of addendum 5, on the held-out markets
    RU.TEST = G.LATENT_WORLDS
    RU.OUT = out
    shutil.copy(FROZEN, out / "rules_frozen.json")
    RU.test(N_BOOT)


if __name__ == "__main__":
    main()
