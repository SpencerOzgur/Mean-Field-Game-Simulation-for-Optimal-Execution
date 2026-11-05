"""
Analyses of addendum 6, experiments 1 and 2.

oracle    The exact generator of each reference market, run through the candidate pipeline: regret, share of cells
          where it picks the true best or within 0.25 bps, rank correlation of predicted and true costs, at 30 and
          100 seeds; the pass criterion of the addendum; for the well-specified markets, the candidate with the
          generating design but refitted parameters.
ablation  For each mechanism knob: the true cost of every strategy at each level, the optimal strategy, and the
          decision-misspecification matrix (regret in market j of the strategy optimal in market i).

    python simrel/experiments/goal2_oracle_ablation.py oracle
    python simrel/experiments/goal2_oracle_ablation.py ablation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_run as G  # noqa: E402

OUT = G.OUT / "addendum6"
GEN_DESIGN = {"W1_hk1_R1a": "h_k1", "W2_hcfm_R1a": "h_c_f_m_k1", "W3_hk1_R1b": "h_k1", "W4_hcfm_R1b": "h_c_f_m_k1", "W5_hk1_R2a": "h_k1", "W6_hcfm_R2a": "h_c_f_m_k1"}


def family(w: str) -> str:
    return "Hawkes-built" if w.startswith("W") else ("queue-reactive" if w.startswith("R1") else "agent-mix")


def oracle() -> None:
    from scipy.stats import spearmanr

    sim = pd.read_csv(G.OUT / "goal2_simulators.csv")
    rows = []
    for w in G.worlds_with_results():
        tr = G.truth(w)
        recs = []
        for s in range(100):
            p = G.RAW / "policy" / f"{w}_oracle_{G.SEED_POL + s}.pkl"
            if p.exists():
                recs += [dict(r, seed_index=s) for r in G.load(p) if not r.get("failed")]
        if not recs:
            continue
        d = pd.DataFrame(recs)
        for size, g in d.groupby("size_frac"):
            size = round(size, 2)
            tc = tr[tr["size_frac"].round(2) == size].set_index("policy")["true_cost"]
            for n_seeds in (30, 100):
                m = g[g["seed_index"] < n_seeds].groupby("policy")["is_bps"].mean().round(9).sort_index()
                pick = m.idxmin()
                reg = float(tc[pick] - tc.min())
                common = m.index.intersection(tc.index)
                rows.append({"world": w, "family": family(w), "size_frac": size, "seeds": n_seeds, "pick": pick, "best": tc.idxmin(), "regret": reg,
                             "near_best": reg <= 0.25, "rank_corr": spearmanr(m[common], tc[common]).correlation,
                             "twap_regret": float(tc["P0_twap"] - tc.min()), "n_seeds_found": int(g["seed_index"].nunique())})
            if w in GEN_DESIGN:  # both seed counts of this cell get the refitted generator's regret
                row = sim[(sim["world"] == w) & (sim["sim"] == GEN_DESIGN[w]) & (sim["size_frac"].round(2) == size)]
                if len(row):
                    for r in rows[-2:]:
                        r["refitted_generator_regret"] = float(row["regret"].iloc[0])
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "oracle.csv", index=False)
    pd.set_option("display.width", 220)
    s = t.groupby(["seeds", "family"]).agg(oracle_regret=("regret", "mean"), twap_regret=("twap_regret", "mean"), near_best=("near_best", "mean"),
                                           median_rank_corr=("rank_corr", "median"), cells=("regret", "size"))
    s["oracle_over_twap"] = s["oracle_regret"] / s["twap_regret"]
    print("oracle: the market's own generator, true parameters, through the candidate pipeline")
    print(s.round(3).to_string())
    s30 = s.loc[30]
    ok = bool((s30["oracle_regret"] <= 0.5).all() and (s30["oracle_over_twap"] <= 0.25).all() and (t[t.seeds == 30]["rank_corr"].median() >= 0.8))
    print(f"\npass criterion (30 seeds: regret <= 0.5 and <= 1/4 of TWAP in every family; median rank correlation >= 0.8): {'PASS' if ok else 'FAIL'}")
    w = t[(t.seeds == 30) & t.world.isin(GEN_DESIGN)]
    if "refitted_generator_regret" in w:
        print("\nwell-specified markets, 30 seeds: oracle (true parameters) against the generating design refitted from data")
        print(w[["world", "size_frac", "regret", "refitted_generator_regret", "twap_regret"]].round(2).to_string(index=False))


def ablation() -> None:
    from simrel.reference_worlds import ablation as AB

    mk = AB.markets()
    costs = {}
    for name in mk:
        p = G.REF / f"ref_{name}" / "rows_reference.pkl"
        if not p.exists():
            continue
        r = pd.DataFrame(G.load(p))
        r = r[r["failed"].isna() | (r["failed"] == False)]  # noqa: E712
        costs[name] = r.groupby(["size_frac", "policy"])["is_bps"].mean()
    rows, mats = [], []
    for knob, (fam, levels, base) in AB.KNOBS.items():
        names = [AB.name(knob, lv) for lv in levels]
        if not all(n in costs for n in names):
            print(f"{knob}: incomplete")
            continue
        for size in (0.05, 0.15):
            c = pd.DataFrame({lv: costs[n].xs(size, level=0) for lv, n in zip(levels, names)})  # policy x level
            opt = c.idxmin()
            for lv in levels:
                rows.append({"knob": knob, "family": fam, "size_frac": size, "level": lv, "is_base": lv == base, "optimal": opt[lv],
                             "twap_regret": c.loc["P0_twap", lv] - c[lv].min(), "immediate_minus_twap": c.loc["P1_immediate", lv] - c.loc["P0_twap", lv],
                             "reversal_minus_twap": c.loc["P6_signal_reversal", lv] - c.loc["P0_twap", lv], "spread_of_costs": c[lv].max() - c[lv].min()})
            for i, li_ in enumerate(levels):  # decide as if the knob were at level i, pay in the market at level j
                for j, lj in enumerate(levels):
                    mats.append({"knob": knob, "size_frac": size, "believed": li_, "true": lj, "steps_off": abs(i - j),
                                 "regret": c.loc[opt[li_], lj] - c[lj].min()})
    t, m = pd.DataFrame(rows), pd.DataFrame(mats)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "ablation_levels.csv", index=False)
    m.to_csv(OUT / "ablation_misspecification.csv", index=False)
    pd.set_option("display.width", 220)
    print("true optimal strategy and cost gaps along each mechanism (bps)")
    print(t.round(2).to_string(index=False))
    one = m[m["steps_off"] == 1].groupby(["knob", "size_frac"])["regret"].max()
    print("\nlargest regret from misjudging a mechanism by one level (a factor of about 2), bps; 'changes the decision' if above 1 bp")
    print(one.round(2).to_string())
    print("\nmean regret by how far the belief is from the truth")
    print(m.groupby(["knob", "steps_off"])["regret"].mean().unstack().round(2).to_string())


if __name__ == "__main__":
    {"oracle": oracle, "ablation": ablation}[sys.argv[1]]()
