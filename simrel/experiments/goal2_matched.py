"""
Design addendum 8, steps 3 and 5 (task variant): matched mechanisms with fixed designs, and the 30-minute task.

matched  For each mechanism knob (26 ablation markets) and each fixed design (h_k1, h_c_f_m_k1, abides, qr) at 100
         seeds: (a) rank correlation across the five levels between the design's predicted gain over TWAP and the true
         gain, per strategy; (b) agreement of the design's recommended strategy with the true optimum across levels;
         (c) mean absolute error in the pairwise cost differences; and the pre-stated test of "decision-boundary
         errors" ((b) below 50% while (c) exceeds the design's error at the base market by more than 1 bp).
long     The six held-out markets under the 30-minute task (sizes 2% and 30%): regret of always TWAP, the oracle and
         the four fixed designs, at 100 seeds.

    python simrel/experiments/goal2_matched.py matched
    python simrel/experiments/goal2_matched.py long
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_addendum8 as A  # noqa: E402
from simrel.experiments import goal2_budget as GB  # noqa: E402
from simrel.reference_worlds import ablation as AB  # noqa: E402

G = A.G
OUT = G.BASE / "addendum8"
DESIGNS = ["h_k1", "h_c_f_m_k1", "abides", "qr"]


def predicted(w, design, size):
    m = GB.matrix(w, design, size, 100)
    if m is None:
        return None
    pols, cost = m
    return pd.Series(np.round(np.nanmean(cost, axis=0), 9), index=pols).sort_index()


def matched() -> None:
    from scipy.stats import spearmanr

    rows, summ = [], []
    for knob, (fam, levels, base) in AB.KNOBS.items():
        names = [AB.name(knob, lv) for lv in levels]
        truth = A.truth_tables(names)
        for size in (0.05, 0.15):
            tcs = {lv: pd.Series(truth[(n, size)][1], index=truth[(n, size)][0]) for lv, n in zip(levels, names)}
            true_opt = {lv: tc.idxmin() for lv, tc in tcs.items()}
            for d in DESIGNS:
                preds = {lv: predicted(n, d, size) for lv, n in zip(levels, names)}
                if any(p is None for p in preds.values()):
                    continue
                opt = {lv: p.idxmin() for lv, p in preds.items()}
                agree = float(np.mean([opt[lv] == true_opt[lv] for lv in levels]))
                regret = float(np.mean([tcs[lv][opt[lv]] - tcs[lv].min() for lv in levels]))
                pair_err = {}
                for lv in levels:
                    p, tc = preds[lv], tcs[lv]
                    common = [x for x in p.index if x in tc.index]
                    pair_err[lv] = float(np.mean([abs((p[i] - p[j]) - (tc[i] - tc[j])) for i, j in itertools.combinations(common, 2)]))
                tracks = []
                for a in tcs[base].index:
                    if a == "P0_twap":
                        continue
                    pg = [preds[lv]["P0_twap"] - preds[lv][a] for lv in levels]
                    tg = [tcs[lv]["P0_twap"] - tcs[lv][a] for lv in levels]
                    tracks.append(spearmanr(pg, tg).correlation)
                worst = max(pair_err[lv] for lv in levels if lv != base)
                boundary = agree < 0.5 and worst - pair_err[base] > 1.0
                summ.append({"knob": knob, "family": fam, "size": size, "design": d, "optimum agreement": agree, "regret": regret,
                             "tracks the truth (median rank corr.)": float(np.nanmedian(tracks)), "pair error at base": pair_err[base],
                             "largest pair error off base": worst, "decision-boundary error": boundary,
                             "true optima": "/".join(true_opt[lv].replace("P6_signal_", "sig_") for lv in levels),
                             "design optima": "/".join(opt[lv].replace("P6_signal_", "sig_") for lv in levels)})
    t = pd.DataFrame(summ)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "matched_mechanisms.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["knob", "size", "design", "optimum agreement", "regret", "tracks the truth (median rank corr.)", "pair error at base", "largest pair error off base", "decision-boundary error"]
    print(t[cols].round(2).to_string(index=False))
    print("\nby knob: share of design-size cells with decision-boundary errors; mean optimum agreement; median tracking")
    print(t.groupby("knob").agg(boundary=("decision-boundary error", "mean"), agreement=("optimum agreement", "mean"),
                                tracking=("tracks the truth (median rank corr.)", "median")).round(2).to_string())


def long() -> None:
    worlds = G.LATENT_LONG
    truth = A.truth_tables(worlds)
    rows = []
    for w in worlds:
        for size in (0.02, 0.30):
            key = (w, size)
            if key not in truth:
                continue
            row = {"world": w, "size": size, "always TWAP": A.regret_of(truth, key, "P0_twap")}
            for d in DESIGNS + ["oracle"]:
                p = predicted(w, d, size)
                row[d] = A.regret_of(truth, key, p.idxmin()) if p is not None else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "long_task.csv", index=False)
    pd.set_option("display.width", 200)
    print("30-minute task (60 decisions), held-out markets: regret (bps) at 100 seeds")
    print(t.round(2).to_string(index=False))
    print("\nmean by size")
    print(t.groupby("size").mean(numeric_only=True).round(2).to_string())


if __name__ == "__main__":
    {"matched": matched, "long": long}[sys.argv[1]]()
