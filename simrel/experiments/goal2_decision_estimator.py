"""
Exploratory: a direct estimator of decision-critical responses (simrel/reports/exploratory_decision_estimator.md).

Observable features of one three-hour sample -> each strategy's true cost difference against TWAP, one gradient-boosted
regression per strategy; leave one market family out.

    python simrel/experiments/goal2_decision_estimator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_addendum8 as A  # noqa: E402
from simrel.experiments import goal2_reliability as R  # noqa: E402

G = A.G
TWAP = "P0_twap"


def family(w: str) -> str:
    if w.startswith("W"):
        return "Hawkes-built"
    if w.startswith(("R1", "Xqr", "TBq")):
        return "queue-reactive"
    if w.startswith(("R3", "TBl")):
        return "latent"
    return "agent-mix"


def _tb_feature(w: str) -> str:
    path = R.OUT / f"{w}_market_0.pkl"
    if path.exists():
        return "cached"
    rec = R.features(R.sample_window(w, "market", R.SEED_MKT))
    rec.update(world=w, name="market", k=0, seed=R.SEED_MKT)
    G.save(path, rec)
    return w


def markets() -> dict:
    """world -> feature dict, for every market with a feature sample and a viable truth."""
    from simrel.reference_worlds import identgrid

    out = {}
    for w in R.dev_markets() + R.TEST_A + R.TEST_B:
        p = R.OUT / f"{w}_market_0.pkl"
        if p.exists():
            out[w] = G.load(p)
    for w in identgrid.markets():
        p = R.IDENT / f"{w}_3h_0.pkl"
        if p.exists() and (G.REF / f"ref_{w}" / "rows_reference.pkl").exists():
            out[w] = G.load(p)
    return {w: f for w, f in out.items() if "failed" not in f}


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    G.R.pool_map(_tb_feature, [w for w in R.TEST_B if not (R.OUT / f"{w}_market_0.pkl").exists()], 6, "Test B features")
    feats = markets()
    truth = A.truth_tables(list(feats))
    rows = []
    for (w, size), (pols, mean, *_r) in truth.items():
        c = pd.Series(mean, index=pols)
        rows.append({"world": w, "family": family(w), "size": size, **{f: feats[w].get(f, np.nan) for f in R.FEATURES},
                     **{f"d_{p}": c[p] - c[TWAP] for p in pols if p != TWAP}, **{f"c_{p}": c[p] for p in pols}})
    d = pd.DataFrame(rows)
    strategies = [c[2:] for c in d.columns if c.startswith("d_")]
    X_cols = R.FEATURES + ["size"]
    out = []
    for held in sorted(d.family.unique()):
        tr, te = d[d.family != held], d[d.family == held].copy()
        pred = pd.DataFrame(index=te.index)
        for s in strategies:
            m = HistGradientBoostingRegressor(random_state=0).fit(tr[X_cols], tr[f"d_{s}"])
            pred[s] = m.predict(te[X_cols])
        best = pred.min(axis=1)
        te["chosen"] = np.where(best < 0, pred.idxmin(axis=1), TWAP)
        nos = {size: min([TWAP] + strategies, key=lambda p: (tr[tr["size"] == size][f"c_{p}"].mean() - tr[tr["size"] == size][[f"c_{q}" for q in [TWAP] + strategies]].min(axis=1).mean(), p))
               for size in (0.05, 0.15)}
        cmin = te[[f"c_{p}" for p in [TWAP] + strategies]].min(axis=1)
        te["estimator"] = [te.loc[i, f"c_{te.loc[i, 'chosen']}"] - cmin[i] for i in te.index]
        te["always TWAP"] = te[f"c_{TWAP}"] - cmin
        te["no-simulator"] = [te.loc[i, f"c_{nos[round(te.loc[i, 'size'], 2)]}"] - cmin[i] for i in te.index]
        te["oracle-best"] = 0.0
        te["departed"] = te["chosen"] != TWAP
        out.append(te[["world", "family", "size", "chosen", "estimator", "always TWAP", "no-simulator", "departed"]])
    t = pd.concat(out)
    res = G.BASE / "exploratory"
    res.mkdir(parents=True, exist_ok=True)
    t.to_csv(res / "decision_estimator.csv", index=False)
    rng = np.random.default_rng(20261116)
    print(f"{t.world.nunique()} markets with features and truth: {t.groupby('family').world.nunique().to_dict()}")
    print("\nheld-out family: mean regret (bp), departure rate, estimator minus TWAP and minus no-simulator [95% market bootstrap]")
    for fam, g in t.groupby("family"):
        per = g.groupby("world")[["estimator", "always TWAP", "no-simulator", "departed"]].mean()
        line = f"  {fam:15s} n={len(per):3d}  estimator {per.estimator.mean():6.2f}  TWAP {per['always TWAP'].mean():6.2f}  no-sim {per['no-simulator'].mean():6.2f}  departs {per.departed.mean():.0%}"
        for ref in ("always TWAP", "no-simulator"):
            dd = (per.estimator - per[ref]).to_numpy()
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            line += f" | minus {ref}: {dd.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"
        print(line)


if __name__ == "__main__":
    main()
