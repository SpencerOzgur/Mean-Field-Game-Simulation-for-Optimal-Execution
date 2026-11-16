"""
Exploratory: own executions alone and with a simulator prior (simrel/reports/exploratory_own_execution.md).

    python simrel/experiments/goal2_own_execution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_budget as GB  # noqa: E402
from simrel.experiments import goal2_decision_estimator as DE  # noqa: E402
from simrel.experiments import goal2_reliability as R  # noqa: E402

G = R.G
TWAP = "P0_twap"
NS = (2, 5, 10, 20, 50, 100)
DRAWS = 200


def per_seed(w: str) -> dict:
    """size -> (strategies, cost matrix seeds x strategies) from the market's reference runs."""
    r = pd.DataFrame(G.load(G.REF / f"ref_{w}" / "rows_reference.pkl"))
    r = r[r["failed"].isna() | (r["failed"] == False)]  # noqa: E712
    out = {}
    for size, g in r.groupby("size_frac"):
        m = g.pivot_table(index="seed", columns="policy", values="is_bps").sort_index(axis=1)
        out[round(size, 2)] = (list(m.columns), m)
    return out


def sim_pred(w: str, size: float, design: str | list) -> pd.Series | None:
    ds = [design] if isinstance(design, str) else design
    preds = []
    for d in ds:
        m = GB.matrix(w, d, size, 100)
        if m is not None and len(m[0]):
            preds.append(pd.Series(np.nanmean(m[1], axis=0), index=m[0]))
    return pd.concat(preds, axis=1).mean(axis=1) if preds else None


def main() -> None:
    test_a = "--test-a" in sys.argv  # replication: prior variance from the development markets, ensemble hybrid only
    worlds = R.dev_markets() + (R.TEST_A if test_a else [])
    rng = np.random.default_rng(20261116)
    data, sims = {}, {}
    for w in worlds:
        for size, (pols, m) in per_seed(w).items():
            own = m.loc[m.index < 150]
            truth = m.loc[m.index >= 150].mean()
            data[(w, size)] = (pols, own.to_numpy(), truth)
            for lab, des in (("ensemble", R.DESIGNS), ("corrected", "h_anchor")):
                p = sim_pred(w, size, des)
                if p is not None:
                    sims[(w, size, lab)] = (p - p[TWAP]).reindex(pols)
    # prior variance: the simulator's mean squared error in the cost difference to TWAP on the other markets (same size)
    err = {}
    for (w, size, lab), d in sims.items():
        pols, _, truth = data[(w, size)]
        err[(w, size, lab)] = float(np.nanmean(((d - (truth - truth[TWAP])).drop(TWAP)) ** 2))
    rows = []
    for (w, size), (pols, own, truth) in data.items():
        if test_a and w not in R.TEST_A:
            continue
        ti = pols.index(TWAP)
        regret = lambda j: float(truth.iloc[j] - truth.min())  # noqa: E731
        base = {"world": w, "family": DE.family(w), "size": size}
        rows.append(dict(base, rule="always TWAP", n=0, regret=regret(ti)))
        for lab in ("ensemble", "corrected"):
            if (w, size, lab) in sims:
                rows.append(dict(base, rule=f"simulator ({lab})", n=0, regret=regret(int(np.nanargmin(sims[(w, size, lab)].to_numpy())))))
        for n in NS:
            acc = {k: [] for k in ("own data", "own data, gated", "hybrid (ensemble)", "hybrid (corrected)")}
            for _ in range(DRAWS):
                idx = np.stack([rng.choice(own.shape[0], n, replace=False) for _ in pols], axis=1)  # independent days per strategy
                x = np.take_along_axis(own, idx, axis=0)
                mu, var = np.nanmean(x, axis=0), np.nanvar(x, axis=0, ddof=1) if n > 1 else np.full(len(pols), np.nan)
                acc["own data"].append(regret(int(np.nanargmin(mu))))
                d_own = mu - mu[ti]
                v_own = (var + var[ti]) / n
                se = np.sqrt(v_own)
                j = int(np.nanargmin(d_own))
                acc["own data, gated"].append(regret(j if j != ti and -d_own[j] > 2 * se[j] else ti))
                for lab in ("ensemble", "corrected"):
                    if (w, size, lab) not in sims:
                        continue
                    dev = set(R.dev_markets())
                    tau2 = np.nanmean([e for (w2, s2, l2), e in err.items() if w2 != w and w2 in dev and s2 == size and l2 == lab])
                    prior = sims[(w, size, lab)].to_numpy()
                    post = (prior / tau2 + d_own / np.maximum(v_own, 1e-9)) / (1 / tau2 + 1 / np.maximum(v_own, 1e-9))
                    post[ti] = 0.0
                    acc[f"hybrid ({lab})"].append(regret(int(np.nanargmin(post))))
            for k, v in acc.items():
                if v:
                    rows.append(dict(base, rule=k, n=n, regret=float(np.mean(v))))
    t = pd.DataFrame(rows)
    out = G.BASE / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / ("own_execution_test_a.csv" if test_a else "own_execution.csv"), index=False)
    per = t.groupby(["family", "rule", "n", "world"]).regret.mean().groupby(["family", "rule", "n"]).mean().unstack("n")
    pd.set_option("display.width", 220)
    print("mean regret (bp) by family, rule and number of own executions per strategy (n = 0: no own data)")
    print(per.round(2).to_string())
    print("\nown executions per strategy (12 strategies) at which each rule first reaches always-TWAP's regret")
    for fam, g in per.groupby(level=0):
        tw = g.loc[(fam, "always TWAP"), 0]
        reach = {}
        for rule in g.index.get_level_values(1):
            if rule.startswith(("own", "hybrid")):
                row = g.loc[(fam, rule)]
                hit = [n for n in NS if row.get(n, np.nan) <= tw]
                reach[rule] = hit[0] if hit else f">{NS[-1]}"
        print(f"  {fam:15s} TWAP {tw:.2f}: {reach}")


if __name__ == "__main__":
    main()
