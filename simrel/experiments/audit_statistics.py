"""
Statistical audit requested in review (17 November 2026): re-analysis of existing results only, no new simulation.

A  Test B: cross-fitted regret and its negative values; the four hypotheses under full truth and with the truth's own
   sampling noise added to the market/seed bootstrap.
B  Own executions: market-bootstrap intervals for regret at each budget, the budget at which own data first reaches
   always-TWAP (with an interval), and two standard non-simulator learning baselines (successive halving over the 12
   strategies; a TWAP-versus-challenger A/B test).

    python simrel/experiments/audit_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_decision_estimator as DE  # noqa: E402
from simrel.experiments import goal2_own_execution as OE  # noqa: E402
from simrel.experiments import goal2_reliability as R  # noqa: E402

G = R.G
TWAP = "P0_twap"
OUT = G.BASE / "audit"
H = {"H10.1": ("rule", "always TWAP"), "H10.2": ("rule", "no-simulator"),
     "H10.3": ("corrected", "original h_c_f_m_k1"), "H10.4": ("corrected", "compound Hawkes")}


def test_b_audit(rng) -> pd.DataFrame:
    t = pd.read_csv(G.BASE / "addendum10" / "test_b.csv")
    hyp = pd.read_csv(G.BASE / "addendum10" / "test_b_hypotheses.csv")
    seeds = {w: OE.per_seed(w) for w in R.TEST_B}
    rows = []
    for h, (a, b) in H.items():
        ca = t[t.arm == a].set_index(["world", "size"])
        cb = t[t.arm == b].set_index(["world", "size"])
        d_cf = (ca.regret - cb.regret).groupby(level=0).mean()
        d_full = (ca.regret_full - cb.regret_full).groupby(level=0).mean()
        # the truth's own noise: paired per-seed cost differences of the two chosen strategies on seeds 150-299
        tvar = []
        for w in d_cf.index:
            per = []
            for s in (0.05, 0.15):
                pa, pb = ca.loc[(w, s), "chosen"], cb.loc[(w, s), "chosen"]
                pols, m = seeds[w][s]
                if not isinstance(pa, str) or not isinstance(pb, str):
                    per.append(np.full(150, np.nan))
                    continue
                per.append((m.loc[m.index >= 150, pa] - m.loc[m.index >= 150, pb]).to_numpy())
            dd = np.nanmean(np.vstack(per), axis=0)  # mean over the two sizes, per seed
            tvar.append(np.nanvar(dd, ddof=1) / np.isfinite(dd).sum())
        se_truth = float(np.sqrt(np.sum(tvar)) / len(d_cf))
        r = hyp.set_index(hyp.hypothesis.str[:5]).loc[h]
        se_boot = float((r.ci_hi - r.ci_lo) / 3.92)
        se_all = float(np.sqrt(se_boot ** 2 + se_truth ** 2))
        bs = [d_full.to_numpy()[rng.integers(0, len(d_full), len(d_full))].mean() for _ in range(2000)]
        rows.append({"hypothesis": h, "estimate (cross-fitted)": float(d_cf.mean()), "estimate (full truth)": float(d_full.mean()),
                     "CI reported": f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", "SE reported (market+seed)": se_boot,
                     "SE from truth noise": se_truth, "CI with truth noise (normal)": f"[{d_cf.mean() - 1.96 * se_all:+.2f}, {d_cf.mean() + 1.96 * se_all:+.2f}]",
                     "CI full truth (market only)": f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]",
                     "one-sided p reported": float(r.p_one_sided), "Holm threshold": float(r.holm_threshold)})
    neg = t[t.regret < 0]
    print(f"Test B cross-fitted regret: {len(neg)} of {len(t)} arm-market-size cells negative; min {t.regret.min():.2f} bp; "
          f"full-truth regret min {t.regret_full.min():.2f} bp (never negative by construction)")
    return pd.DataFrame(rows)


def learners(data, rng, budgets=(120, 240, 600, 1200, 1800), draws=200) -> pd.DataFrame:
    """Simple regret of two non-simulator learners for a total budget of own executions."""
    rows = []
    nosim = "P6_signal_reversal"  # the development markets' best strategy at both sizes (frozen in testb_freeze.json)
    for (w, size), (pols, own, truth) in data.items():
        ti, ci = pols.index(TWAP), pols.index(nosim)
        regret = lambda j: float(truth.iloc[j] - truth.min())  # noqa: E731
        for N in budgets:
            sh, ab = [], []
            for _ in range(draws):
                # successive halving: equal shares per round, halve the surviving strategies each round
                alive = list(range(len(pols)))
                used = {j: [] for j in alive}
                perm = {j: rng.permutation(own.shape[0]) for j in alive}
                rounds = int(np.ceil(np.log2(len(alive))))
                for _r in range(rounds):
                    k = max(1, int(N / rounds / len(alive)))
                    for j in alive:
                        start = len(used[j])
                        used[j] += list(own[perm[j][start:start + k], j])
                    means = {j: np.nanmean(used[j]) for j in alive}
                    alive = sorted(alive, key=lambda j: means[j])[: max(1, len(alive) // 2)]
                sh.append(regret(alive[0]))
                # A/B: TWAP against the development-best strategy, N/2 each (capped by the 150 seeds), switch if z > 2
                k = min(N // 2, own.shape[0])
                xa, xb = own[rng.permutation(own.shape[0])[:k], ti], own[rng.permutation(own.shape[0])[:k], ci]
                z = (np.nanmean(xa) - np.nanmean(xb)) / np.sqrt(np.nanvar(xa, ddof=1) / k + np.nanvar(xb, ddof=1) / k)
                ab.append(regret(ci if z > 2 else ti))
            rows.append({"world": w, "family": DE.family(w), "size": size, "N": N, "successive halving": float(np.mean(sh)),
                         "A/B TWAP vs signal reversal": float(np.mean(ab)), "always TWAP": regret(ti)})
    return pd.DataFrame(rows)


def own_execution_audit(rng) -> None:
    for label, fn in (("development", "own_execution.csv"), ("Test A", "own_execution_test_a.csv")):
        t = pd.read_csv(G.BASE / "exploratory" / fn)
        if label == "development":
            t = t[t.world.isin(R.dev_markets())]
        am = t[t.family == "agent-mix"]
        per = am.groupby(["rule", "n", "world"]).regret.mean().unstack("world")
        tw = per.loc[("always TWAP", 0)]
        own = per.loc["own data"]
        W = list(per.columns)
        cross = []
        for _ in range(2000):
            ws = rng.choice(W, len(W))
            hit = [n for n in own.index if own.loc[n, ws].mean() <= tw[ws].mean()]
            cross.append(hit[0] if hit else np.inf)
        cross = np.array(cross)
        print(f"\n{label}, agent-mix ({len(W)} markets): own data reaches always-TWAP at n per strategy (market bootstrap): "
              f"median {np.median(cross)}, share reaching by n=100: {np.mean(np.isfinite(cross)):.0%}, "
              f"5th-95th percentile {np.percentile(cross, 5)}-{np.percentile(cross, 95)}")
        for rule in ("own data", "hybrid (ensemble)"):
            for n in (10, 100):
                v = per.loc[(rule, n)].to_numpy() - tw.to_numpy()
                bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                print(f"   {rule:18s} n={n:3d}: minus TWAP {v.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")


def main() -> None:
    rng = np.random.default_rng(20261117)
    OUT.mkdir(parents=True, exist_ok=True)
    tb = test_b_audit(rng)
    tb.to_csv(OUT / "test_b_audit.csv", index=False)
    pd.set_option("display.width", 250)
    print(tb.round(3).to_string(index=False))
    own_execution_audit(rng)
    data = {}
    for w in R.dev_markets() + R.TEST_A:
        if DE.family(w) != "agent-mix":
            continue
        for size, (pols, m) in OE.per_seed(w).items():
            data[(w, size)] = (pols, m.loc[m.index < 150].to_numpy(), m.loc[m.index >= 150].mean())
    lr = learners(data, rng)
    lr["set"] = np.where(lr.world.isin(R.TEST_A), "Test A", "development")
    lr.to_csv(OUT / "learners.csv", index=False)
    print("\nnon-simulator learners, agent-mix markets: mean regret (bp) by total own executions N")
    print(lr.groupby(["set", "N"])[["successive halving", "A/B TWAP vs signal reversal", "always TWAP"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
