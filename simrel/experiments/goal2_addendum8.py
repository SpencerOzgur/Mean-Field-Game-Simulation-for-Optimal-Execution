"""
Design addendum 8: steps 1, 2 and 4 (simrel/reports/lob_impact_design_addendum8.md).

decompose  Four decision makers per market: A the oracle with 100 seeds, B the oracle with 30, C each candidate with
           100, D each candidate with 30 (B and D: the mean over three disjoint 30-seed blocks, seeds 0-29, 30-59,
           60-89, so that one arbitrary block does not decide the comparison). Sampling contribution B - A, structural
           C - A, combined D - A and the interaction D - B - C + A, by market group, with 95% bootstrap intervals
           (markets, designs, seeds). Regret uses the full truth (300 seeds); a cross-fitted version (best strategy
           identified on truth seeds 0-149, costs evaluated on 150-299) is reported beside it.
decision   E_decision, the mean absolute error over the 66 pairs of strategies of the predicted cost difference, and its
           two-pair proxy (selling at once against TWAP), with split seeds: measures from seeds 50-99, recommendation
           (hence regret) from seeds 0-49. Rank correlation with regret, beside LOB-Bench, LOB-Impact and the
           metaorder measure (also from seeds 50-99), and the paired difference E_decision - metaorder, three-level
           bootstrap.

    SIMREL_POLICY_SEEDS=100 python simrel/experiments/goal2_addendum8.py decompose
    SIMREL_POLICY_SEEDS=100 python simrel/experiments/goal2_addendum8.py decision
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SIMREL_POLICY_SEEDS", "100")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_cluster_bootstrap as B  # noqa: E402
from simrel.experiments import goal2_run as G  # noqa: E402

OUT = G.BASE / "addendum8"
N_BOOT = int(os.environ.get("SIMREL_N_BOOT", 1000))
BLOCKS = [range(0, 30), range(30, 60), range(60, 90)]
TWAP, IMM = "P0_twap", "P1_immediate"


def group(w: str) -> str:
    if w.startswith("W"):
        return "well-specified"
    if w.startswith("R3"):
        return "held-out latent"
    return "queue-reactive" if w.startswith("R1") else "agent-mix"


def truth_tables(worlds) -> dict:
    """(world, size) -> (policies, mean cost, cross-fit (best on seeds < 150, costs on seeds >= 150), true TWAP peak)."""
    out = {}
    for w in worlds:
        r = pd.DataFrame(G.load(G.REF / f"ref_{w}" / "rows_reference.pkl"))
        r = r[r["failed"].isna() | (r["failed"] == False)]  # noqa: E712
        for size, g in r.groupby("size_frac"):
            size = round(size, 2)
            m = g.groupby("policy")["is_bps"].mean().sort_index()
            a = g[g["seed"] < 150].groupby("policy")["is_bps"].mean().reindex(m.index)
            b = g[g["seed"] >= 150].groupby("policy")["is_bps"].mean().reindex(m.index)
            peak = g[g["policy"] == TWAP]["disp_end_bps"].mean()
            out[(w, size)] = (list(m.index), m.to_numpy(), a.idxmin(), b, peak)
    return out


def seed_matrix(w: str, name: str, size: float):
    """Per-seed costs (seed index x policy, NaN where a seed is missing), policies sorted by name, and TWAP peaks."""
    rows = []
    for s in range(100):
        p = G.RAW / "policy" / f"{w}_{name}_{G.SEED_POL + s}.pkl"
        if p.exists():
            rows += [dict(r, seed_index=s) for r in G.load(p) if not r.get("failed") and round(r["size_frac"], 2) == size]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    cost = d.pivot_table(index="seed_index", columns="policy", values="is_bps").reindex(range(100)).sort_index(axis=1)
    peak = d[d["policy"] == TWAP].set_index("seed_index")["disp_end_bps"].reindex(range(100))
    return list(cost.columns), cost.to_numpy(), peak.to_numpy()


def pick(pols, cost, idx) -> str | None:
    c = cost[list(idx)]
    if np.all(np.isnan(c)):
        return None
    m = np.round(np.nanmean(c, axis=0), 9)
    return pols[int(np.nanargmin(m))]


def regret_of(truth, key, p, crossfit=False) -> float:
    pols, mean, best_a, b, _ = truth[key]
    if p is None:
        return np.nan
    if crossfit:
        return float(b[p] - b[best_a])
    i = pols.index(p)
    return float(mean[i] - mean.min())


def load_all(worlds):
    names = list(G.family())
    mats = {}
    for w in worlds:
        for size in (0.05, 0.15):
            for n in names + ["oracle"]:
                m = seed_matrix(w, n, size)
                if m is not None:
                    mats[(w, n, size)] = m
    return names, mats


def decompose() -> None:
    worlds = [w for w in G.worlds_with_results() if not w.endswith("_long")]
    truth = truth_tables(worlds)
    names, mats = load_all(worlds)
    groups = sorted({group(w) for w in worlds})

    def stats(rng, boot: bool, crossfit: bool) -> dict:
        res = {}
        for gname in groups:
            ws = [w for w in worlds if group(w) == gname]
            ws = list(rng.choice(ws, len(ws))) if boot else ws
            ds = list(rng.choice(names, len(names))) if boot else names
            A, Bv, C, D = [], [], [], []
            for w in ws:
                for size in (0.05, 0.15):
                    key = (w, size)
                    if key not in truth or (w, "oracle", size) not in mats:
                        continue
                    seeds = rng.integers(0, 100, 100) if boot else np.arange(100)
                    pols, cost, _ = mats[(w, "oracle", size)]
                    A.append(regret_of(truth, key, pick(pols, cost, seeds), crossfit))
                    Bv.append(np.nanmean([regret_of(truth, key, pick(pols, cost, [seeds[i] for i in blk]), crossfit) for blk in BLOCKS]))
                    for n in ds:
                        if (w, n, size) not in mats:
                            continue
                        pols, cost, _ = mats[(w, n, size)]
                        C.append(regret_of(truth, key, pick(pols, cost, seeds), crossfit))
                        D.append(np.nanmean([regret_of(truth, key, pick(pols, cost, [seeds[i] for i in blk]), crossfit) for blk in BLOCKS]))
            a, b, c, d = (np.nanmean(x) if len(x) else np.nan for x in (A, Bv, C, D))
            res.update({f"{gname}|A oracle 100": a, f"{gname}|B oracle 30": b, f"{gname}|C candidates 100": c, f"{gname}|D candidates 30": d,
                        f"{gname}|sampling B-A": b - a, f"{gname}|structural C-A": c - a, f"{gname}|combined D-A": d - a,
                        f"{gname}|interaction D-B-C+A": d - b - c + a})
        return res

    OUT.mkdir(parents=True, exist_ok=True)
    for crossfit in (False, True):
        rng = np.random.default_rng(2026110801)
        est = stats(rng, False, crossfit)
        bs = pd.DataFrame([stats(rng, True, crossfit) for _ in range(N_BOOT)])
        t = pd.DataFrame({"estimate": pd.Series(est), "lo": bs.quantile(0.025), "hi": bs.quantile(0.975)})
        t.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in t.index], names=["group", "quantity"])
        tag = "crossfit" if crossfit else "full_truth"
        t.to_csv(OUT / f"decomposition_{tag}.csv")
        print(f"\nfour decision makers ({'cross-fitted truth' if crossfit else 'full truth'}), mean regret in bps, 95% intervals")
        print(t.round(2).to_string())


def decision() -> None:
    from scipy.stats import spearmanr  # noqa: F401  (ranks are taken within cells below)

    worlds = [w for w in G.worlds_with_results() if not w.endswith("_long")]
    truth = truth_tables(worlds)
    names, mats = load_all(worlds)
    li = pd.read_csv(G.BASE / "lob_impact" / "goal2_with_lob_impact.csv")
    fixed = li.groupby(["world", "sim"])[["order_book", "price", "impact", "li_peak_error", "li_flow_error"]].first()
    SEL, MEAS = np.arange(0, 50), np.arange(50, 100)

    def rows_for(rng, boot):
        ws = list(rng.choice(worlds, len(worlds))) if boot else worlds
        ds = list(rng.choice(names, len(names))) if boot else names
        rows = []
        for wi, w in enumerate(ws):
            for size in (0.05, 0.15):
                key = (w, size)
                if key not in truth:
                    continue
                tp, tmean, _, _, tpeak = truth[key]
                tc = pd.Series(tmean, index=tp)
                for n in ds:
                    if (w, n, size) not in mats or (w, n) not in fixed.index:
                        continue
                    pols, cost, peak = mats[(w, n, size)]
                    sel = rng.choice(SEL, len(SEL)) if boot else SEL
                    mea = rng.choice(MEAS, len(MEAS)) if boot else MEAS
                    p = pick(pols, cost, sel)
                    pm = pd.Series(np.nanmean(cost[mea], axis=0), index=pols)
                    common = [x for x in pols if x in tc.index]
                    pairs = list(itertools.combinations(common, 2))
                    e_dec = float(np.mean([abs((pm[i] - pm[j]) - (tc[i] - tc[j])) for i, j in pairs]))
                    e_proxy = float(abs((pm[IMM] - pm[TWAP]) - (tc[IMM] - tc[TWAP])))
                    rows.append({"cell": f"{wi}_{size}", "group": group(w), "regret": regret_of(truth, key, p), "e_decision": e_dec, "e_proxy_immediate_twap": e_proxy,
                                 "metaorder_impact_error": float(abs(np.nanmean(peak[mea]) - tpeak)), **fixed.loc[(w, n)].to_dict()})
        return pd.DataFrame(rows)

    measures = ["e_decision", "e_proxy_immediate_twap", "metaorder_impact_error", "order_book", "price", "impact", "li_peak_error", "li_flow_error"]

    def stats(df):
        res = {}
        for gname, d in [("all", df)] + list(df.groupby("group")):
            for m in measures:
                res[f"{gname}|rho {m}"] = B.spearman_within(d, m)
            res[f"{gname}|rho e_decision - rho metaorder"] = res[f"{gname}|rho e_decision"] - res[f"{gname}|rho metaorder_impact_error"]
            for m in ("e_decision", "e_proxy_immediate_twap", "metaorder_impact_error", "order_book"):
                sel = d.loc[d.groupby("cell")[m].idxmin()]
                res[f"{gname}|regret choosing by {m}"] = float(sel["regret"].mean())
        return res

    rng = np.random.default_rng(2026110802)
    est = stats(rows_for(rng, False))
    bs = pd.DataFrame([stats(rows_for(rng, True)) for _ in range(N_BOOT)])
    t = pd.DataFrame({"estimate": pd.Series(est), "lo": bs.quantile(0.025), "hi": bs.quantile(0.975)})
    t.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in t.index], names=["group", "quantity"])
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "decision_aware_metric.csv")
    pd.set_option("display.width", 200)
    print("decision-aware validation (split seeds: measures from seeds 50-99, recommendations from 0-49)")
    print(t.round(3).to_string())


if __name__ == "__main__":
    {"decompose": decompose, "decision": decision}[sys.argv[1]]()
