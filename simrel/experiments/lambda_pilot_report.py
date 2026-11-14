"""
Tables, figure and verdict of the lambda pilot, from the stored stages of lambda_pilot_run.py.

    E1  costs: true best strategy, always-TWAP regret (plain and cross-fitted), always P6_signal_reversal against the
        best, each strategy against TWAP
    E2  observability: addendum 10 diagnostic features, each arm against lambda = 0; cross-validated AUC with a
        permutation reference (the method of the reactive pilot)
    E3  viability rule at each arm
    E4  TWAP impact persistence at each arm against the literature band
    verdict: the rule of simrel/reports/lambda_pilot_rule.md, applied mechanically
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import reference as C  # noqa: E402
from simrel.experiments import lambda_pilot as P  # noqa: E402
from simrel.experiments import lambda_pilot_calib as K  # noqa: E402
from simrel.experiments import reactive_pilot_report as PR  # noqa: E402
from simrel.experiments.lambda_pilot_run import ARMS  # noqa: E402

OUT = P.OUT
JUDGED = ["all_band_low", "all_max_feasible"]
REGRET_RISE_BP, BEST_SHARE, KEEP_COST_BP = 1.0, 0.5, 0.5  # rule (b); share of cells and cost threshold of rule (a)
BAND = (2.0 / 3.0, 0.87)  # literature values of end over peak (lob_impact_design.md); not measured on our data
REV = "P6_signal_reversal"


def load(name: str):
    return pickle.load(open(P.RAW / f"{name}.pkl", "rb"))


def se(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def fam(m: str) -> str:
    return "latent" if "_l" in m[len(P.LW.PILOT_PREFIX) - 1:] else "agent-mix"


# ---- E3
def e3() -> pd.DataFrame:
    tw = pd.DataFrame(load("twins"))
    rows = []
    for (m, a), d in tw.groupby(["market", "arm"]):
        ok = d[d["failed"].isna()]
        fail = 1.0 - len(ok) / len(d)
        wild = float((ok["move_bps_abs"] > C.WILD_MOVE_BPS).mean()) if len(ok) else 1.0
        rows.append({"market": m, "arm": a, "n": len(d), "fail_share": fail, "wild_share": wild,
                     "viable": bool(fail <= C.MAX_FAIL_SHARE and wild <= C.MAX_WILD_SHARE), "volume": ok["volume"].mean(),
                     "spread_bps": ok["spread_bps"].mean(), "vol_bps": ok["vol_bps"].mean(), "move_bps_abs": ok["move_bps_abs"].mean(),
                     "ret_acf1": ok["ret_acf1"].mean(), "absret_acf1": ok["absret_acf1"].mean(), "kurtosis": ok["kurtosis"].mean()})
    out = pd.DataFrame(rows)
    out["arm"] = pd.Categorical(out["arm"], ARMS, ordered=True)
    return out.sort_values(["market", "arm"])


# ---- E1
def e1(truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = truth[truth["is_bps"].notna()]
    cost = t.pivot_table(index=["market", "arm", "size_frac", "seed"], columns="policy", values="is_bps")
    rows, summ = [], []
    base = {(m, f): d.droplevel([0, 1, 2]).dropna() for (m, a, f), d in cost.groupby(level=[0, 1, 2]) if a == "lam0"}
    for (m, a, f), d in cost.groupby(level=[0, 1, 2]):
        d = d.droplevel([0, 1, 2]).dropna()
        means = d.mean()
        for pol in d.columns:
            rows.append({"market": m, "arm": a, "size_frac": f, "policy": pol, "n": len(d), "cost_bps": means[pol], "cost_se": se(d[pol]),
                         "minus_twap_bps": (d[pol] - d["P0_twap"]).mean(), "minus_twap_se": se(d[pol] - d["P0_twap"])})
        best = means.idxmin()
        order = means.sort_values()
        ha, hb = d[d.index < P.N_TRUTH // 2], d[d.index >= P.N_TRUTH // 2]
        ba, bb = ha.mean().idxmin(), hb.mean().idxmin()
        xfit = 0.5 * ((hb["P0_twap"] - hb[ba]).mean() + (ha["P0_twap"] - ha[bb]).mean())
        b0 = base[(m, f)].mean().idxmin()
        # rule (a): cost of keeping the lambda = 0 best over the arm's best, the arm's best chosen on the other half of the seeds
        keep = 0.5 * ((hb[b0] - hb[ba]).mean() + (ha[b0] - ha[bb]).mean())
        summ.append({"market": m, "family": fam(m), "arm": a, "size_frac": f, "n": len(d), "twap_cost_bps": means["P0_twap"], "twap_cost_se": se(d["P0_twap"]),
                     "best_policy": best, "best_cost_bps": means[best], "second_policy": order.index[1], "second_minus_best_bps": order.iloc[1] - order.iloc[0],
                     "second_minus_best_se": se(d[order.index[1]] - d[best]),
                     "lam0_best_policy": b0, "best_differs_from_lam0": bool(best != b0),
                     "lam0_best_minus_best_bps": means[b0] - means[best], "lam0_best_minus_best_se": se(d[b0] - d[best]),
                     "keep_lam0_best_crossfit_bps": keep, "cell_changed": bool(keep >= KEEP_COST_BP),
                     "twap_regret_bps": means["P0_twap"] - means[best], "twap_regret_se": se(d["P0_twap"] - d[best]),
                     "best_policy_half_a": ba, "best_policy_half_b": bb, "twap_regret_crossfit_bps": xfit,
                     "reversal_minus_best_bps": means[REV] - means[best], "reversal_minus_best_se": se(d[REV] - d[best]),
                     "reversal_minus_twap_bps": means[REV] - means["P0_twap"], "reversal_minus_twap_se": se(d[REV] - d["P0_twap"]),
                     "reversal_rank": int(order.index.get_loc(REV)) + 1, "unfilled_frac": t[(t.market == m) & (t.arm == a) & (t.size_frac == f)]["unfilled_frac"].mean()})
    costs, summ = pd.DataFrame(rows), pd.DataFrame(summ)
    for df in (costs, summ):
        df["arm"] = pd.Categorical(df["arm"], ARMS, ordered=True)
    return costs.sort_values(["market", "size_frac", "arm", "policy"]), summ.sort_values(["market", "size_frac", "arm"])


# ---- verdict
def verdict(summ: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """The rule of lambda_pilot_rule.md. Exclusions were fixed before the run from the lambda = 0 calibration runs
    (calib_exclusion.csv, lambda_pilot_calib.exclusion_table)."""
    ex = pd.read_csv(OUT / "calib_exclusion.csv")
    excluded = sorted(ex[ex["excluded"]]["market"].unique())
    am = summ[summ["family"] == "agent-mix"]
    kept = [m for m in sorted(am["market"].unique()) if m not in excluded]
    base = am[(am["arm"] == "lam0") & am["market"].isin(kept)].set_index(["market", "size_frac"])
    rows, cells = [], []
    for arm in [a for a in ARMS if a != "lam0"]:
        dd = am[(am["arm"] == arm) & am["market"].isin(kept)].set_index(["market", "size_frac"])
        n_cells, n_changed = len(dd), int(dd["cell_changed"].sum())
        # (b): cross-fitted regret averaged over sizes within market, then the mean over markets
        rise_m = (dd["twap_regret_crossfit_bps"] - base["twap_regret_crossfit_bps"]).groupby(level=0).mean()
        plain_m = (dd["twap_regret_bps"] - base["twap_regret_bps"]).groupby(level=0).mean()
        a_ok, b_ok = n_changed >= BEST_SHARE * n_cells, rise_m.mean() >= REGRET_RISE_BP
        rows.append({"arm": arm, "judged_by_rule": arm in JUDGED, "markets": len(kept), "cells": n_cells, "a_cells_changed": n_changed,
                     "a_share": n_changed / n_cells, "a_holds": bool(a_ok), "a_mean_keep_cost_bps": dd["keep_lam0_best_crossfit_bps"].mean(),
                     "label_changes_description": int(dd["best_differs_from_lam0"].sum()),
                     "b_crossfit_regret_lam0_bps": base["twap_regret_crossfit_bps"].groupby(level=0).mean().mean(),
                     "b_crossfit_regret_arm_bps": dd["twap_regret_crossfit_bps"].groupby(level=0).mean().mean(), "b_rise_bps": rise_m.mean(),
                     "b_rise_se_over_markets": se(rise_m), "b_markets_rising": int((rise_m > 0).sum()), "b_holds": bool(b_ok),
                     "plain_regret_rise_bps": plain_m.mean(), "passes": bool(a_ok or b_ok) if arm in JUDGED else None})
        for (m, f), r in dd.iterrows():
            cells.append({"arm": arm, "market": m, "size_frac": f, "lam0_best": base.loc[(m, f), "best_policy"], "best": r["best_policy"],
                          "label_differs": r["best_differs_from_lam0"], "keep_lam0_best_crossfit_bps": r["keep_lam0_best_crossfit_bps"],
                          "cell_changed": r["cell_changed"], "lam0_best_minus_best_bps": r["lam0_best_minus_best_bps"],
                          "lam0_best_minus_best_se": r["lam0_best_minus_best_se"], "crossfit_regret_lam0": base.loc[(m, f), "twap_regret_crossfit_bps"],
                          "crossfit_regret": r["twap_regret_crossfit_bps"]})
    return pd.DataFrame(rows), pd.DataFrame(cells), excluded


def best_noise(truth: pd.DataFrame, kept: list, n_boot: int = 500) -> pd.DataFrame:
    """Descriptive, not part of the rule: how often the best strategy of a cell changes when the 100 seeds are resampled
    (lambda = 0 against itself), so that criterion (a) can be read against its own noise."""
    t = truth[truth["is_bps"].notna() & truth["market"].isin(kept)]
    rng = np.random.default_rng(0)
    rows = []
    for (m, a, f), d in t.groupby(["market", "arm", "size_frac"]):
        c = d.pivot_table(index="seed", columns="policy", values="is_bps").dropna()
        best = c.mean().idxmin()
        x = c.to_numpy()
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        b = x[idx].mean(1).argmin(1)
        rows.append({"market": m, "arm": a, "size_frac": f, "best": best, "share_of_resamples_with_another_best": float((c.columns[b] != best).mean())})
    return pd.DataFrame(rows)


# ---- E4
def e4(truth: pd.DataFrame) -> pd.DataFrame:
    t = truth[(truth["policy"] == "P0_twap") & truth["disp_end_bps"].notna()]
    rows = []
    for (a, f), d in t[t["market"].map(fam) == "agent-mix"].groupby(["arm", "size_frac"]):
        rows.append({"arm": a, "size_frac": f, "market": "pooled agent-mix", **K.persistence(d)})
    for (m, a, f), d in t.groupby(["market", "arm", "size_frac"]):
        rows.append({"arm": a, "size_frac": f, "market": m, **K.persistence(d)})
    out = pd.DataFrame(rows)
    out["band_lo"], out["band_hi"] = BAND
    out["arm"] = pd.Categorical(out["arm"], ARMS, ordered=True)
    return out.sort_values(["size_frac", "market", "arm"])


# ---- E2
def e2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(load("e2"))
    if "failed" not in raw:
        raw["failed"] = np.nan
    d = raw[raw["failed"].isna()].copy()
    F = PR.FEATURES
    usable = [f for f in F if d[f].notna().mean() >= 0.8]
    offB = d[(d["arm"] == "lam0") & (d["tag"] == "B")]
    offA = d[(d["arm"] == "lam0") & (d["tag"] == "A")].set_index(["market", "seed"])
    z = d.copy()
    ms = sorted(d["market"].unique())
    for m in ms:
        ref = offB[offB["market"] == m]
        for f in F:
            sd = ref[f].std()
            z.loc[z["market"] == m, f] = (d.loc[d["market"] == m, f] - ref[f].mean()) / (sd if sd and np.isfinite(sd) else 1.0)
    diffs, aucs, ident = [], [], []
    for arm in ARMS:
        sel = (d["arm"] == arm) & (d["tag"] == "A")
        A = d[sel].set_index(["market", "seed"])
        same = A["tape_hash"].eq(offA["tape_hash"].reindex(A.index))
        ident.append({"arm": arm, "n": len(A), "samples_bit_identical_to_lam0": int(same.sum())})
        for m in ms + ["pooled"]:
            a = (z if m == "pooled" else d)[sel]
            b = (z if m == "pooled" else d)[(d["arm"] == "lam0") & (d["tag"] == "B")]
            if m != "pooled":
                a, b = a[a["market"] == m], b[b["market"] == m]
            X = np.vstack([a[usable].to_numpy(float), b[usable].to_numpy(float)])
            y = np.r_[np.ones(len(a)), np.zeros(len(b))]
            g = np.r_[a["market"].to_numpy(), b["market"].to_numpy()] if m == "pooled" else None
            mean, sd = PR.cv_auc(X, y, g)
            row = {"arm": arm, "market": m, "n_on": len(a), "n_off": len(b), "auc": mean, "auc_sd_over_repeats": sd,
                   "role": "null (lambda 0 against lambda 0, other seeds)" if arm == "lam0" else "arm against lambda 0"}
            if m == "pooled":
                rng = np.random.default_rng(1)
                perm = []
                for _ in range(200):
                    yp = y.copy()
                    for mk in np.unique(g):
                        i = np.nonzero(g == mk)[0]
                        yp[i] = rng.permutation(yp[i])
                    perm.append(PR.cv_auc(X, yp, g, repeats=1, seed=int(rng.integers(1 << 30)))[0])
                row["perm_p95"], row["perm_p_value"] = float(np.percentile(perm, 95)), float((1 + np.sum(np.array(perm) >= mean)) / 201)
            aucs.append(row)
        if arm == "lam0":
            continue
        for f in F:
            on, off = d[sel], offB
            pooled_z = z[sel][f].mean()  # in s.d. of the lambda 0 (B) samples of the same market
            paired = (A[f] - offA[f].reindex(A.index)).dropna()
            diffs.append({"arm": arm, "feature": f, "mean_on": on[f].mean(), "mean_lam0": off[f].mean(), "diff_in_sd_of_lam0": pooled_z,
                          "diff_se_in_sd": se(z[sel][f]), "paired_diff": paired.mean(), "paired_se": se(paired),
                          "markets_same_sign": int(sum(np.sign(z[sel & (d.market == m)][f].mean()) == np.sign(pooled_z) for m in ms)), "n_markets": len(ms)})
    aucs = pd.DataFrame(aucs)
    aucs["arm"] = pd.Categorical(aucs["arm"], ARMS, ordered=True)
    return pd.DataFrame(diffs), aucs.sort_values(["market", "arm"]), pd.DataFrame(ident)


# ---- figure
def figure(summ: pd.DataFrame, per: pd.DataFrame, ver: pd.DataFrame, excluded: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2933", "#52606d", "#d9dee4"
    col = {"lam0": "#9aa5b1", "all_dose": "#a9c7ee", "all_band_low": "#2a78d6", "all_max_feasible": "#12427f", "trader_band_low": "#c2571a"}
    lab = {"lam0": "lambda 0", "all_dose": "all-flow 0.25", "all_band_low": "all-flow 0.6", "all_max_feasible": "all-flow 1.0",
           "trader_band_low": "trader-only 0.6 (comparison)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    am = summ[(summ["family"] == "agent-mix") & ~summ["market"].isin(excluded)]
    ms = sorted(am["market"].unique())
    # 1: cross-fitted TWAP regret per market (mean over sizes)
    ax = axes[0]
    w = 0.16
    for i, a in enumerate(ARMS):
        v = am[am["arm"] == a].groupby("market")["twap_regret_crossfit_bps"].mean().reindex(ms)
        ax.bar(np.arange(len(ms)) + (i - 2) * w, v, w * 0.92, color=col[a], label=lab[a])
    ax.set_xticks(np.arange(len(ms)), [m.replace(P.LW.PILOT_PREFIX, "") for m in ms])
    ax.axhline(0, color=muted, lw=0.8)
    ax.set_title("Always-TWAP regret, cross-fitted (bp), mean of the two sizes", color=ink, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    # 2: rule criteria
    ax = axes[1]
    v = ver.set_index("arm")
    x = np.arange(len(v))
    ax.bar(x, v["b_rise_bps"], 0.5, color=[col[a] for a in v.index])
    ax.errorbar(x, v["b_rise_bps"], yerr=v["b_rise_se_over_markets"], fmt="none", ecolor=ink, lw=1)
    ax.axhline(REGRET_RISE_BP, color="#b42318", lw=1, ls="--")
    ax.text(len(v) - 0.5, REGRET_RISE_BP, " rule (b): 1.0 bp", color="#b42318", fontsize=8, va="bottom", ha="right")
    for i, a in enumerate(v.index):
        ax.text(i, min(v["b_rise_bps"].min(), 0) - 0.02, f"(a) {int(v.loc[a, 'a_cells_changed'])}/{int(v.loc[a, 'cells'])}", ha="center", va="top", fontsize=8, color=ink)
    ax.set_xticks(x, [lab[a].replace(" (comparison)", "\n(comparison)") for a in v.index], fontsize=8)
    ax.axhline(0, color=muted, lw=0.8)
    ax.set_title("Rise of cross-fitted TWAP regret over lambda 0 (bp);\n(a): cells where keeping the lambda 0 best costs 0.5 bp or more", color=ink, fontsize=10, loc="left")
    # 3: persistence
    ax = axes[2]
    p = per[(per["market"] == "pooled agent-mix") & (per["size_frac"] == 0.15)].set_index("arm").reindex(ARMS)
    x = np.arange(len(p))
    ax.axhspan(*BAND, color="#e6f0e9", zorder=0)
    ax.text(len(p) - 0.55, BAND[1], "literature band 2/3 to 0.87 ", fontsize=8, color=muted, va="bottom", ha="right")
    ax.errorbar(x, p["persistence"], yerr=[p["persistence"] - p["persistence_lo"], p["persistence_hi"] - p["persistence"]], fmt="o", color=ink, ecolor=muted, lw=1)
    ax.set_xticks(x, [lab[a].replace(" (comparison)", "\n(comparison)") for a in p.index], fontsize=8)
    ax.set_title("TWAP impact persistence at 15% (evaluation seeds):\ndisplacement 120 s after the deadline over at the deadline", color=ink, fontsize=10, loc="left")
    for ax in axes:
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lambda_pilot.png", dpi=160)
    plt.close(fig)


def main() -> None:
    truth = pd.DataFrame(load("truth"))
    via = e3()
    via.to_csv(OUT / "e3_viability.csv", index=False)
    costs, summ = e1(truth)
    costs.to_csv(OUT / "e1_costs.csv", index=False)
    summ.to_csv(OUT / "e1_summary.csv", index=False)
    ver, cells, excluded = verdict(summ)
    ver.to_csv(OUT / "verdict.csv", index=False)
    cells.to_csv(OUT / "verdict_cells.csv", index=False)
    kept = sorted(set(cells["market"]))
    noise = best_noise(truth, kept)
    noise.to_csv(OUT / "e1_best_strategy_noise.csv", index=False)
    per = e4(truth)
    per.to_csv(OUT / "e4_persistence.csv", index=False)
    diffs, aucs, ident = e2()
    diffs.to_csv(OUT / "e2_feature_differences.csv", index=False)
    aucs.to_csv(OUT / "e2_auc.csv", index=False)
    ident.to_csv(OUT / "e2_identity.csv", index=False)
    figure(summ, per, ver, excluded)
    pd.set_option("display.width", 250)
    print("excluded (calib_exclusion.csv):", excluded)
    print(ver.round(3).to_string(index=False))
    print(cells.round(2).to_string(index=False))
    print(noise.groupby("arm")["share_of_resamples_with_another_best"].mean().round(2).to_string())
    print(per[per["market"] == "pooled agent-mix"].round(3).to_string(index=False))
    print(aucs[aucs["market"] == "pooled"].round(3).to_string(index=False))
    print(ident.to_string(index=False))
    print(via.groupby("arm", observed=True)[["viable", "fail_share", "wild_share", "volume", "spread_bps", "vol_bps"]].mean().round(3).to_string())
    print(summ[summ.family == "agent-mix"].groupby(["arm", "size_frac"], observed=True)[["twap_cost_bps", "twap_regret_bps", "twap_regret_crossfit_bps", "reversal_minus_best_bps", "reversal_minus_twap_bps"]].mean().round(2).to_string())
    print(summ.pivot_table(index=["market", "size_frac"], columns="arm", values="best_policy", aggfunc="first", observed=True).to_string())


if __name__ == "__main__":
    main()
