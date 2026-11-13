"""
Design addendum 11, development stage: tables and figures of the reactive-family pilot.

Reads the stage files written by reactive_pilot.py (results/simrel/goal2/addendum11_pilot/raw) and writes one CSV per
experiment and four figures next to them.

    python simrel/experiments/reactive_pilot.py report
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
from simrel.experiments import reactive_pilot as P  # noqa: E402

OUT, RAW = P.OUT, P.RAW
FEATURES = ["H", "gamma_flow", "R100_over_R1", "block_ac25", "vr30", "vr120", "vr600", "depth_flow"]
ARM_COLOR = {"off": "#a9c7ee", "weak": "#6ea3e3", "medium": "#2a78d6", "strong": "#12427f",
             "weak_tight": "#9bdcc4", "medium_tight": "#1baf7a", "strong_tight": "#0c6b49", P.LOOSE: "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(name: str):
    return pickle.load(open(RAW / f"{name}.pkl", "rb"))


def se(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def family(market: str) -> str:
    return "agent-mix" if P.MARKETS[market][0] == "am" else "latent"


# ---- E4
def e4() -> pd.DataFrame:
    tw = pd.DataFrame(load("twins_off") + load("twins_on"))
    rows = []
    for (m, a), d in tw.groupby(["market", "arm"]):
        ok = d[d["failed"].isna()]
        fail = 1.0 - len(ok) / len(d)
        wild = float((ok["move_bps_abs"] > C.WILD_MOVE_BPS).mean()) if len(ok) else 1.0
        rows.append({"market": m, "arm": a, "n": len(d), "fail_share": fail, "wild_share": wild,
                     "viable": bool(fail <= C.MAX_FAIL_SHARE and wild <= C.MAX_WILD_SHARE), "volume": ok["volume"].mean(),
                     "spread_bps": ok["spread_bps"].mean(), "vol_bps": ok["vol_bps"].mean(), "move_bps_abs": ok["move_bps_abs"].mean(),
                     "twin_triggers_per_episode": ok["pred_triggers"].mean(), "twin_pred_volume_share": ok["pred_volume"].sum() / ok["volume"].sum()})
    out = pd.DataFrame(rows)
    out["arm"] = pd.Categorical(out["arm"], P.ARMS, ordered=True)
    return out.sort_values(["market", "arm"])


# ---- E1
def longest_run(s: str) -> int:
    best = cur = 0
    for x in s.split(","):
        cur = cur + 1 if int(x) > 0 else 0
        best = max(best, cur)
    return best


def e1(truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = truth[truth["failed"].isna()].copy()
    t["longest_run"] = [longest_run(s) if isinstance(s, str) else np.nan for s in t["first_slices"]]
    rows = []
    for (m, a, f, pol), d in t.groupby(["market", "arm", "size_frac", "policy"]):
        rows.append({"market": m, "arm": a, "size_frac": f, "policy": pol, "n": len(d), "trigger_rate": d["trig_task"].mean(),
                     "first_trigger_s_median": d["trig_task_first_s"].median(), "children": d["n_children"].mean(),
                     "longest_run": d["longest_run"].mean(), "child_size_cv": d["child_size_cv"].mean(),
                     "pred_volume_over_qty": (d["pred_volume"] / d["qty"]).mean()})
    by = pd.DataFrame(rows)
    # how each strategy looks to the detector: strength off (no feedback), pooled over markets, with the detector sweep
    off = t[(t["arm"] == "off")]
    sw = off[off["seed"] < P.N_SWEEP]
    look = []
    for (f, pol), d in off.groupby(["size_frac", "policy"]):
        s = sw[(sw["size_frac"] == f) & (sw["policy"] == pol)]
        row = {"size_frac": f, "policy": pol, "children": d["n_children"].mean(), "longest_run": d["longest_run"].mean(),
               "child_size_cv": d["child_size_cv"].mean(), "trigger_rate": d["trig_task"].mean(),
               "first_trigger_s_median": d["trig_task_first_s"].median()}
        for tag in P.SWEEP:
            row[f"rate_{tag}"] = s[f"sw_{tag}"].astype(float).mean()
        look.append(row)
    look = pd.DataFrame(look)
    # passive episodes
    pa = pd.DataFrame(load("passive"))
    pa["fit_like"] = pa["seed"] < C.SEED_FIT + 6
    rows = []
    for (m, a), d in pa.groupby(["market", "arm"]):
        six = d[d["fit_like"]]
        row = {"market": m, "arm": a, "episodes": len(d), "triggers_per_episode": d["pred_triggers"].mean(),
               "share_episodes_with_trigger": (d["pred_triggers"] > 0).mean(), "triggers_in_six_fit_episodes": int(six["pred_triggers"].sum()),
               "pred_orders_per_episode": d["pred_orders"].mean(), "pred_volume_share": d["pred_volume"].sum() / d["market_volume"].sum(),
               "pred_volume_share_six": six["pred_volume"].sum() / six["market_volume"].sum(), "residual_inventory_share": (d["pred_inv_end"] != 0).mean()}
        if a == "off":
            for tag in P.SWEEP:
                row[f"per_episode_{tag}"] = d[f"sw_{tag}"].mean()
        rows.append(row)
    false = pd.DataFrame(rows)
    false["arm"] = pd.Categorical(false["arm"], P.ARMS, ordered=True)
    return by, look, false.sort_values(["market", "arm"])


# ---- E2
def e2(truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = truth[truth["failed"].isna() & (truth["policy"] != "none")].copy()
    t["pred_pnl_bps"] = t["pred_pnl_ticks"] / (t["qty"] * t["arrival_mid"]) * 1e4
    cost = t.pivot_table(index=["market", "arm", "size_frac", "seed"], columns="policy", values="is_bps")
    rows, summ = [], []
    off_twap = {(m, f): d.droplevel([0, 1, 2]).dropna()["P0_twap"] for (m, a, f), d in cost.groupby(level=[0, 1, 2]) if a == "off"}
    for (m, a, f), d in cost.groupby(level=[0, 1, 2]):
        d = d.droplevel([0, 1, 2]).dropna()
        sub = t[(t["market"] == m) & (t["arm"] == a) & (t["size_frac"] == f)]
        means = d.mean()
        for pol in d.columns:
            s = sub[sub["policy"] == pol]
            rows.append({"market": m, "arm": a, "size_frac": f, "policy": pol, "n": len(d), "cost_bps": means[pol], "cost_se": se(d[pol]),
                         "minus_twap_bps": (d[pol] - d["P0_twap"]).mean(), "minus_twap_se": se(d[pol] - d["P0_twap"]),
                         "pred_pnl_bps": s["pred_pnl_bps"].mean(), "pred_pnl_se": se(s["pred_pnl_bps"]),
                         "pred_volume_over_qty": (s["pred_volume"] / s["qty"]).mean(), "trigger_rate": s["trig_task"].mean(),
                         "residual_inventory_share": (s["pred_inv_end"] != 0).mean(), "unfilled_frac": s["unfilled_frac"].mean()})
        best = means.idxmin()
        a_half, b_half = d[d.index < P.N_TRUTH // 2], d[d.index >= P.N_TRUTH // 2]
        ba, bb = a_half.mean().idxmin(), b_half.mean().idxmin()
        xfit = 0.5 * ((b_half["P0_twap"] - b_half[ba]).mean() + (a_half["P0_twap"] - a_half[bb]).mean())
        base = off_twap.get((m, f))
        dt = (d["P0_twap"] - base).dropna() if base is not None else pd.Series(dtype=float)
        tw = sub[sub["policy"] == "P0_twap"]
        summ.append({"market": m, "family": family(m), "arm": a, "size_frac": f, "n": len(d), "twap_cost_bps": means["P0_twap"],
                     "best_policy": best, "best_cost_bps": means[best], "twap_regret_bps": means["P0_twap"] - means[best],
                     "twap_regret_se": se(d["P0_twap"] - d[best]), "best_policy_half_a": ba, "best_policy_half_b": bb,
                     "twap_regret_crossfit_bps": xfit, "p0r_minus_twap_bps": (d["P0r_twap_random"] - d["P0_twap"]).mean(),
                     "p0r_minus_twap_se": se(d["P0r_twap_random"] - d["P0_twap"]),
                     "twap_cost_minus_off_bps": dt.mean() if len(dt) else np.nan, "twap_cost_minus_off_se": se(dt) if len(dt) else np.nan,
                     "pred_pnl_vs_twap_bps": tw["pred_pnl_bps"].mean(), "pred_pnl_vs_twap_se": se(tw["pred_pnl_bps"]),
                     "pred_pnl_all_policies_bps": sub["pred_pnl_bps"].mean()})
    costs, summ = pd.DataFrame(rows), pd.DataFrame(summ)
    for df in (costs, summ):
        df["arm"] = pd.Categorical(df["arm"], P.TRUTH_ARMS, ordered=True)
    return costs.sort_values(["market", "size_frac", "arm", "policy"]), summ.sort_values(["market", "size_frac", "arm"])


# ---- E3
def _fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    from scipy.optimize import minimize

    Z = np.column_stack([np.ones(len(X)), X])

    def loss(w):
        z = Z @ w
        return np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * ridge * np.sum(w[1:] ** 2)

    def grad(w):
        g = Z.T @ (1.0 / (1.0 + np.exp(-(Z @ w))) - y)
        g[1:] += ridge * w[1:]
        return g

    return minimize(loss, np.zeros(Z.shape[1]), jac=grad, method="L-BFGS-B").x


def auc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, repeats: int = 20, folds: int = 5, seed: int = 0) -> tuple[float, float]:
    """Mean and s.d. over repeats of the AUC of out-of-fold scores of a ridge logistic classifier (stratified folds).

    Missing values are replaced by the training mean and features are standardised on the training folds. With groups
    (the market of each sample) the folds are stratified by group and class.
    """
    rng = np.random.default_rng(seed)
    strata = y.astype(int) if groups is None else (pd.factorize(groups)[0] * 2 + y.astype(int))
    vals = []
    for _ in range(repeats):
        fold = np.zeros(len(y), dtype=int)
        for s in np.unique(strata):
            idx = rng.permutation(np.nonzero(strata == s)[0])
            fold[idx] = np.arange(len(idx)) % folds
        score = np.zeros(len(y))
        for k in range(folds):
            tr, te = fold != k, fold == k
            mu = np.nanmean(X[tr], axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            A, B = np.where(np.isfinite(X[tr]), X[tr], mu), np.where(np.isfinite(X[te]), X[te], mu)
            sd = A.std(axis=0)
            sd[sd == 0] = 1.0
            w = _fit_logistic((A - mu) / sd, y[tr])
            score[te] = w[0] + ((B - mu) / sd) @ w[1:]
        vals.append(auc(score, y))
    return float(np.mean(vals)), float(np.std(vals))


def e3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(load("e3"))
    if "failed" not in raw:
        raw["failed"] = np.nan
    d = raw[raw["failed"].isna()].copy()
    d["pred_volume_share"] = d["pred_volume"] / d["market_volume"]
    usable = [f for f in FEATURES if d[f].notna().mean() >= 0.8]
    offB = d[(d["arm"] == "off") & (d["tag"] == "B")]
    # features standardised within market by the off (B) samples, so that markets can be pooled
    z = d.copy()
    for m in P.MARKETS:
        ref = offB[offB["market"] == m]
        for f in FEATURES:
            sd = ref[f].std()
            z.loc[z["market"] == m, f] = (d.loc[d["market"] == m, f] - ref[f].mean()) / (sd if sd and np.isfinite(sd) else 1.0)
    diffs, aucs = [], []
    for arm in P.ARMS:
        A = d[(d["arm"] == arm) & (d["tag"] == "A")]
        for m in list(P.MARKETS) + ["pooled"]:
            a = (z if m == "pooled" else d)[(d["arm"] == arm) & (d["tag"] == "A")]
            b = (z if m == "pooled" else d)[(d["arm"] == "off") & (d["tag"] == "B")]
            if m != "pooled":
                a, b = a[a["market"] == m], b[b["market"] == m]
            X = np.vstack([a[usable].to_numpy(float), b[usable].to_numpy(float)])
            y = np.r_[np.ones(len(a)), np.zeros(len(b))]
            g = np.r_[a["market"].to_numpy(), b["market"].to_numpy()] if m == "pooled" else None
            mean, sd = cv_auc(X, y, g)
            row = {"arm": arm, "market": m, "n_on": len(a), "n_off": len(b), "auc": mean, "auc_sd_over_repeats": sd,
                   "role": "null (off against off, other seeds)" if arm == "off" else "predator on against off",
                   "pred_volume_share": A["pred_volume"].sum() / A["market_volume"].sum() if m == "pooled" else a["pred_volume"].sum() / a["market_volume"].sum(),
                   "triggers_per_sample": (A if m == "pooled" else a)["pred_triggers"].mean()}
            if m == "pooled":  # permutation reference: labels shuffled within market
                rng = np.random.default_rng(1)
                perm = []
                for _ in range(200):
                    yp = y.copy()
                    for mk in np.unique(g):
                        i = np.nonzero(g == mk)[0]
                        yp[i] = rng.permutation(yp[i])
                    perm.append(cv_auc(X, yp, g, repeats=1, seed=int(rng.integers(1 << 30)))[0])
                row["perm_p95"], row["perm_p_value"] = float(np.percentile(perm, 95)), float((1 + np.sum(np.array(perm) >= mean)) / 201)
            aucs.append(row)
        if arm == "off":
            continue
        # per-feature differences: independent seeds (on A against off B) and paired seeds (on A against off A)
        offA = d[(d["arm"] == "off") & (d["tag"] == "A")].set_index(["market", "seed"])
        for f in FEATURES:
            for m in P.MARKETS:
                a, b = A[A["market"] == m], offB[offB["market"] == m]
                pair = (a.set_index(["market", "seed"])[f] - offA[f]).dropna()
                sd = b[f].std()
                diffs.append({"arm": arm, "market": m, "feature": f, "off_mean": b[f].mean(), "off_sd": sd, "on_mean": a[f].mean(),
                              "diff": a[f].mean() - b[f].mean(), "diff_in_off_sd": (a[f].mean() - b[f].mean()) / sd if sd else np.nan,
                              "paired_diff": pair.mean(), "paired_se": se(pair), "paired_identical_share": float((pair == 0).mean()) if len(pair) else np.nan,
                              "missing_share_on": a[f].isna().mean()})
    z["arm"] = pd.Categorical(z["arm"], P.ARMS, ordered=True)
    return raw, pd.DataFrame(diffs), pd.DataFrame(aucs), z


# ---- figures
def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)


def figures(look, false, costs, summ, z) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [p for p in look[look["size_frac"] == 0.05].sort_values("trigger_rate")["policy"]]
    # 1. trigger rate by strategy (strength off: detection without feedback), both sizes
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, f in zip(axes, C.SIZES):
        d = look[look["size_frac"] == f].set_index("policy").loc[order]
        y = np.arange(len(order))
        ax.barh(y, d["trigger_rate"], height=0.62, color="#2a78d6", label="family detector (noisy time stamps, size-led)")
        ax.scatter(d["rate_tight"], y, s=34, color="#eb6834", edgecolor="white", linewidth=0.8, zorder=3, label="tight-timing detector (comparison)")
        for yi, v, v2, t in zip(y, d["trigger_rate"], d["rate_tight"], d["first_trigger_s_median"]):
            ax.text(1.06, yi, f"{v:.0%} | {v2:.0%}" + (f"   first at {t:.0f} s" if np.isfinite(t) else ""), va="center", fontsize=7.5, color=MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(0, 1.6)
        ax.text(1.06, len(order) - 0.35, "family | tight", va="center", fontsize=7.5, color=INK)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        _style(ax, f"Order size {f:.0%} of volume", "share of episodes in which the detector triggers")
        ax.grid(axis="y", visible=False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8, frameon=False, loc="lower center", ncol=2)
    fig.suptitle("E1. Detector trigger rate by strategy (predator not trading; six pilot markets pooled; 100 seeds, tight detector 30 seeds)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / "fig_e1_trigger_rates.png", dpi=160)
    plt.close(fig)

    # 2. cost relative to TWAP by strength
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharey=True, sharex="col")
    pols = [p for p in sorted(costs["policy"].unique()) if p != "P0_twap"]
    costs = costs.assign(family=[family(m) for m in costs["market"]])
    for i, fam in enumerate(["agent-mix", "latent"]):
        for j, f in enumerate(C.SIZES):
            ax = axes[i, j]
            d = costs[(costs["family"] == fam) & (costs["size_frac"] == f)].groupby(["policy", "arm"])["minus_twap_bps"].mean().unstack()
            y = np.arange(len(pols))
            ax.axvline(0, color=MUTED, linewidth=0.8)
            for k, arm in enumerate(P.STRENGTHS):
                ax.scatter(d.loc[pols, arm], y + (k - 1.5) * 0.17, s=30, color=ARM_COLOR[arm], edgecolor="white", linewidth=0.6, label=arm, zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels(pols)
            _style(ax, f"{fam} markets, order size {f:.0%}", "cost minus TWAP cost, bps (negative: cheaper than TWAP)" if i == 1 else "")
            ax.grid(axis="y", visible=False)
    axes[0, 1].legend(title="predator strength", fontsize=8, title_fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("E2. True cost of each strategy relative to TWAP, by predator strength (family detector; mean over the family's pilot markets)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_e2_cost_vs_twap.png", dpi=160)
    plt.close(fig)

    # 3. headroom (always-TWAP regret) by strength, one line per market; rows: family detector, tight-timing detector
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharey=True)
    fam_color = {"agent-mix": "#2a78d6", "latent": "#eb6834"}
    for i, (arms, label) in enumerate([(P.STRENGTHS, "family detector"), (["off"] + P.TIGHT, "tight-timing detector (comparison)")]):
        for ax, f in zip(axes[i], C.SIZES):
            for m in P.MARKETS:
                d = summ[(summ["market"] == m) & (summ["size_frac"] == f)].set_index("arm").loc[arms]
                ax.plot(range(len(d)), d["twap_regret_crossfit_bps"], color=fam_color[family(m)], linewidth=2, marker="o", markersize=5)
                ax.text(len(d) - 1 + 0.08, d["twap_regret_crossfit_bps"].iloc[-1], m.replace(P.RX.PILOT_PREFIX, ""), fontsize=7.5, color=MUTED, va="center")
            ax.set_xticks(range(len(P.STRENGTHS)))
            ax.set_xticklabels(P.STRENGTHS)
            ax.set_xlim(-0.2, len(P.STRENGTHS) - 0.45)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            _style(ax, f"{label}, order size {f:.0%}", "predator strength" if i == 1 else "", "always-TWAP regret, bps (cross-fitted)" if f == C.SIZES[0] else "")
    for fam, c in fam_color.items():
        axes[0, 0].plot([], [], color=c, linewidth=2, label=fam)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("E2. Headroom over always-TWAP by predator strength (one line per pilot market)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_e2_headroom.png", dpi=160)
    plt.close(fig)

    # 4. E3 feature distributions (standardised within market by the predator-off samples)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True)
    zz = z[~((z["arm"] == "off") & (z["tag"] == "A"))]
    SHOW = P.STRENGTHS + ["strong_tight", P.LOOSE]
    for ax, f in zip(axes.ravel(), FEATURES):
        data = [zz[zz["arm"] == a][f].dropna().clip(-6, 6).to_numpy() for a in SHOW]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True, medianprops={"color": "white", "linewidth": 1.4},
                        flierprops={"marker": ".", "markersize": 3, "markeredgecolor": MUTED}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch, a in zip(bp["boxes"], SHOW):
            patch.set_facecolor(ARM_COLOR[a])
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(SHOW) + 1))
        ax.set_xticklabels(["off", "weak", "med.", "strong", "tight", "loose"], fontsize=7.5)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, f)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("value in s.d. of the predator-off samples", fontsize=8.5, color=MUTED)
    axes[1, 0].set_ylabel("value in s.d. of the predator-off samples", fontsize=8.5, color=MUTED)
    fig.suptitle("E3. Diagnostic features of passive three-hour samples, predator off and on (20 samples per market and arm, six markets pooled;\n"
                 "\"tight\" and \"loose\": the strong predator with the tight-timing detector and with a loose detector, the negative control)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_e3_features.png", dpi=160)
    plt.close(fig)


def main() -> None:
    truth = pd.DataFrame(load("truth"))
    if "failed" not in truth:
        truth["failed"] = np.nan
    truth["failed"] = truth["failed"].where(truth["failed"].notna(), np.nan)
    print("truth rows", len(truth), "failed", int(truth["failed"].notna().sum()))
    v = e4()
    v.to_csv(OUT / "e4_viability.csv", index=False)
    by, look, false = e1(truth)
    by.to_csv(OUT / "e1_detection_by_market.csv", index=False)
    look.to_csv(OUT / "e1_detection_by_strategy.csv", index=False)
    false.to_csv(OUT / "e1_false_triggers_passive.csv", index=False)
    costs, summ = e2(truth)
    costs.to_csv(OUT / "e2_costs.csv", index=False)
    summ.to_csv(OUT / "e2_summary.csv", index=False)
    raw, diffs, aucs, z = e3()
    raw.to_csv(OUT / "e3_samples.csv", index=False)
    diffs.to_csv(OUT / "e3_feature_differences.csv", index=False)
    aucs.to_csv(OUT / "e3_auc.csv", index=False)
    figures(look, false, costs, summ, z)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400, "display.float_format", lambda x: f"{x:.3f}")
    print("\nE4\n", v.to_string(index=False))
    print("\nE1 look\n", look.to_string(index=False))
    print("\nE1 false\n", false.to_string(index=False))
    print("\nE2 summary\n", summ.to_string(index=False))
    print("\nE3 auc\n", aucs.to_string(index=False))


if __name__ == "__main__":
    main()
