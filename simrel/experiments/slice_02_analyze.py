"""
Analysis of the scenario-slice experiment: realism per slice, the pre-fixed realism gate, and best policies.

    python simrel/experiments/slice_02_analyze.py

Writes CSV tables and figures to results/simrel/slices/.
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel import slices as S  # noqa: E402
from simrel.configs import slices as C  # noqa: E402
from simrel.metrics.cost import boot_ci  # noqa: E402

OUT = S.out_dir()
RAW = OUT / "raw"
SOURCES = {"real_fit_days": "Real (other days)", "hawkes": "Hawkes", "abides": "ABIDES", "trades": "TRADES"}
MOMENTS = ["volatility_bps", "spread_bps", "volume_visible", "events", "depth10", "trend_bps"]


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def realism_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agg, per_stat, mom = [], [], []
    for p in sorted(glob.glob(str(RAW / "score" / "*.pkl"))):
        stem = Path(p).stem
        name = next(n for n in S.slice_names() if stem.endswith("_" + n))
        source = stem[: -len(name) - 1]
        d = load(p)
        if "failed" in d:
            agg.append({"slice": name, "source": source, "failed": d["failed"][-120:]})
            continue
        agg.append({"slice": name, "source": source, **d["aggregate"], "n_real": d["n_real"], "n_gen": d["n_gen"]})
        t = d["table"].copy()
        t.insert(0, "source", source)
        t.insert(0, "slice", name)
        per_stat.append(t)
        row = {"slice": name, "source": source}
        for k in MOMENTS:
            row[k] = float(np.mean([f[k] for f in d["features"]]))
            row[k + "_real"] = float(np.mean([f[k] for f in d["real_features"]]))
        mom.append(row)
    return pd.DataFrame(agg), pd.concat(per_stat, ignore_index=True), pd.DataFrame(mom)


def gate(agg: pd.DataFrame) -> pd.DataFrame:
    """The pre-fixed rule: each simulator at or below the ceiling, and the two within the band of each other."""
    w = agg.pivot(index="slice", columns="source", values="l1_mean")
    sims = [s for s in ("hawkes", "abides", "trades") if s in w.columns]
    out = w.rename(columns={s: f"l1_{s}" for s in w.columns}).copy()
    for s in sims:
        out[f"{s}_realistic"] = w[s] <= C.REALISM_CEILING
    for i, a in enumerate(sims):
        for b in sims[i + 1:]:
            out[f"similar_{a}_{b}"] = (w[a] - w[b]).abs() <= C.SIMILAR_BAND
            out[f"gate_{a}_{b}"] = out[f"similar_{a}_{b}"] & out[f"{a}_realistic"] & out[f"{b}_realistic"]
    return out.reset_index()


def policy_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, twins = [], []
    for p in glob.glob(str(RAW / "policy" / "*.pkl")):
        d = load(p)
        rows.extend(d["rows"])
        twins.append(d["twin"])
    return pd.DataFrame(rows), pd.DataFrame(twins)


def policy_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (simulator, slice): mean cost of each policy, the best one, and which others are statistically tied with it."""
    df = df[df["failed"].isna() | (df["failed"] == False)]  # noqa: E712
    summary, best = [], []
    for (sim, name), d in df.groupby(["sim", "slice"]):
        wide = d.pivot(index="seed", columns="policy", values="is_bps").dropna()
        if len(wide) < 5:
            continue
        means = wide.mean().sort_values()
        top = means.index[0]
        tied = []
        for pol in wide.columns:
            lo, hi = boot_ci(wide[pol].to_numpy())
            dlo, dhi = boot_ci((wide[pol] - wide[top]).to_numpy()) if pol != top else (0.0, 0.0)
            imp = d[d["policy"] == pol]["impact_bps"].mean()
            summary.append({"sim": sim, "slice": name, "policy": pol, "n": len(wide), "mean_is_bps": wide[pol].mean(), "ci_lo": lo, "ci_hi": hi,
                            "sd": wide[pol].std(), "impact_bps": imp, "excess_over_best": wide[pol].mean() - means.iloc[0], "excess_lo": dlo, "excess_hi": dhi,
                            "rank": int(list(means.index).index(pol)) + 1, "unfilled_frac": d[d["policy"] == pol]["unfilled_frac"].mean()})
            if pol != top and not (dlo > 0):
                tied.append(pol)
        best.append({"sim": sim, "slice": name, "n": len(wide), "best": top, "best_is_bps": means.iloc[0], "runner_up": means.index[1],
                     "gap_to_runner_up": means.iloc[1] - means.iloc[0], "n_tied_with_best": len(tied), "tied_with_best": " ".join(tied),
                     "twap_is_bps": wide["P0_twap"].mean(), "spread_of_means": means.iloc[-1] - means.iloc[0]})
    return pd.DataFrame(summary), pd.DataFrame(best)


def cross(summary: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """For each slice and pair of simulators: what does one simulator's best policy cost in the other?"""
    rows = []
    sims = sorted(best["sim"].unique())
    for name in sorted(best["slice"].unique()):
        b = best[best["slice"] == name].set_index("sim")
        for i, a in enumerate(sims):
            for c in sims[i + 1:]:
                if a not in b.index or c not in b.index:
                    continue
                sa = summary[(summary["slice"] == name) & (summary["sim"] == a)].set_index("policy")
                sc = summary[(summary["slice"] == name) & (summary["sim"] == c)].set_index("policy")
                pa, pc = b.loc[a, "best"], b.loc[c, "best"]
                common = sa.index.intersection(sc.index)
                rho = float(pd.Series(sa.loc[common, "mean_is_bps"]).rank().corr(pd.Series(sc.loc[common, "mean_is_bps"]).rank()))
                rows.append({
                    "slice": name, "sim_a": a, "sim_b": c, "best_a": pa, "best_b": pc, "same_best": pa == pc,
                    f"regret_of_b_best_in_a": sa.loc[pc, "excess_over_best"], "regret_b_in_a_lo": sa.loc[pc, "excess_lo"], "regret_b_in_a_hi": sa.loc[pc, "excess_hi"],
                    f"regret_of_a_best_in_b": sc.loc[pa, "excess_over_best"], "regret_a_in_b_lo": sc.loc[pa, "excess_lo"], "regret_a_in_b_hi": sc.loc[pa, "excess_hi"],
                    "disagree": bool(pa != pc and sa.loc[pc, "excess_lo"] > 0 and sc.loc[pa, "excess_lo"] > 0),
                    "one_sided": bool(pa != pc and ((sa.loc[pc, "excess_lo"] > 0) != (sc.loc[pa, "excess_lo"] > 0))),
                    "rank_correlation": rho,
                })
    return pd.DataFrame(rows)


def figures(agg: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#59636e", "#d8dee4"
    colors = {"real_fit_days": "#8b949e", "hawkes": "#2f6fb5", "abides": "#c2571a", "trades": "#3b8552"}
    names = [n for n in S.slice_names() if n in set(agg["slice"])]
    srcs = [s for s in SOURCES if s in set(agg["source"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.8 / len(srcs)
    for j, s in enumerate(srcs):
        d = agg[agg["source"] == s].set_index("slice").reindex(names)
        ax.bar(np.arange(len(names)) + (j - (len(srcs) - 1) / 2) * width, d["l1_mean"], width * 0.92, color=colors[s], label=SOURCES[s])
    ax.axhline(C.REALISM_CEILING, color=ink, lw=1, ls="--")
    ax.text(len(names) - 0.45, C.REALISM_CEILING + 0.008, f"ceiling {C.REALISM_CEILING}", ha="right", va="bottom", fontsize=8, color=ink)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8, color=ink)
    ax.set_ylabel("LOB-Bench mean L1 distance (lower is more realistic)", fontsize=9, color=ink)
    ax.set_title("Realism by slice, scored against held-out real windows", fontsize=10, color=ink, loc="left")
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=8, ncol=len(srcs), loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.tight_layout()
    fig.savefig(OUT / "fig_realism.png", dpi=170)
    plt.close(fig)

    if summary.empty:
        return
    sims = [s for s in ("hawkes", "abides", "trades") if s in set(summary["sim"])]
    pols = sorted(summary["policy"].unique())
    fig, axes = plt.subplots(1, len(sims), figsize=(5.6 * len(sims), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    vmax = float(np.nanpercentile(summary["excess_over_best"], 95))
    for ax, sim in zip(axes, sims):
        m = summary[summary["sim"] == sim].pivot(index="policy", columns="slice", values="excess_over_best").reindex(index=pols, columns=names)
        im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, "best" if v == 0 else f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 * vmax else ink,
                            fontweight="bold" if v == 0 else "normal")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7, color=ink)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels(pols, fontsize=8, color=ink)
        ax.set_title(f"{SOURCES[sim]}: cost above that simulator's best policy (bps)", fontsize=9, color=ink, loc="left")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(OUT / "fig_policy_excess.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    agg, per_stat, mom = realism_tables()
    agg.to_csv(OUT / "realism.csv", index=False)
    per_stat.to_csv(OUT / "realism_by_statistic.csv", index=False)
    mom.to_csv(OUT / "realism_moments.csv", index=False)
    g = gate(agg)
    g.to_csv(OUT / "realism_gate.csv", index=False)
    print("== realism (LOB-Bench mean L1; ceiling", C.REALISM_CEILING, ", band", C.SIMILAR_BAND, ")")
    print(g.round(3).to_string(index=False))
    print(per_stat.pivot_table(index="statistic", columns="source", values="l1").round(2).to_string())
    print(mom.groupby("source")[[c for c in mom.columns if c not in ("slice", "source")]].mean().round(1).T.to_string())

    df, twins = policy_rows()
    summary = pd.DataFrame()
    if len(df):
        df.to_csv(OUT / "policy_episodes.csv", index=False)
        twins.to_csv(OUT / "policy_twins.csv", index=False)
        summary, best = policy_tables(df)
        summary.to_csv(OUT / "policy_summary.csv", index=False)
        best.to_csv(OUT / "policy_best.csv", index=False)
        x = cross(summary, best)
        if x.empty:
            print("too few finished seeds for a cross-simulator comparison yet")
            raise SystemExit
        x = x.merge(g, on="slice", how="left")
        x.to_csv(OUT / "policy_cross.csv", index=False)
        print("== best policy per slice and simulator")
        print(best.round(2).to_string(index=False))
        print("== cross-simulator comparison")
        print(x[[c for c in x.columns if not c.startswith("l1_real")]].round(2).to_string(index=False))
        fails = twins[twins["failed"].notna() & (twins["failed"] != False)] if "failed" in twins else twins.iloc[0:0]  # noqa: E712
        print("failed twin runs:", len(fails), "of", len(twins))
    figures(agg, summary)
