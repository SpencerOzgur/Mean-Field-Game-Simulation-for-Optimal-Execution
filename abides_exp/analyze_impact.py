"""
Impact-curve analysis for a harness sweep.

Usage (repo root):  python -m abides_exp.analyze_impact results/impact_sweep
Writes impact_curve.png and impact_summary.csv into the same folder.

Main metric: paired impact, i.e. (baseline mid - run mid) for the same seed, averaged
over the execution window. Pairing removes the market's own drift, so it is far less
noisy than shortfall against the arrival price.

Note on impact_peak_bps in runs.csv: it is the max of a noisy series per run, so it is
biased upward even when the true impact is zero. Do not use it for the curve.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e6e5e1", "#fcfcfb"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue, validated light->dark


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main(folder: str):
    folder = Path(folder)
    runs = pd.read_csv(folder / "runs.csv")
    paths = pd.read_csv(folder / "paths.csv").dropna()
    paths = paths.merge(runs[["seed", "qty", "pct"]], on=["seed", "qty"])
    dur = int(paths.t_rel_s[paths.t_rel_s > 0].min())  # grid step, to locate window
    meta_dur = 1800
    try:
        import json
        meta = json.load(open(folder / "meta.json"))
        h, m, s = map(int, meta["schedule"]["duration"].split(":"))
        meta_dur = h * 3600 + m * 60 + s
    except Exception:
        pass

    # Per-run paired averages
    def window_mean(lo, hi):
        sel = paths[(paths.t_rel_s > lo) & (paths.t_rel_s <= hi)]
        return sel.groupby(["seed", "pct"]).impact_bps.mean()

    during = window_mean(0, meta_dur).rename("impact_during")
    after = window_mean(meta_dur, meta_dur + 1800).rename("impact_after_30m")
    per_run = pd.concat([during, after], axis=1).reset_index()
    per_run = per_run.merge(runs[["seed", "pct", "shortfall_bps", "filled", "qty"]], on=["seed", "pct"])

    summ = per_run.groupby("pct").agg(
        qty=("qty", "first"),
        n=("seed", "count"),
        impact_during=("impact_during", "mean"),
        impact_during_ci=("impact_during", ci95),
        impact_after_30m=("impact_after_30m", "mean"),
        impact_after_30m_ci=("impact_after_30m", ci95),
        shortfall=("shortfall_bps", "mean"),
        shortfall_ci=("shortfall_bps", ci95),
        fill_rate=("filled", "sum"),
    ).reset_index()
    summ["fill_rate"] = summ.fill_rate / (summ.qty * summ.n)
    summ.round(3).to_csv(folder / "impact_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})

    # Panel A: mean impact paths for smallest, a middle, and largest size
    pcts = sorted(paths.pct.unique())
    show = [pcts[0], pcts[len(pcts) // 2], pcts[-1]] if len(pcts) >= 3 else pcts
    a.axvspan(0, meta_dur / 60, color=GRID, alpha=0.6, lw=0)
    a.text(meta_dur / 120, 0.97, "execution window", transform=a.get_xaxis_transform(),
           ha="center", va="top", color=INK_2, fontsize=9)
    a.axhline(0, color=INK_2, lw=0.8)
    for color, pct in zip(RAMP, show):
        d = paths[paths.pct == pct].groupby("t_rel_s").impact_bps
        m, c = d.mean(), d.apply(ci95)
        t = m.index / 60
        a.fill_between(t, m - c, m + c, color=color, alpha=0.18, lw=0)
        a.plot(t, m, color=color, lw=2, label=f"{round(pct * 100, 1):g}% of window volume")
    a.set_xlabel("minutes from execution start")
    a.set_ylabel("impact vs same-seed baseline (bps)")
    a.set_title("A. Mean impact path (95% CI band)", loc="left", fontsize=10.5)
    a.legend(frameon=False, fontsize=9, loc="lower left")
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_xlim(t.min(), t.max())

    # Panel B: window-average impact vs size
    x = summ.pct * 100
    b.axhline(0, color=INK_2, lw=0.8)
    b.errorbar(x, summ.impact_during, yerr=summ.impact_during_ci, fmt="o", ms=7,
               color=RAMP[1], ecolor=RAMP[1], elinewidth=2, capsize=0,
               markeredgecolor=SURFACE, markeredgewidth=2)
    b.set_xscale("log")
    b.set_xticks(x)
    b.set_xticklabels([f"{round(v, 1):g}%" for v in x])
    b.minorticks_off()
    b.set_xlabel("order size (% of market volume in window, log scale)")
    b.set_ylabel("mean impact during execution (bps)")
    b.set_title("B. Impact vs order size (95% CI)", loc="left", fontsize=10.5)
    b.grid(axis="y", color=GRID, lw=0.8)

    n = int(summ.n.iloc[0])
    fig.suptitle(f"TWAP sell, rmsc04 market, {n} seeds per size", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    out = folder / "impact_curve.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/impact_sweep")
