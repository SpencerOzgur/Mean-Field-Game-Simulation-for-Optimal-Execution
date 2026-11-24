"""
Figures for the manuscript, from the committed result files only.

    python paper_unreliable_advice/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "simrel" / "goal2"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
# Okabe-Ito: colourblind-safe categorical palette, fixed order
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "grey": "#666666", "black": "#000000"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42,
                     "font.family": "sans-serif", "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL = 3.33  # single ACM column, inches
rng = np.random.default_rng(20261124)


def boot_ci(v, B=2000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fig_goal1() -> None:
    D = {"INTC": [("Hawkes original", 3.53, 1.94, 5.31), ("Hawkes queue", 21.50, 16.32, 26.72), ("Hawkes refill", 0.35, -0.60, 1.40),
                  ("ABIDES original", -4.10, -4.50, -3.68), ("ABIDES calibrated", -0.74, -0.95, -0.54)],
         "CSCO": [("Hawkes original", 2.20, 0.88, 3.58), ("Hawkes queue", 8.82, 4.87, 13.16), ("Hawkes refill", 0.93, 0.34, 1.54),
                  ("ABIDES original", -4.95, -5.50, -4.39), ("ABIDES calibrated", -0.74, -1.00, -0.46)],
         "MSFT": [("Hawkes original", 2.23, 0.72, 3.81), ("Hawkes queue", 0.03, -1.96, 2.02), ("Hawkes refill", -0.49, -1.25, 0.25),
                  ("ABIDES original", -2.36, -3.88, -0.69), ("ABIDES calibrated", 0.09, -2.38, 2.57)]}
    names = [r[0] for r in D["INTC"]]
    col = {"Hawkes original": C["blue"], "Hawkes queue": C["sky"], "Hawkes refill": C["green"], "ABIDES original": C["vermillion"], "ABIDES calibrated": C["orange"]}
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    stocks = list(D)
    for si, s in enumerate(stocks):
        for k, (n, e, lo, hi) in enumerate(D[s]):
            y = si * (len(names) + 1.2) + k
            ax.plot([max(lo, -7), min(hi, 12)], [y, y], color=col[n], lw=1.4, solid_capstyle="round")
            ax.plot([min(e, 12)], [y], "o", ms=3.2, color=col[n], mec="white", mew=0.5)
            if e > 12:
                ax.annotate(f"{e:.1f} \u2192", (12, y), xytext=(-22, -2), textcoords="offset points", fontsize=5.5, color=C["grey"], ha="right")
    ax.axvline(0, color=C["black"], lw=0.6)
    ax.set_yticks([si * (len(names) + 1.2) + 2 for si in range(3)])
    ax.set_yticklabels([f"{s}\n({'discovery' if s == 'INTC' else 'confirmatory'})" for s in stocks])
    ax.invert_yaxis()
    ax.set_xlim(-7, 13)
    ax.set_xlabel("D = cost(TWAP) − cost(sell at once), bp   (>0: advises selling at once)")
    handles = [plt.Line2D([], [], color=col[n], marker="o", ms=3, lw=1.2, label=n) for n in names]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "fig1_goal1.pdf")
    plt.close(fig)


def fig_budget() -> None:
    b = pd.read_csv(RES / "addendum9" / "regret_by_budget_summary.csv")
    rm = pd.read_csv(RES / "addendum9" / "removed_mechanism.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2.1, 1.65), sharey=False)
    ax = axes[0]
    g = b[b.group == "agent-mix"].sort_values("seeds")
    ax.axhline(g["always TWAP"].iloc[0], color=C["grey"], ls="--", lw=0.9, label="always TWAP")
    for c, lab, colr in (("oracle", "true generator", C["green"]), ("single candidate", "fitted candidate", C["vermillion"]), ("ensemble", "candidates' ensemble", C["orange"])):
        ax.plot(g.seeds, g[c], "-o", ms=2.4, color=colr, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 30, 100]); ax.set_xticklabels(["5", "10", "30", "100"])
    ax.set_xlabel("simulation runs per decision"); ax.set_ylabel("regret (bp)")
    ax.set_title("(a) agent-mix markets", loc="left")
    ax.set_ylim(0, 11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.4)
    ax = axes[1]
    m = rm.groupby(["simulator", "seeds"]).regret.mean().unstack(0)
    for c, lab, colr in (("oracle", "true generator", C["green"]), ("oracle-no-fund", "− anchoring", C["vermillion"]),
                         ("oracle-no-trend", "− trend followers", C["blue"]), ("oracle-no-withdraw", "− withdrawal", C["purple"])):
        ax.plot(m.index, m[c], "-o", ms=2.4, color=colr, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(["10", "30", "100", "300"])
    ax.set_xlabel("simulation runs per decision")
    ax.set_title("(b) one mechanism removed", loc="left")
    ax.set_ylim(0, 13)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.4)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig2_budget.pdf")
    plt.close(fig)


def fig_testb() -> None:
    t = pd.read_csv(RES / "addendum10" / "test_b.csv")
    arms = ["no-simulator", "always TWAP", "rule", "corrected", "original h_c_f_m_k1", "compound Hawkes"]
    labels = {"no-simulator": "no-simulator rule", "always TWAP": "always TWAP", "rule": "frozen trust rule", "corrected": "corrected Hawkes",
              "original h_c_f_m_k1": "original Hawkes", "compound Hawkes": "compound Hawkes"}
    col = {"no-simulator": C["grey"], "always TWAP": C["black"], "rule": C["purple"], "corrected": C["green"], "original h_c_f_m_k1": C["blue"], "compound Hawkes": C["sky"]}
    fams = ["agent-mix", "latent", "queue-reactive"]
    per = t.groupby(["family", "arm", "world"]).regret.mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(COL * 2.1, 1.45), sharey=True)
    for ax, fam in zip(axes, fams):
        for k, a in enumerate(arms):
            e, lo, hi = boot_ci(per[(per.family == fam) & (per.arm == a)].regret)
            ax.plot([lo, hi], [k, k], color=col[a], lw=1.4)
            ax.plot([e], [k], "o", ms=3.2, color=col[a], mec="white", mew=0.5)
        ax.axvline(0, color=C["black"], lw=0.5)
        ax.set_title(f"{fam} (16 markets)", loc="left")
        ax.set_xlabel("cross-fitted regret (bp)")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels([labels[a] for a in arms])
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig3_testb.pdf")
    plt.close(fig)


def fig_own() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.0), sharey=True)
    from simrel.experiments import goal2_reliability as R
    for ax, (fn, title, keep) in zip(axes, (("own_execution.csv", "(a) development (22)", "dev"), ("own_execution_test_a.csv", "(b) Test A (5)", "testa"))):
        t = pd.read_csv(RES / "exploratory" / fn)
        t = t[(t.family == "agent-mix") & (t.world.isin(R.dev_markets() if keep == "dev" else R.TEST_A))]
        per = t.groupby(["rule", "n", "world"]).regret.mean().reset_index()
        tw = boot_ci(per[per.rule == "always TWAP"].regret)
        ax.axhspan(tw[1], tw[2], color=C["grey"], alpha=0.15, lw=0)
        ax.axhline(tw[0], color=C["grey"], ls="--", lw=0.9, label="always TWAP")
        for rule, lab, colr in (("own data", "own executions only", C["blue"]), ("hybrid (ensemble)", "with simulator prior", C["vermillion"])):
            g = per[per.rule == rule]
            ns = sorted(g.n.unique())
            st = [boot_ci(g[g.n == n].regret) for n in ns]
            ax.fill_between(ns, [s[1] for s in st], [s[2] for s in st], color=colr, alpha=0.15, lw=0)
            ax.plot(ns, [s[0] for s in st], "-o", ms=2.4, color=colr, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("regret (bp)")
    fig.supxlabel("own executions per strategy (12 strategies; agent-mix markets)", fontsize=7, y=0.13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "fig4_own.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    fig_goal1()
    fig_budget()
    fig_testb()
    fig_own()
    print(sorted(p.name for p in FIG.glob("*.pdf")))
