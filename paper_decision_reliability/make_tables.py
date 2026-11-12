"""
Build the LaTeX fragments for the paper from simrel/reports/. Every number in
the paper comes from here.

    python paper_decision_reliability/make_tables.py
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "simrel" / "reports"
RAW = ROOT / "results" / "simrel" / "raw"
GEN = ROOT / "paper_decision_reliability" / "generated"
FIG = ROOT / "paper_decision_reliability" / "figures"

POL = {"P0_twap": "P0 TWAP", "P1_immediate": "P1 Immediate", "P2_front_b2": "P2 Front-loaded ($b{=}2$)", "P2_front_b4": "P2 Front-loaded ($b{=}4$)",
       "P3_back_b-2": "P3 Back-loaded", "P4_pov_1x": "P4 POV ($1\\times$)", "P4_pov_2x": "P4 POV ($2\\times$)", "P5_liquidity": "P5 Liquidity-aware",
       "P6_signal_momentum": "P6 Signal (momentum)", "P6_signal_reversal": "P6 Signal (reversal)", "P7_risk_0.5": "P7 Risk-aware (0.5)", "P7_risk_2": "P7 Risk-aware (2)"}
SHORTPOL = {"P0_twap": "TWAP", "P1_immediate": "Immediate", "P2_front_b2": "Front $b{=}2$", "P2_front_b4": "Front $b{=}4$", "P3_back_b-2": "Back",
            "P4_pov_1x": "POV $1\\times$", "P4_pov_2x": "POV $2\\times$", "P5_liquidity": "Liquidity", "P6_signal_momentum": "Momentum",
            "P6_signal_reversal": "Reversal", "P7_risk_0.5": "Risk 0.5", "P7_risk_2": "Risk 2"}
SIMNAME = {"hawkes": "Hawkes", "hawkes_linear": "Hawkes (linear)", "hawkes_sqrt": "Hawkes (sqrt)", "abides": "ABIDES", "mars": "MarS",
           "always_TWAP": "Always TWAP", "always_immediate": "Always immediate", "random_policy": "Random candidate", "worst_policy": "Worst candidate"}
WORLD_DESC = {"R1a_qr_resilient": "Queue-reactive, deep and resilient", "R1b_qr_fragile": "Queue-reactive, thin, flow chases imbalance",
              "R2a_am_anchored": "Agent mix, strong fundamental anchor", "R2b_am_trending": "Agent mix, trend followers dominate"}


def short(w):
    return w.split("_")[0]


def f(x, d=1):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pct(x):
    return f"{100 * x:.0f}\\%"


def table(cols, header, rows, caption, label, size="\\footnotesize", place="!htbp"):
    # the tabular is shrunk to the text width only if it would otherwise overflow
    return (f"\\begin{{table}}[{place}]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{%\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def write(name, body):
    (GEN / f"{name}.tex").write_text(body)


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for p in REP.glob("fig_*.pdf"):
        shutil.copy(p, FIG / p.name)
    M = {}

    # ---- gates
    rows = ["ABIDES & Pass & Pass & Pass$^a$ & Pass$^b$ & Pass \\\\",
            "MarS (2M, 10M) & Pass$^c$ & Pass & \\textbf{Fail} & \\textbf{Conditional} & Pass \\\\",
            "Hawkes order book & Pass & Pass & Pass & Pass & Pass \\\\"]
    write("tab_gates", table("lccccc", "Simulator & A Runnable & B Interactive & C Comparable & D Stable & E Reproducible", rows,
          "Feasibility gates. $^a$After setting the price level so the tick matches. $^b$For the configurations used; others were unstable in an earlier pilot. "
          "$^c$On CPU, outside the supported Docker/CUDA set-up.", "tab:gates", size="\\small"))

    # ---- MarS pilot facts
    v = json.loads((REP / "pilot_mars_validation.json").read_text())
    o = json.loads((REP / "pilot_mars_official.json").read_text())
    st = json.loads((REP / "pilot_mars_stability.json").read_text())
    early = [abs(w["ret_bps"]) for x in st if x["model"].endswith("10m") for w in x["windows"][1:4] if w]
    M.update({"marsNllTwo": f"{v['Don-Don/mars-order-2m']['nll']:.2f}", "marsNllFive": f"{v['Don-Don/mars-order-5m']['nll']:.2f}",
              "marsNllTen": f"{v['Don-Don/mars-order-10m']['nll']:.2f}", "marsNllUniform": f"{v['uniform_nll']:.2f}",
              "marsOfficialBase": f"{o['mean_ret_bps_baseline']:.0f}", "marsOfficialTwap": f"{o['mean_ret_bps_twap']:,.0f}",
              "marsEarlyShare": pct(float(np.mean(np.array(early) > 200)))})

    # ---- pilot
    audit = pd.read_csv(REP / "pilot_stability_audit.csv")
    cost = pd.read_csv(REP / "pilot_policy_cost.csv")
    cross = pd.read_csv(REP / "pilot_cross_simulator.csv")
    rk = pd.read_csv(REP / "pilot_rank_agreement.csv")
    PS = json.loads((REP / "pilot_summary.json").read_text())
    vol = json.loads((ROOT / "results" / "simrel" / "pilot_volume.json").read_text())
    worlds, sizes = PS["worlds"], PS["sizes"]
    M["sizesText"] = " and ".join(pct(s) for s in sizes)
    rows = []
    for r in audit.itertuples():
        vf = vol[r.world]
        rows.append(f"{short(r.world)} & {r.family} & {vf['n'] + vf['failed']} & {vf['failed']} & {r.seeds_run} & {r.failed} & {r.screened_out_pre_move} & {r.seeds_used} & "
                    f"{f(r.twin_move_sd_bps, 0)} & {pct(r.share_twin_move_gt_300bps)} & {f(r.spread_bps, 1)} \\\\")
    write("tab_pilot_audit", table("llrrrrrrrrr", "World & Family & \\multicolumn{2}{c}{Volume pass} & \\multicolumn{4}{c}{Main pass (seeds)} & Move s.d. & $>$3\\% & Spread \\\\\n"
          " & & run & failed & run & failed & screened & used & (bps) & & (bps)", rows,
          "Stability audit of the pilot. Volume pass: unperturbed runs used to size the parent order. Main pass: a seed fails if its burn-in or its unperturbed twin "
          "fails; it is screened out if the mid moved more than 200\\,bps in the five minutes before the task. Last three columns describe the unperturbed twin over the "
          "ten-minute task in the seeds used: standard deviation of the mid move, share of seeds moving more than 3\\%, and mean quoted spread.", "tab:audit"))
    for w in worlds:
        a = audit[audit.world == w].iloc[0]
        tag = short(w).replace("1", "One").replace("2", "Two").replace("5", "Five").replace("0", "Zero")
        M[f"used{tag}"] = int(a.seeds_used)
    mars = [w for w in worlds if w.startswith("M")]
    M["marsVolFail"] = " and ".join(f"{vol[w]['failed']} of {vol[w]['n'] + vol[w]['failed']}" for w in mars)
    M["marsMainFail"] = " and ".join(f"{int(audit[audit.world == w].failed.iloc[0]) + int(audit[audit.world == w].screened_out_pre_move.iloc[0])} of {int(audit[audit.world == w].seeds_run.iloc[0])}" for w in mars)

    order = [p for p in POL if p in set(cost.policy)]
    for s in sizes:
        sub = cost[cost.size_frac == s].set_index(["policy", "world"])
        rows = []
        for p in order:
            cells = []
            for w in worlds:
                if (p, w) not in sub.index:
                    cells.append("--")
                    continue
                r = sub.loc[(p, w)]
                c = f"{r.mean_is_bps:.1f}{{\\tiny$\\,\\pm${(r.ci_hi - r.ci_lo) / 2:.1f}}}"
                best = min((q for q in order if (q, w) in sub.index), key=lambda q: sub.loc[(q, w), "mean_is_bps"])
                cells.append(f"\\textbf{{{c}}}" if best == p else c)
            rows.append(f"{POL[p]} & " + " & ".join(cells) + " \\\\")
        rows += ["\\midrule", "Seeds & " + " & ".join(str(int(sub.loc[("P0_twap", w), "n"])) if ("P0_twap", w) in sub.index else "--" for w in worlds) + " \\\\",
                 "Half-spread (bps) & " + " & ".join(f(sub.loc[("P0_twap", w), "half_spread_bps"], 1) if ("P0_twap", w) in sub.index else "--" for w in worlds) + " \\\\"]
        write(f"tab_pilot_cost_{int(round(s * 100))}", table("l" + "r" * len(worlds), "Policy & " + " & ".join(short(w) for w in worlds), rows,
              f"Pilot: implementation shortfall in bps (mean $\\pm$ 95\\% half-width) for a parent order of {pct(s)} of background volume. Bold: lowest mean in the column. "
              "Costs are not comparable across columns: the worlds have different price levels and tick sizes.", f"tab:pilotcost{int(round(s * 100))}"))

    rows = []
    for s in sizes:
        rows.append(f"\\multicolumn{{{len(worlds) + 2}}}{{l}}{{\\emph{{Parent order = {pct(s)} of background volume}}}} \\\\")
        d = cross[cross.size_frac == s]
        for wi in worlds:
            cells = []
            for wj in worlds:
                r = d[(d.optimised_in == wi) & (d.evaluated_in == wj)].iloc[0]
                cells.append("0" if wi == wj and r.regret_half_spreads == 0 else f"{r.regret_half_spreads:.1f}" + ("$^\\dagger$" if r.regret_ci_lo > 0 else ""))
            rows.append(f"{short(wi)} & {SHORTPOL[d[d.optimised_in == wi].policy.iloc[0]]} & " + " & ".join(cells) + " \\\\")
    write("tab_pilot_cross", table("ll" + "r" * len(worlds), "Optimised in & Chosen policy & " + " & ".join(short(w) for w in worlds), rows,
          "Pilot: cross-simulator regret. Rows: world in which the policy was selected (first half of its seeds). Columns: world in which it is evaluated (second half). "
          "Entries are regret relative to the best candidate in the evaluation world, in units of that world's half-spread. $^\\dagger$paired 95\\% interval above zero. "
          "A diagonal entry above zero means the policy selected on training seeds was not the best on test seeds.", "tab:cross"))
    M.update({"rankSame": f"{PS['mean_spearman_same_family']:.2f}", "rankCross": f"{PS['mean_spearman_cross_family']:.2f}",
              "nSigCross": int((cross.regret_ci_lo > 0).sum()), "nCross": int((cross.optimised_in != cross.evaluated_in).sum())})

    # ---- reference experiment
    reg = pd.read_csv(REP / "reference_regret.csv")
    pc = pd.read_csv(REP / "reference_policy_cost.csv")
    status = pd.read_csv(REP / "reference_simulator_status.csv")
    RS = json.loads((REP / "reference_summary.json").read_text())
    rworlds = sorted(reg.world.unique())
    rsizes = sorted(reg.size_frac.unique())

    rows = []
    for w in rworlds:
        for s in rsizes:
            d = pc[(pc.world == w) & (pc.size_frac == s)]
            best = d[d.is_best].iloc[0]
            worse = int((d.ci_lo > 0).sum())
            tw, im = d[d.policy == "P0_twap"].iloc[0], d[d.policy == "P1_immediate"].iloc[0]
            worst = d.sort_values("regret_bps").iloc[-1]
            rows.append(f"{short(w)} & {pct(s)} & {SHORTPOL[best.policy]} & {best.ref_cost_bps:.1f} & {tw.regret_bps:+.1f} & {im.regret_bps:+.1f} & "
                        f"{SHORTPOL[worst.policy]} ({worst.regret_bps:+.1f}) & {worse} / {len(d) - 1} & {d.raw_is_sd.mean():.0f} & {d.adjusted_sd.mean():.0f} \\\\")
    write("tab_ref_truth", table("llrrrrlrrr", "World & Size & Best & Cost & TWAP & Immediate & Worst candidate & Sig.\\ worse & Raw s.d. & Adj.\\ s.d.", rows,
          "Reference worlds: the truth each simulator is judged against. Best: lowest-cost candidate on half the reference seeds; its cost and all regrets (bps) are "
          "measured on the other half. Sig.\\ worse: candidates whose paired 95\\% interval lies above the best. Raw and adjusted s.d.: per-seed standard deviation of "
          "shortfall before and after the control-variate adjustment.", "tab:reftruth"))
    rows = [f"{short(w)} & {WORLD_DESC.get(w, w)} \\\\" for w in rworlds]
    write("tab_ref_worlds", table("ll", "World & Mechanism and regime", rows, "Reference worlds.", "tab:refworlds", size="\\small"))

    sims = [s for s in ["hawkes", "abides", "mars"] if s in set(status.sim)]
    rows = []
    for w in rworlds:
        for sim in ["hawkes", "abides", "mars"]:
            stt = status[(status.world == w) & (status.sim == sim)]
            if stt.empty or str(stt.status.iloc[0]) == "not run":
                continue
            r = stt.iloc[0]
            evaluated = str(r.status).startswith("evaluated")
            ratio = r.volume / r.reference_volume if np.isfinite(r.volume) else np.nan
            rows.append(f"{short(w)} & {SIMNAME[sim]} & {pct(r.twin_fail_share) if np.isfinite(r.twin_fail_share) else '--'} & "
                        f"{pct(r.twin_wild_share) if np.isfinite(r.twin_wild_share) else '--'} & {'yes' if evaluated else '\\textbf{no}'} & "
                        f"{f(r.realism_error, 2) if evaluated else '--'} & {f(ratio, 2)} \\\\")
    write("tab_ref_sims", table("llrrlrr", "World & Simulator & Runs failed & Moves $>$3\\% & Evaluated & Realism error & Volume ratio", rows,
          "Fitted simulators in each reference world. Runs failed and moves above 3\\%: unperturbed runs of the fitted simulator. A simulator is evaluated only if at most "
          "half fail and at most a quarter of the rest move more than 3\\% in ten minutes. Realism error: mean scaled distance of six stylised facts from held-out "
          "reference data (lower is better; about 1 is within the reference world's own run-to-run variation). Volume ratio: simulator volume over reference volume.",
          "tab:refsims"))

    main = reg[reg.recommender.isin(["hawkes", "abides", "mars"])]
    rows = []
    for w in rworlds:
        for s in rsizes:
            for sim in ["hawkes", "abides", "mars"]:
                d = main[(main.world == w) & (main.size_frac == s) & (main.recommender == sim)]
                if d.empty:
                    continue
                r = d.iloc[0]
                sig = "$^\\dagger$" if r.ci_lo > 0 else ""
                rows.append(f"{short(w)} & {pct(s)} & {SIMNAME[sim]} & {SHORTPOL[r.policy]} & {SHORTPOL[r.ref_best]} & {r.regret_bps:+.1f}{sig} & [{r.ci_lo:.1f}, {r.ci_hi:.1f}] & "
                            f"{int(r.n_tied)} & {r.tied_set_worst_regret_bps:+.1f} & {r.rank_corr_with_reference:.2f} & {f(r.realism_error, 2)} \\\\")
        rows.append("\\addlinespace")
    write("tab_ref_regret", table("lllllrrrrrr", "World & Size & Simulator & Recommends & Ref.\\ best & Regret & 95\\% CI & Tied & Worst tied & Rank $\\rho$ & Realism", rows[:-1],
          "Reference regret of each fitted simulator's recommendation (bps; $^\\dagger$interval above zero). Tied: number of candidates the simulator cannot distinguish "
          "from its favourite; Worst tied: the largest reference regret among them. Rank $\\rho$: Spearman correlation between the simulator's ranking of the twelve "
          "candidates and the reference world's.", "tab:refregret"))

    by = pd.read_csv(REP / "reference_regret_by_recommender.csv", index_col=0)
    rows = []
    for k in ["hawkes", "hawkes_linear", "hawkes_sqrt", "abides", "mars", "always_TWAP", "always_immediate", "random_policy", "worst_policy"]:
        if k in by.index:
            r = by.loc[k]
            rows.append(f"{SIMNAME[k]} & {int(r.settings)} & {r.mean_regret_bps:.1f} & {r.mean_regret_half_spreads:.2f} & {r.max_regret_half_spreads:.2f} & {int(r.significantly_worse)} \\\\")
    write("tab_ref_summary", table("lrrrrr", "Recommender & Settings & Mean regret (bps) & Mean (half-spreads) & Max (half-spreads) & Sig.\\ worse than best", rows,
          "Reference regret by recommender, over all reference worlds and sizes in which it was evaluated. The last four rows are simulator-free baselines.",
          "tab:refsummary", size="\\small"))

    hv = reg[reg.recommender.isin(["hawkes", "hawkes_linear", "hawkes_sqrt"])]
    if hv.recommender.nunique() > 1:
        rows = []
        for w in rworlds:
            for s in rsizes:
                cells = []
                for k in ["hawkes", "hawkes_linear", "hawkes_sqrt"]:
                    d = hv[(hv.world == w) & (hv.size_frac == s) & (hv.recommender == k)]
                    cells.append("--" if d.empty else f"{SHORTPOL[d.policy.iloc[0]]} ({d.regret_bps.iloc[0]:+.1f}{'$^\\dagger$' if d.ci_lo.iloc[0] > 0 else ''})")
                rows.append(f"{short(w)} & {pct(s)} & " + " & ".join(cells) + " \\\\")
        real = []
        for k in ["hawkes", "hawkes_linear", "hawkes_sqrt"]:
            r = status[status.sim == k].realism_error
            real.append(f"{r.mean():.2f}" if len(r) and r.notna().any() else "--")
        rows += ["\\midrule", "\\multicolumn{2}{l}{Mean realism error} & " + " & ".join(real) + " \\\\"]
        write("tab_hawkes_variants", table("llrrr", "World & Size & Capped linear & Linear & Square-root", rows,
              "Three Hawkes simulators that differ only in how strongly a market order excites later flow as a function of its size, each fitted to the same reference data. "
              "Entries: recommended policy and its reference regret in bps ($^\\dagger$interval above zero).", "tab:hawkesvariants"))
        M["hasVariants"] = 1

    pts = reg[reg.recommender.isin(RS.get("simulators", [])) & reg.realism_error.notna()]
    M.update({"refPoints": RS.get("n_points", 0), "refRho": f"{RS.get('realism_vs_regret_spearman', float('nan')):.2f}",
              "refRhoP": f"{RS.get('realism_vs_regret_perm_p', float('nan')):.2f}",
              "refPairsAgree": (RS.get("pairs_more_realistic_has_lower_regret") or [0, 0])[0], "refPairsTotal": (RS.get("pairs_more_realistic_has_lower_regret") or [0, 0])[1]})
    cfgs = __import__("importlib").import_module
    import sys

    sys.path.insert(0, str(ROOT))
    from simrel.configs import reference as C

    M.update({"nRefSeeds": C.N_REFERENCE, "nFitEpisodes": C.N_FIT_EPISODES, "nHawkesSeeds": C.N_SIM["hawkes"], "nAbidesSeeds": C.N_SIM["abides"],
              "nRefWorlds": len(rworlds), "nPilotRows": f"{PS['n_rows']:,}", "nPilotUsed": f"{PS['n_used']:,}", "nRefRows": f"{len(rworlds) * C.N_REFERENCE * len(rsizes) * 12:,}"})
    write("numbers", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    print("\n".join(f"{k} = {v}" for k, v in M.items()))
