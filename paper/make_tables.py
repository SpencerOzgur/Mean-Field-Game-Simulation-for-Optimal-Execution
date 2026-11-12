"""
Turn the CSV/JSON outputs in results/simbench/ into the LaTeX fragments the
paper inputs. Every number in the paper comes from this script.

    python paper/make_tables.py
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "simbench"
GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"

WORLD_DESC = {
    "A1_rmsc04": "RMSC04 reference: 1000 noise, 102 value, 12 momentum traders, 2 market makers",
    "A2_no_mm": "no market makers",
    "A3_momentum": "36 momentum traders ($3\\times$)",
    "A4_few_value": "40 value traders (weaker anchoring, less flow)",
    "A5_deep_mm": "4 market makers at $4\\times$ participation",
    "A6_volatile": "$4\\times$ fundamental volatility",
}
POL = {"exp_b-2": "Schedule $b=-2$ (back-loaded)", "exp_b0": "Schedule $b=0$ (TWAP)", "exp_b2": "Schedule $b=2$",
       "exp_b4": "Schedule $b=4$", "exp_b8": "Schedule $b=8$ (front-loaded)", "immediate": "Immediate",
       "pov": "POV", "urgency": "Urgency rule"}


def tex(name: str) -> str:
    return name.replace("_", "\\_")


def short(w: str) -> str:
    return w.split("_")[0]


def write(name: str, body: str):
    (GEN / f"{name}.tex").write_text(body)


def table(cols: str, header: str, rows: list, caption: str, label: str, size="\\small", star=False, place="!htbp") -> str:
    env = "table*" if star else "table"
    return (f"\\begin{{{env}}}[{place}]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) +
            f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def f(x, d=1):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for p in (RES / "figures").glob("*.pdf"):
        shutil.copy(p, FIG / p.name)

    F = json.loads((RES / "findings.json").read_text())
    worlds, sizes = F["worlds"], F["sizes"]
    vol = json.loads((RES / "world_volume.json").read_text())
    gates = json.loads((RES / "gates.json").read_text())
    long = pd.read_csv(RES / "table_policy_cost_long.csv")
    transfer = pd.read_csv(RES / "table_transfer.csv")
    mfg = pd.read_csv(RES / "table_mfg_calibration.csv")
    mfgx = pd.read_csv(RES / "table_mfg_transfer.csv")
    ranks = pd.read_csv(RES / "table_rank_agreement.csv")
    shape = pd.read_csv(RES / "table_impact_shape.csv")
    obs = pd.read_csv(RES / "table_observables.csv", index_col=0)
    dist = pd.read_csv(RES / "table_observational_distance.csv", index_col=0)
    real = json.loads((RES / "real_reference.json").read_text()) if (RES / "real_reference.json").exists() else {}

    # ---- worlds
    rows = [f"{short(w)} & {WORLD_DESC[w]} & {vol[w]:,.0f} \\\\" for w in worlds]
    write("tab_worlds", table("llr", "World & Background population (change from RMSC04) & Volume", rows,
          "ABIDES worlds. Volume is mean background shares traded 10:00--10:30 over unperturbed runs.", "tab:worlds"))

    # ---- observables
    names = [("obs_spread_bps", "Quoted spread (bps)", 2), ("obs_depth5", "Depth, top 5 levels (shares)", 0),
             ("obs_vol10s_bps", "10\\,s return volatility (bps)", 2), ("obs_ret_acf1", "Return autocorrelation, lag 1", 2),
             ("obs_absret_acf1", "$|$return$|$ autocorrelation, lag 1", 2), ("obs_kurtosis", "Excess kurtosis", 1),
             ("obs_trade_size", "Mean trade size (shares)", 0), ("obs_ofi_acf1", "Order-flow imbalance autocorr.", 2),
             ("move_sd_bps", "30-min mid move, s.d. (bps)", 1), ("move_gt1pct", "Share of runs with $|$move$|>1\\%$", 2)]
    r_ = real.get("REAL_INTC_1000_1030", {})
    rows = [f"{lab} & " + " & ".join(f(obs.loc[w, k], d) for w in worlds) + f" & {f(r_.get(k, np.nan), d)} \\\\" for k, lab, d in names]
    free = ["obs_spread_bps", "obs_vol10s_bps", "obs_ret_acf1", "obs_absret_acf1", "obs_kurtosis", "obs_ofi_acf1"]
    if r_:
        sd = obs.loc[worlds, free].std(ddof=0)
        err = {w: float(np.sqrt((((obs.loc[w, free] - pd.Series({k: r_[k] for k in free})) / sd) ** 2).mean())) for w in worlds}
        rows.append("\\midrule")
        rows.append("Distance to INTC (scale-free statistics) & " + " & ".join(f(err[w], 2) for w in worlds) + " & 0 \\\\")
        (RES / "realism_distance.json").write_text(json.dumps(err, indent=2))
    write("tab_observables", table("l" + "r" * (len(worlds) + 1), "Statistic & " + " & ".join(short(w) for w in worlds) + " & INTC",
          rows, "Stylised facts of each unperturbed world over the task window (means over 200 seeds; 199 in A3). "
          "INTC is the real LOBSTER sample day (21 June 2012, 10:00--10:30), a single reference point. "
          "Distance to INTC: root-mean-square difference over spread, volatility, the two return autocorrelations, kurtosis and order-flow autocorrelation, "
          "each scaled by its standard deviation across worlds.", "tab:obs", size="\\footnotesize", star=True))

    # ---- policy cost, one panel per size
    order = [p for p in POL if p in set(long.policy)]
    rows = []
    for s in sizes:
        rows.append(f"\\multicolumn{{{len(worlds) + 1}}}{{l}}{{\\emph{{Parent order = {s:.0%} of background volume}}}} \\\\".replace("%", "\\%"))
        sub = long[long.size_frac == s].set_index(["policy", "world"])
        best = {w: min(order, key=lambda p: sub.loc[(p, w), "cost_bps"]) for w in worlds}
        for p in order:
            cells = []
            for w in worlds:
                r = sub.loc[(p, w)]
                c = f"{r.cost_bps:.1f}{{\\tiny$\\,\\pm${(r.ci_hi - r.ci_lo) / 2:.1f}}}"
                cells.append(f"\\textbf{{{c}}}" if best[w] == p else c)
            rows.append(f"{POL[p]} & " + " & ".join(cells) + " \\\\")
        if s != sizes[-1]:
            rows.append("\\addlinespace")
    write("tab_cost", table("l" + "r" * len(worlds), "Policy & " + " & ".join(short(w) for w in worlds), rows,
          "Paired impact cost of selling, in basis points (mean $\\pm$ half-width of the 95\\% bootstrap interval over seeds). "
          "Bold marks the lowest mean in each column; many differences within a column are inside the intervals.", "tab:cost", size="\\footnotesize", star=True))

    # ---- estimator noise: raw shortfall vs paired cost under TWAP
    rows = []
    for s in sizes:
        sub = long[(long.size_frac == s) & (long.policy == "exp_b0")].set_index("world")
        rows.append(f"{s:.0%} & raw shortfall & ".replace("%", "\\%") + " & ".join(f(sub.loc[w, "raw_is_sd"], 1) for w in worlds) + " \\\\")
        rows.append(" & paired cost & " + " & ".join(f(sub.loc[w, "cost_sd_bps"], 1) for w in worlds) + " \\\\")
    write("tab_noise", table("ll" + "r" * len(worlds), "Size & Estimator & " + " & ".join(short(w) for w in worlds), rows,
          "Standard deviation across seeds (bps) of raw implementation shortfall and of the paired impact cost, TWAP sell. "
          "Pairing removes most of the noise in the stable worlds and little of it where paths diverge quickly.", "tab:noise", size="\\footnotesize"))

    # ---- decisions: simulation-optimal schedule, MFG-calibrated schedule
    rows = []
    for s in sizes:
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{Parent order = {s:.0%} of background volume}}}} \\\\".replace("%", "\\%"))
        for w in worlds:
            m = mfg[(mfg.size_frac == s) & (mfg.world == w)].iloc[0]
            clip = "$^\\ast$" if abs(m.b_mfg) in (2.0, 8.0) else ""
            rows.append(f"{short(w)} & {f(m.a_temp_bps, 2)} & {f(m.lambda_perm_bps, 1)} [{f(m.lambda_lo, 1)}, {f(m.lambda_hi, 1)}] & {f(m.b_mfg, 1)}{clip} & {f(m.b_star_sim, 0)} & "
                        f"{f(m.cost_mfg_own, 1)} & {f(m.cost_twap_own, 1)} & {f(m.cost_best_sched_own, 1)} \\\\")
    write("tab_mfg", table("lrrrrrrr", "World & $a$ & $\\lambda$ [95\\% CI] & $b_{\\mathrm{MFG}}$ & $b^\\star_{\\mathrm{sim}}$ & "
          "$C(b_{\\mathrm{MFG}})$ & $C(\\mathrm{TWAP})$ & $C(b^\\star)$", rows,
          "MFG calibration per world (training seeds) and the schedule it prescribes, against the schedule chosen by direct search "
          "($b^\\star_{\\mathrm{sim}}$, training seeds). $a$: temporary impact at the TWAP rate (bps); $\\lambda$: mean mid displacement at the deadline "
          "under TWAP (bps). Costs are test-seed means in bps. $^\\ast$ clipped to the grid range $[-2,8]$.", "tab:mfg", size="\\footnotesize", star=True))

    # ---- each alternative against TWAP (paired)
    vt = pd.read_csv(RES / "table_vs_twap.csv").set_index(["size_frac", "policy", "world"])
    rows = []
    for s in sizes:
        rows.append(f"\\multicolumn{{{len(worlds) + 1}}}{{l}}{{\\emph{{Parent order = {s:.0%} of background volume}}}} \\\\".replace("%", "\\%"))
        for p in [q for q in order if q != "exp_b0"]:
            cells = []
            for w in worlds:
                r = vt.loc[(s, p, w)]
                c = f"{r.diff_bps:+.1f}"
                cells.append(f"\\textbf{{{c}}}" if (r.ci_lo > 0 or r.ci_hi < 0) else c)
            rows.append(f"{POL[p]} & " + " & ".join(cells) + " \\\\")
        if s != sizes[-1]:
            rows.append("\\addlinespace")
    write("tab_vs_twap", table("l" + "r" * len(worlds), "Policy minus TWAP & " + " & ".join(short(w) for w in worlds), rows,
          "Paired difference in impact cost between each policy and TWAP, in bps (positive: worse than TWAP), over all seeds. "
          "Bold: the 95\\% bootstrap interval of the paired difference excludes zero.", "tab:vstwap", size="\\footnotesize", star=True))
    sig = vt.reset_index()
    sig["sig"] = (sig.ci_lo > 0) | (sig.ci_hi < 0)

    # ---- rank agreement, for all worlds and for the stable subset
    stable = F["stable_worlds"]
    rows = []
    for label, keep in (("All six worlds", worlds), (f"Stable worlds ({', '.join(short(w) for w in stable)})", stable)):
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for s in sizes:
            d = ranks[(ranks.size_frac == s) & ranks.world_i.isin(keep) & ranks.world_j.isin(keep)]
            rows.append(f"{s:.0%} & {d.spearman.mean():.2f} & {d.spearman.min():.2f} & {d.spearman.max():.2f} & {d.pairs_flipped.mean():.1f} / {int(d.n_policy_pairs.iloc[0])} & "
                        f"{int(d.pairs_flipped_significant.sum())} / {int(d.n_policy_pairs.sum())} \\\\".replace("%", "\\%"))
    write("tab_rank", table("lrrrrr", "Size & Mean $\\rho$ & Min $\\rho$ & Max $\\rho$ & Mean flipped pairs & Significant reversals", rows,
          "Agreement of policy rankings between worlds, over world pairs. $\\rho$: Spearman correlation of the eight policies' mean costs. "
          "A policy pair is \\emph{flipped} if its order differs between two worlds, and a \\emph{significant reversal} if both paired 95\\% intervals exclude zero with opposite signs.", "tab:rank"))

    # ---- transfer regret matrices
    for s in sizes:
        d = transfer[transfer.size_frac == s]
        rows = []
        for wi in worlds:
            cells = []
            for wj in worlds:
                r = d[(d.train_world == wi) & (d.eval_world == wj)].iloc[0]
                sig = "$^\\dagger$" if (r.regret_ci_lo > 0 or r.regret_ci_hi < 0) else ""
                cells.append("0" if wi == wj else f"{r.regret_bps:.1f}{sig}")
            rows.append(f"{short(wi)} ($b^\\star\\!=\\!{d[d.train_world == wi].b_star_train.iloc[0]:g}$) & " + " & ".join(cells) + " \\\\")
        write(f"tab_regret_{int(round(s * 100))}", table("l" + "r" * len(worlds), "Optimised in $\\downarrow$ / evaluated in $\\rightarrow$ & " + " & ".join(short(w) for w in worlds), rows,
              f"Transfer regret $R_{{ij}}$ in bps, parent order {s:.0%} of background volume (test seeds). ".replace("%", "\\%") +
              "$^\\dagger$ paired 95\\% interval excludes zero.", f"tab:regret{int(round(s * 100))}", size="\\footnotesize", place="H"))

    # ---- impact shape and asymmetry
    rows = []
    for r in shape.to_dict("records"):
        cells = [f"{r['cost_exponent_sell']:.2f}" if r["world"] in F["stable_worlds"] else "--"]
        for s in sizes:
            cells.append(f"{r[f'buy_minus_sell_{s:g}']:.1f} [{r[f'bms_lo_{s:g}']:.1f}, {r[f'bms_hi_{s:g}']:.1f}]")
        rows.append(f"{short(r['world'])} & " + " & ".join(cells) + " \\\\")
    write("tab_impact", table("lr" + "r" * len(sizes), "World & Size exponent (sell) & " + " & ".join(f"Buy$-$sell, {s:.0%}".replace("%", "\\%") for s in sizes), rows,
          "Shape of TWAP impact cost. Size exponent: log--log slope of mean sell cost against size over the three sizes (1 = linear, 0.5 = square-root); "
          "a point estimate from three sizes, not reported for unstable worlds. Buy$-$sell: paired difference in cost (bps) with 95\\% interval.", "tab:impact", size="\\footnotesize", star=True))

    # ---- macros
    rq = F["rq4"]
    n_sig = sum(int(((transfer[transfer.size_frac == s].regret_ci_lo > 0)).sum()) for s in sizes)
    n_off = len(worlds) * (len(worlds) - 1) * len(sizes)
    clipped = int(((mfg.b_mfg.abs() == 8.0) | (mfg.b_mfg.abs() == 2.0)).sum())
    macros = {
        "nWorlds": len(worlds), "nSeeds": F["n_seeds"], "nTrain": F["n_train"], "nTest": F["n_test"],
        "nEpisodes": f"{F['n_episodes_run']:,}", "sizesText": ", ".join(f"{s:.0%}" for s in sizes[:-1]).replace("%", "\\%") + f" and {sizes[-1]:.0%}".replace("%", "\\%"),
        "bGridText": ", ".join(f"{b:g}" for b in sorted({float(p.replace('exp_b', '')) for p in order if p.startswith('exp_b')})),
        "rqRhoPooled": f"{rq['pooled']['spearman']:.2f}", "rqPPooled": f"{rq['pooled']['perm_p_two_sided']:.2f}",
        "nSigRegret": n_sig, "nOffDiag": n_off, "nMfgClipped": clipped, "nMfgCells": len(mfg),
        "nGatesPassed": sum(all(v for k, v in g.items() if k.startswith("G")) for g in gates.values()),
    }
    ts = transfer[(transfer.train_world != transfer.eval_world) & transfer.train_world.isin(stable) & transfer.eval_world.isin(stable)]
    macros.update({"nStable": len(stable), "stableList": ", ".join(short(w) for w in stable),
                   "unstableList": ", ".join(short(w) for w in worlds if w not in stable) or "none",
                   "maxRegretStable": f"{ts.regret_bps.max():.1f}", "meanRegretStable": f"{ts.regret_bps.mean():.1f}",
                   "nSigRegretStable": int((ts.regret_ci_lo > 0).sum()), "nOffDiagStable": len(ts),
                   "rqRhoStable": f"{rq.get('pooled_stable', {}).get('spearman', float('nan')):.2f}",
                   "rqNStable": rq.get("pooled_stable", {}).get("n_pairs", 0)})
    for s in sizes:
        tag = {0.01: "One", 0.05: "Five", 0.2: "Twenty"}.get(s, f"{int(s * 100)}")
        d = transfer[(transfer.size_frac == s) & (transfer.train_world != transfer.eval_world)]
        macros[f"maxRegret{tag}"] = f"{d.regret_bps.max():.1f}"
        macros[f"meanRegret{tag}"] = f"{d.regret_bps.mean():.1f}"
        macros[f"rhoMean{tag}"] = f"{ranks[ranks.size_frac == s].spearman.mean():.2f}"
        macros[f"rqRho{tag}"] = f"{rq[f'{s:g}']['spearman']:.2f}"
        macros[f"rqP{tag}"] = f"{rq[f'{s:g}']['perm_p_two_sided']:.2f}"
    body = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    tb, tp = F.get("timeouts_baselines", []), F.get("timeouts_policies", [])
    if tb or tp:
        where = ", ".join(sorted({f"{short(t[0])} seed {t[1]}" for t in tb + tp}))
        run = (f"In total, {len(tb)} unperturbed and {len(tp)} intervention episodes hit the wall-clock limit on every attempt ({where}); "
               f"they are excluded together with all other episodes of the same seed, leaving {F['n_seeds']} of {F['n_seeds_run']} seeds in the main analysis.")
    else:
        run = "No episode in the main experiment hit the limit."
    body += f"\\newcommand{{\\runawayText}}{{{run}}}\n"

    # ---- TRADES arm (optional: only once that stage has run)
    tr_path = RES / "table_short_horizon_all.csv"
    if tr_path.exists():
        wide = pd.read_csv(tr_path, index_col=0)
        both = pd.concat([pd.read_csv(RES / "table_short_horizon_abides.csv"), pd.read_csv(RES / "table_short_horizon_trades.csv")]).set_index(["policy", "world"])
        cols = list(wide.columns)
        sched = [p for p in ["exp_b-2", "exp_b0", "exp_b2", "exp_b8"] if p in wide.index]
        rows = []
        for p in sched + ["immediate"]:
            cells = []
            for w in cols:
                r = both.loc[(p, w)]
                c = f"{r.cost_bps:.1f}{{\\tiny$\\,\\pm${(r.ci_hi - r.ci_lo) / 2:.1f}}}"
                cells.append(f"\\textbf{{{c}}}" if p in sched and wide.loc[sched, w].idxmin() == p else c)
            rows.append(f"{POL[p]} & " + " & ".join(cells) + " \\\\")
        n_row = "Episodes per cell & " + " & ".join(str(int(both.loc[("exp_b0", w), "n"])) for w in cols) + " \\\\"
        write("tab_short", table("l" + "r" * len(cols), "Policy & " + " & ".join(short(w) for w in cols), rows + ["\\midrule", n_row],
              "The 10-minute task (sell 10\\% of background volume in 20 slices) in every ABIDES world and in the retrained TRADES market T1. "
              "Paired impact cost in bps, mean $\\pm$ 95\\% half-width. Bold: cheapest schedule in each column.", "tab:short", size="\\footnotesize", star=True))
        cmp = pd.read_csv(RES / "table_trades_vs_real.csv", index_col=0)
        lab = {"obs_spread_bps": "Quoted spread (bps)", "obs_vol10s_bps": "10\\,s return volatility (bps)", "obs_ret_acf1": "Return autocorrelation, lag 1",
               "obs_absret_acf1": "$|$return$|$ autocorrelation, lag 1", "obs_volume": "Traded volume (shares)"}
        rows = [f"{lab[k]} & {f(cmp.loc[k, 'TRADES_retrained_mean'], 2 if 'volume' not in k else 0)} & {f(cmp.loc[k, 'TRADES_retrained_sd'], 2 if 'volume' not in k else 0)} & "
                f"{f(cmp.loc[k, 'REAL_INTC_0946_0956'], 2 if 'volume' not in k else 0)} \\\\" for k in lab if k in cmp.index]
        write("tab_trades_real", table("lrrr", "Statistic (09:46--09:56) & TRADES mean & TRADES s.d. & Real INTC", rows,
              "Unperturbed retrained-TRADES market against the real day it was trained on, over the task window. The real window lies in the training split.", "tab:tradesreal"))
        T = json.loads((RES / "findings_trades.json").read_text())
        body += f"\\newcommand{{\\tradesSeeds}}{{{T['n_twins']}}}\n\\newcommand{{\\nTradesEpisodes}}{{{T['n_twins'] + T['n_runs']}}}\n"
        body += f"\\newcommand{{\\tradesBest}}{{{POL[T['best_schedule']['T1_trades_intc']].split(' (')[0]}}}\n"
        body += f"\\newcommand{{\\tradesQty}}{{{T['parent_qty']:,}}}\n\\newcommand{{\\tradesWall}}{{{T['mean_wall_s'] / 60:.0f}}}\n"
        sp, vs = T["spread_path_bps"], T["vs_twap"]
        tm = {"tradesSpreadReplay": f"{sp['replay_last_minute']:.1f}", "tradesSpreadFirst": f"{sp['first_minute']:.1f}",
              "tradesSpreadMid": f"{sp['minute_5']:.1f}", "tradesSpreadLast": f"{sp['last_minute']:.1f}",
              "tradesPreIdentical": f"{100 * T['pre_start_identical_frac']:.0f}\\%", "tradesRuns": T["n_runs"]}
        for key, tag in (("exp_b8", "Front"), ("immediate", "Imm"), ("exp_b-2", "Back"), ("exp_b2", "Two")):
            if key in vs:
                tm[f"tradesDiff{tag}"] = f"{vs[key]['diff_bps']:+.1f}"
                tm[f"tradesDiff{tag}CI"] = f"[{vs[key]['ci_lo']:.1f}, {vs[key]['ci_hi']:.1f}]"
                tm[f"tradesDiff{tag}N"] = f"{vs[key]['n'] - vs[key]['n_positive']} of {vs[key]['n']}"
        w = wide["T1_trades_intc"]
        tm.update({"tradesCostTwap": f"{w['exp_b0']:.1f}", "tradesCostFront": f"{w['exp_b8']:.1f}", "tradesCostImm": f"{w['immediate']:.1f}"})
        tsh = pd.read_csv(RES / "table_short_horizon_trades.csv").set_index("policy")
        ash = pd.read_csv(RES / "table_short_horizon_abides.csv")
        ash = ash[ash.world.isin(stable) & (ash.policy == "exp_b0")]
        tm.update({"tradesDispTwap": f"{tsh.loc['exp_b0', 'disp_end_bps']:.0f}", "tradesDispTwapPost": f"{tsh.loc['exp_b0', 'disp_post_bps']:.0f}",
                   "tradesDispMin": f"{tsh.disp_end_bps.min():.0f}", "tradesDispMax": f"{tsh.disp_end_bps.max():.0f}",
                   "abidesShortDispMax": f"{ash.disp_end_bps.abs().max():.1f}"})
        rg = pd.read_csv(RES / "table_short_horizon_regret.csv", index_col=0)  # rows: selected in, columns: evaluated in
        t2a = rg.loc["T1_trades_intc", stable]
        a2t = rg.loc[stable, "T1_trades_intc"]
        tm.update({"tradesToAbidesMin": f"{t2a.min():.1f}", "tradesToAbidesMax": f"{t2a.max():.1f}",
                   "abidesToTradesMin": f"{a2t.min():.1f}", "abidesToTradesMax": f"{a2t.max():.1f}",
                   "abidesShortBest": ", ".join(f"{short(w)}: $b={T['best_schedule'][w].replace('exp_b', '')}$" for w in stable)})
        body += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in tm.items())
    write("numbers", body)
    print(body)
