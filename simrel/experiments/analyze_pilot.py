"""
Phase 3 analysis: stability audit, policy costs, cross-simulator cost and regret matrices.

    python simrel/experiments/analyze_pilot.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simrel.configs.pilot import WORLDS
from simrel.metrics.cost import boot_ci
from simrel.metrics.plotting import AXIS, GRID, HUES, INK, SURF, plt, save

OUT = REPO_ROOT / "results" / "simrel"
REP = REPO_ROOT / "simrel" / "reports"
PRE_MOVE_LIMIT_BPS = 200.0  # pre-treatment stability screen: mid move in the 5 minutes before the task


if __name__ == "__main__":
    rows = pickle.load(open(OUT / "raw" / "pilot_rows.pkl", "rb"))
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("disp_path", "grid_t")} for r in rows])
    if "failed" not in df:
        df["failed"] = None
    worlds = [w for w in WORLDS if w in set(df.world)]
    sizes = sorted(df.size_frac.dropna().unique())

    # ---- stability audit (Gate D), using only information from before the task and from the unperturbed twin
    audit = []
    for w in worlds:
        d = df[df.world == w]
        # a seed fails if its burn-in or twin failed (then every row of it fails); single policy rollouts that fail are counted separately
        seeds = d.groupby("seed").agg(failed=("failed", lambda x: x.notna().mean() >= 0.5), branch_failures=("failed", lambda x: int(x.notna().sum())),
                                      n_rows=("failed", "size"), pre=("pre_move_bps", "max"), twin=("twin_move_bps", "max"),
                                      spread=("twin_spread_bps", "max"), vol=("background_volume", "max"))
        ok = ~seeds.failed
        screened = ok & (seeds.pre.abs() <= PRE_MOVE_LIMIT_BPS)
        audit.append({"world": w, "family": WORLDS[w][0], "seeds_run": len(seeds), "failed": int(seeds.failed.sum()),
                      "screened_out_pre_move": int((ok & ~screened).sum()), "seeds_used": int(screened.sum()),
                      "policy_rollouts": int(seeds.n_rows[screened].sum()), "policy_rollouts_failed": int(seeds.branch_failures[screened].sum()),
                      "twin_move_sd_bps": float(seeds.twin[screened].std()), "twin_move_max_bps": float(seeds.twin[screened].abs().max()),
                      "share_twin_move_gt_300bps": float((seeds.twin[screened].abs() > 300).mean()),
                      "spread_bps": float(seeds.spread[screened].mean()), "volume": float(seeds.vol[screened].mean())})
        df.loc[df.world == w, "use"] = df[df.world == w].seed.map(screened).fillna(False).to_numpy()
    audit = pd.DataFrame(audit)
    audit.round(3).to_csv(REP / "pilot_stability_audit.csv", index=False)
    fail_reasons = df[df.failed.notna()].groupby(["world", "failed"]).seed.nunique().reset_index(name="seeds")
    fail_reasons.to_csv(REP / "pilot_failures.csv", index=False)

    use = df[(df.use == True) & df.failed.isna() & df.is_bps.notna()].copy()
    policies = sorted(use.policy.unique())

    # ---- policy costs
    cost_rows = []
    for (w, s, p), d in use.groupby(["world", "size_frac", "policy"]):
        v = d.is_bps.to_numpy()
        lo, hi = boot_ci(v)
        cost_rows.append({"world": w, "size_frac": s, "policy": p, "n": len(v), "mean_is_bps": v.mean(), "ci_lo": lo, "ci_hi": hi, "median": np.median(v),
                          "p90": np.percentile(v, 90), "p95": np.percentile(v, 95), "impact_bps": d.impact_bps.mean(), "disp_end_bps": d.disp_end_bps.mean(),
                          "unfilled_frac": d.unfilled_frac.mean(), "completion_frac": d.completion_frac.mean(), "half_spread_bps": d.twin_spread_bps.mean() / 2})
    cost = pd.DataFrame(cost_rows)
    cost.round(3).to_csv(REP / "pilot_policy_cost.csv", index=False)

    # ---- selection on the first half of each world's seeds, evaluation on the second half
    def split(w):
        s = sorted(use[use.world == w].seed.unique())
        return s[: len(s) // 2], s[len(s) // 2:]

    def mean_cost(w, s, seeds):
        d = use[(use.world == w) & (use.size_frac == s) & use.seed.isin(seeds)]
        return d.groupby("policy").is_bps.mean()

    def paired(w, s, pa, pb, seeds):
        d = use[(use.world == w) & (use.size_frac == s) & use.seed.isin(seeds)]
        a = d[d.policy == pa].set_index("seed").is_bps
        b = d[d.policy == pb].set_index("seed").is_bps
        return (a - b).dropna().to_numpy()

    chosen, mat = {}, []
    for s in sizes:
        for w in worlds:
            tr, _ = split(w)
            mc_tr = mean_cost(w, s, tr)
            chosen[(s, w)] = mc_tr.idxmin() if len(mc_tr) else None
        for wi in worlds:
            for wj in worlds:
                _, te = split(wj)
                mc = mean_cost(wj, s, te)
                if chosen[(s, wi)] is None or chosen[(s, wi)] not in mc.index:
                    continue
                best_j = mc.idxmin()
                d = paired(wj, s, chosen[(s, wi)], best_j, te)
                lo, hi = boot_ci(d) if len(d) > 1 and chosen[(s, wi)] != best_j else (0.0, 0.0)
                d_tw = paired(wj, s, chosen[(s, wi)], "P0_twap", te)
                hs = use[(use.world == wj) & (use.size_frac == s)].twin_spread_bps.mean() / 2
                mat.append({"size_frac": s, "optimised_in": wi, "evaluated_in": wj, "policy": chosen[(s, wi)], "best_in_eval_world": best_j,
                            "cost_bps": mc[chosen[(s, wi)]], "regret_bps": float(d.mean()) if len(d) else 0.0, "regret_ci_lo": lo, "regret_ci_hi": hi,
                            "regret_half_spreads": (float(d.mean()) if len(d) else 0.0) / hs, "vs_twap_bps": float(d_tw.mean()) if len(d_tw) else 0.0})
    mat = pd.DataFrame(mat)
    mat.round(3).to_csv(REP / "pilot_cross_simulator.csv", index=False)

    # ---- policy rank agreement between worlds
    rk = []
    for s in sizes:
        tab = cost[cost.size_frac == s].pivot(index="policy", columns="world", values="mean_is_bps")
        for i, wi in enumerate(worlds):
            for wj in worlds[i + 1:]:
                rk.append({"size_frac": s, "world_i": wi, "world_j": wj, "spearman": spearmanr(tab[wi], tab[wj]).statistic,
                           "same_family": WORLDS[wi][0] == WORLDS[wj][0]})
    rk = pd.DataFrame(rk)
    rk.round(3).to_csv(REP / "pilot_rank_agreement.csv", index=False)

    summary = {
        "n_rows": int(len(df)), "n_used": int(len(use)), "worlds": worlds, "sizes": sizes,
        "chosen": {f"{s:g}": {w: chosen[(s, w)] for w in worlds} for s in sizes},
        "worst_regret_half_spreads": {f"{s:g}": float(mat[mat.size_frac == s].regret_half_spreads.max()) for s in sizes},
        "mean_spearman_same_family": float(rk[rk.same_family].spearman.mean()), "mean_spearman_cross_family": float(rk[~rk.same_family].spearman.mean()),
    }
    (REP / "pilot_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure 1: intervention against no intervention (mid shift relative to the twin), TWAP and immediate
    s_big = sizes[-1]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.2), sharex=True)
    for ax, w in zip(axes.ravel(), worlds):
        ok_seeds = set(use[use.world == w].seed)
        for pol, hue, lab in (("P0_twap", HUES[0], "TWAP"), ("P1_immediate", HUES[1], "Immediate")):
            paths = [r["disp_path"] for r in rows if r.get("world") == w and r.get("policy") == pol and r.get("size_frac") == s_big and r.get("seed") in ok_seeds and "disp_path" in r]
            if not paths:
                continue
            m = min(len(p) for p in paths)
            t = next(r["grid_t"] for r in rows if r.get("world") == w and "grid_t" in r)[:m]
            arr = np.array([p[:m] for p in paths])
            ax.plot(t / 60, np.median(arr, axis=0), color=hue, lw=2, label=lab)
            ax.fill_between(t / 60, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0), color=hue, alpha=0.13, lw=0)
        ax.axvspan(0, 10, color=GRID, alpha=0.45, lw=0)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(w.replace("_", " ", 1))
    for ax in axes[-1]:
        ax.set_xlabel("minutes from task start (execution shaded)")
    for ax in axes[:, 0]:
        ax.set_ylabel("adverse mid shift vs twin (bps)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Selling {s_big:.0%} of background volume: median and interquartile range across seeds", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, REP / "fig_pilot_intervention")

    # ---- figure 2: cross-simulator regret in half-spreads of the evaluation world
    short = {w: w.split("_")[0] for w in worlds}
    cmap = matplotlib_cmap = __import__("matplotlib").colors.LinearSegmentedColormap.from_list("seq", [SURF, "#86b6ef", "#0d366b"])
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.3 * len(sizes), 3.9))
    vmax = max(mat.regret_half_spreads.clip(lower=0).max(), 1e-9)
    for ax, s in zip(np.atleast_1d(axes), sizes):
        d = mat[mat.size_frac == s]
        m = d.pivot(index="optimised_in", columns="evaluated_in", values="regret_half_spreads").loc[worlds, worlds]
        lo = d.pivot(index="optimised_in", columns="evaluated_in", values="regret_ci_lo").loc[worlds, worlds]
        ax.imshow(np.sqrt(m.clip(lower=0).to_numpy() / vmax), cmap=cmap, vmin=0, vmax=1)
        for i in range(len(worlds)):
            for j in range(len(worlds)):
                v, sig = m.iloc[i, j], lo.iloc[i, j] > 0
                ax.text(j, i, f"{v:.1f}" + ("*" if sig else ""), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > 0.45 * vmax else INK, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(len(worlds)), [short[w] for w in worlds])
        ax.set_yticks(range(len(worlds)), [short[w] for w in worlds])
        ax.set_title(f"Size {s:.0%}: regret in half-spreads")
        ax.set_xlabel("evaluated in")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("policy optimised in")
    fig.tight_layout()
    save(fig, REP / "fig_pilot_regret")

    print(audit.round(2).to_string())
    print(json.dumps(summary, indent=2))
    for s in sizes:
        print(f"--- mean IS (bps), size {s:g}")
        print(cost[cost.size_frac == s].pivot(index="policy", columns="world", values="mean_is_bps")[worlds].round(1).to_string())
        print(mat[mat.size_frac == s].pivot(index="optimised_in", columns="evaluated_in", values="regret_half_spreads").loc[worlds, worlds].round(2).to_string())
