"""
Phase 4 analysis: reference regret of each fitted simulator's recommendation,
and whether observational realism predicts it.

    python simrel/experiments/analyze_reference.py
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

from simrel.metrics.cost import boot_ci
from simrel.configs.reference import WILD_MOVE_BPS
from simrel.metrics.realism import realism_error
from simrel.metrics.plotting import AXIS, HUES, INK2, plt, save
from simrel.reference_worlds.data import registry

RAW = REPO_ROOT / "results" / "simrel" / "raw"
REP = REPO_ROOT / "simrel" / "reports"
SIMS = ["hawkes", "hawkes_linear", "hawkes_sqrt", "abides", "mars"]
MAIN = ["hawkes", "abides", "mars"]  # the three candidate simulators; the Hawkes variants are an ablation
CONTROLS = ["P0_twap", "P2_front_b4", "P3_back_b-2"]  # fixed schedules whose timing terms serve as control variates


def frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.DataFrame(pickle.load(open(path, "rb")))
    if "failed" not in df:
        df["failed"] = None
    return df


if __name__ == "__main__":
    regret_rows, cost_rows, status_rows, rank_rows, pred_rows, pair_rows = [], [], [], [], [], []
    recs_by_sim, wides = {}, {}
    for W in registry():
        D = RAW / f"ref_{W}"
        if not (D / "rows_reference.pkl").exists():
            continue
        info = json.loads((D / "info.json").read_text()) if (D / "info.json").exists() else {}
        # realism is recomputed here from the stored unperturbed twins, so the score definition lives in one place
        heldout = [r for r in pickle.load(open(D / "heldout.pkl", "rb")) if not r.get("failed")]
        for sim in SIMS:
            if (D / f"twins_{sim}.pkl").exists():
                allr = pickle.load(open(D / f"twins_{sim}.pkl", "rb"))
                tw = [r for r in allr if not r.get("failed")]
                info[f"{sim}_twin_fail_share"] = 1.0 - len(tw) / max(len(allr), 1)
                info[f"{sim}_twin_wild_share"] = float(np.mean([r["move_bps_abs"] > WILD_MOVE_BPS for r in tw])) if tw else 1.0
                if len(tw) >= 3:
                    info[f"{sim}_realism_error"], info[f"{sim}_realism_parts"] = realism_error(tw, heldout)
        ref = frame(D / "rows_reference.pkl")
        ref = ref[ref.failed.isna()]
        seeds = sorted(ref.seed.unique())
        A, B = seeds[: len(seeds) // 2], seeds[len(seeds) // 2:]
        for s in sorted(ref.size_frac.unique()):
            r = ref[ref.size_frac == s]
            raw = r.pivot(index="seed", columns="policy", values="is_bps")
            # Variance reduction. Shortfall = impact against the twin + a timing term (the twin's own drift, weighted by
            # when the policy traded). For a fixed schedule the timing term has mean zero, because the reference worlds
            # are symmetric between buying and selling. The timing terms of three fixed schedules are therefore valid
            # control variates for every policy: each policy's shortfall is regressed on them and the fitted part removed.
            timing = pd.DataFrame({c: raw[c] - r[r.policy == c].set_index("seed").impact_bps for c in CONTROLS if c in raw})
            Z = timing.loc[raw.index].to_numpy()
            Zc = np.column_stack([np.ones(len(Z)), Z])
            wide = raw.copy()
            for p_ in raw.columns:
                y = raw[p_].to_numpy()
                ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
                beta = np.linalg.lstsq(Zc[ok], y[ok], rcond=None)[0]
                wide[p_] = y - Z @ beta[1:]
            wide = wide.dropna()
            A_, B_ = [x for x in A if x in wide.index], [x for x in B if x in wide.index]
            A, B = A_, B_
            best = wide.loc[A].mean().idxmin()  # best-in-class, chosen on half the reference seeds
            mean_B = wide.loc[B].mean()
            half_spread = r.twin_spread_bps.mean() / 2
            for p in wide.columns:
                d = (wide.loc[B, p] - wide.loc[B, best]).to_numpy()
                lo, hi = boot_ci(d) if p != best else (0.0, 0.0)
                cost_rows.append({"world": W, "size_frac": s, "policy": p, "ref_cost_bps": mean_B[p], "regret_bps": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                                  "is_best": p == best, "impact_bps": r[r.policy == p].impact_bps.mean(), "raw_is_sd": float(raw[p].std()),
                                  "adjusted_sd": float(wide[p].std()), "n_eval": len(B)})
            ref_rank = wide.mean()
            wides[(W, s)] = (wide, B)

            def add(sim, policy, extra=None):
                d = (wide.loc[B, policy] - wide.loc[B, best]).to_numpy()
                lo, hi = boot_ci(d) if policy != best else (0.0, 0.0)
                row = {"world": W, "size_frac": s, "recommender": sim, "policy": policy, "ref_best": best, "regret_bps": float(d.mean()),
                       "ci_lo": lo, "ci_hi": hi, "regret_half_spreads": float(d.mean()) / half_spread, "ref_cost_best_bps": mean_B[best],
                       "regret_pct_of_best_cost": 100 * float(d.mean()) / max(abs(mean_B[best]), 1e-9)}
                row.update(extra or {})
                regret_rows.append(row)

            add("always_TWAP", "P0_twap")
            add("always_immediate", "P1_immediate")
            rand = float(np.mean([(wide.loc[B, p] - wide.loc[B, best]).mean() for p in wide.columns]))
            regret_rows.append({"world": W, "size_frac": s, "recommender": "random_policy", "policy": "(mean over candidates)", "ref_best": best,
                                "regret_bps": rand, "ci_lo": np.nan, "ci_hi": np.nan, "regret_half_spreads": rand / half_spread,
                                "ref_cost_best_bps": mean_B[best], "regret_pct_of_best_cost": 100 * rand / max(abs(mean_B[best]), 1e-9)})
            worst = float(max((wide.loc[B, p] - wide.loc[B, best]).mean() for p in wide.columns))
            regret_rows.append({"world": W, "size_frac": s, "recommender": "worst_policy", "policy": "(worst candidate)", "ref_best": best,
                                "regret_bps": worst, "ci_lo": np.nan, "ci_hi": np.nan, "regret_half_spreads": worst / half_spread,
                                "ref_cost_best_bps": mean_B[best], "regret_pct_of_best_cost": 100 * worst / max(abs(mean_B[best]), 1e-9)})

            for sim in SIMS:
                sdf = frame(D / f"rows_{sim}.pkl")
                if sdf.empty:
                    continue
                n_seeds = sdf.seed.nunique()
                okdf = sdf[sdf.failed.isna() & sdf.is_bps.notna() & (sdf.size_frac == s)]
                bad_seeds = sdf[sdf.failed.notna()].seed.nunique()
                if okdf.seed.nunique() < 3:
                    continue
                own_wide = okdf.pivot(index="seed", columns="policy", values="is_bps").dropna()
                own = own_wide.mean()
                rec = own.idxmin()
                common = [p for p in wide.columns if p in own.index]
                rho = spearmanr(own[common], ref_rank[common]).statistic
                # policies the simulator cannot distinguish from its favourite (paired 95% interval includes zero)
                tied = [rec] + [p for p in own.index if p != rec and boot_ci((own_wide[p] - own_wide[rec]).to_numpy())[0] <= 0]
                tied_regret = [float((wide.loc[B, p] - wide.loc[B, best]).mean()) for p in tied if p in wide.columns]
                # how well the simulator predicts what each policy costs in the reference world
                err = (own[common] - mean_B[common])
                for p_ in common:
                    pred_rows.append({"world": W, "size_frac": s, "sim": sim, "policy": p_, "predicted_bps": float(own[p_]), "reference_bps": float(mean_B[p_]),
                                      "error_bps": float(err[p_])})
                recs_by_sim[(W, s, sim)] = rec
                add(sim, rec, {"cost_mae_bps": float(err.abs().mean()), "cost_bias_bps": float(err.mean()),
                               "pred_immediate_minus_twap_bps": float(own["P1_immediate"] - own["P0_twap"]),
                               "ref_immediate_minus_twap_bps": float(mean_B["P1_immediate"] - mean_B["P0_twap"]),
                               "realism_error": info.get(f"{sim}_realism_error"), "sim_seeds_ok": int(okdf.seed.nunique()), "sim_seeds_failed": int(bad_seeds),
                               "rank_corr_with_reference": float(rho), "sim_cost_of_rec_bps": float(own[rec]), "sim_cost_twap_bps": float(own.get("P0_twap", np.nan)),
                               "n_tied": len(tied), "tied_set_mean_regret_bps": float(np.mean(tied_regret)), "tied_set_worst_regret_bps": float(np.max(tied_regret)),
                               "tied_set_includes_ref_worst": bool(wide.loc[B].mean().idxmax() in tied),
                               "sim_rank_of_ref_worst": int(own.rank()[wide.loc[B].mean().idxmax()]), "n_policies": len(own)})
        # head-to-head: is one simulator's recommendation significantly cheaper than another's in the reference world?
        for s in sorted(ref.size_frac.unique()):
            have = [x for x in SIMS if (W, s, x) in recs_by_sim]
            for i, a in enumerate(have):
                for b in have[i + 1:]:
                    wd = wides[(W, s)]
                    d = (wd[0].loc[wd[1], recs_by_sim[(W, s, a)]] - wd[0].loc[wd[1], recs_by_sim[(W, s, b)]]).to_numpy()
                    lo, hi = boot_ci(d) if recs_by_sim[(W, s, a)] != recs_by_sim[(W, s, b)] else (0.0, 0.0)
                    ra, rb = info.get(f"{a}_realism_error"), info.get(f"{b}_realism_error")
                    pair_rows.append({"world": W, "size_frac": s, "sim_a": a, "sim_b": b, "a_minus_b_bps": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                                      "realism_a": ra, "realism_b": rb, "more_realistic": a if (ra is not None and rb is not None and ra < rb) else b})
        for sim in SIMS:
            status_rows.append({"world": W, "sim": sim, "status": info.get(f"{sim}_status", "evaluated" if (D / f"rows_{sim}.pkl").exists() else "not run"),
                                "twin_failures": info.get(f"{sim}_twin_failures"), "twin_fail_share": info.get(f"{sim}_twin_fail_share"),
                                "twin_wild_share": info.get(f"{sim}_twin_wild_share"), "failure_reasons": "; ".join(info.get(f"{sim}_twin_failure_reasons") or []),
                                "realism_error": info.get(f"{sim}_realism_error"),
                                "volume": info.get(f"{sim}_volume"), "reference_volume": info.get("reference_volume"),
                                **{f"part_{k}": v for k, v in (info.get(f"{sim}_realism_parts") or {}).items()}})

    regret = pd.DataFrame(regret_rows)
    costs = pd.DataFrame(cost_rows)
    status = pd.DataFrame(status_rows)
    regret.round(3).to_csv(REP / "reference_regret.csv", index=False)
    costs.round(3).to_csv(REP / "reference_policy_cost.csv", index=False)
    status.round(3).to_csv(REP / "reference_simulator_status.csv", index=False)
    pd.DataFrame(pred_rows).round(3).to_csv(REP / "reference_cost_prediction.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.round(3).to_csv(REP / "reference_head_to_head.csv", index=False)

    sims_present = [s for s in MAIN if s in set(regret.recommender)]
    pts = regret[regret.recommender.isin(sims_present) & regret.realism_error.notna()]
    summary = {"n_points": int(len(pts)), "worlds": sorted(regret.world.unique()), "simulators": sims_present}
    if len(pts) >= 4:
        rho = spearmanr(pts.realism_error, pts.regret_half_spreads).statistic
        # permutation test that respects dependence: shuffle realism scores among simulators within each world
        rng = np.random.RandomState(0)
        null = []
        for _ in range(5000):
            perm = pts.copy()
            for w, g in pts.groupby("world"):
                per_sim = g.groupby("recommender").realism_error.first()
                shuffled = dict(zip(per_sim.index, rng.permutation(per_sim.to_numpy())))
                perm.loc[g.index, "realism_error"] = g.recommender.map(shuffled).to_numpy()
            null.append(spearmanr(perm.realism_error, perm.regret_half_spreads).statistic)
        summary["realism_vs_regret_spearman"] = float(rho)
        summary["realism_vs_regret_perm_p"] = float(np.mean(np.abs(np.array(null)) >= abs(rho) - 1e-12))
        # within a (world, size) setting: does the more realistic simulator have the lower regret?
        agree = total = 0
        for _, g in pts.groupby(["world", "size_frac"]):
            g = g.sort_values("realism_error")
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if g.regret_bps.iloc[i] != g.regret_bps.iloc[j]:
                        total += 1
                        agree += g.regret_bps.iloc[i] < g.regret_bps.iloc[j]
        summary["pairs_more_realistic_has_lower_regret"] = [int(agree), int(total)]
        hh = pairs[pairs.sim_a.isin(["hawkes", "abides", "mars"]) & pairs.sim_b.isin(["hawkes", "abides", "mars"])]
        sig = hh[(hh.ci_lo > 0) | (hh.ci_hi < 0)]
        winner = np.where(sig.a_minus_b_bps < 0, sig.sim_a, sig.sim_b)
        summary["head_to_head_significant"] = int(len(sig))
        summary["head_to_head_total"] = int(len(hh))
        summary["significant_wins_by_more_realistic"] = int((winner == sig.more_realistic).sum())
    by = regret.groupby("recommender").agg(mean_regret_half_spreads=("regret_half_spreads", "mean"), max_regret_half_spreads=("regret_half_spreads", "max"),
                                             mean_regret_bps=("regret_bps", "mean"), settings=("regret_bps", "size"),
                                             picked_best=("regret_bps", lambda x: int((x == 0).sum())),
                                             significantly_worse=("ci_lo", lambda x: int((x > 0).sum())))
    by.round(3).to_csv(REP / "reference_regret_by_recommender.csv")
    summary["by_recommender"] = by.round(3).to_dict("index")
    (REP / "reference_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure: reference regret of each recommender, one panel per reference world
    worlds = sorted(regret.world.unique())
    order = sims_present + ["always_TWAP", "always_immediate", "random_policy"]
    hue = dict(zip(["hawkes", "abides", "mars", "always_TWAP", "always_immediate", "random_policy", "hawkes_linear", "hawkes_sqrt"], HUES))
    fig, axes = plt.subplots(1, len(worlds), figsize=(3.3 * len(worlds), 3.3), sharey=False)
    for ax, W in zip(np.atleast_1d(axes), worlds):
        sizes = sorted(regret[regret.world == W].size_frac.unique())
        for k, rec in enumerate(order):
            d = regret[(regret.world == W) & (regret.recommender == rec)].set_index("size_frac")
            if d.empty:
                continue
            x = np.arange(len(sizes)) + (k - (len(order) - 1) / 2) * 0.12
            y = np.array([d.regret_bps.get(s, np.nan) for s in sizes])
            lo = np.array([d.ci_lo.get(s, np.nan) for s in sizes])
            hi = np.array([d.ci_hi.get(s, np.nan) for s in sizes])
            err = np.vstack([np.nan_to_num(y - lo), np.nan_to_num(hi - y)])
            ax.errorbar(x, y, yerr=err, fmt="o", ms=6, color=hue[rec], ecolor=hue[rec], elinewidth=1.2, capsize=0, label=rec.replace("_", " "))
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(range(len(sizes)), [f"{s:.0%}" for s in sizes])
        ax.set_xlim(-0.6, len(sizes) - 0.4)
        ax.set_title(W.replace("_", " ", 1))
        ax.set_xlabel("parent size (share of volume)")
    np.atleast_1d(axes)[0].set_ylabel("reference regret (bps)")
    np.atleast_1d(axes)[-1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, REP / "fig_reference_regret")

    if len(pts) >= 2:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for sim in sims_present:
            d = pts[pts.recommender == sim]
            ax.scatter(d.realism_error, d.regret_half_spreads, s=42, color=hue[sim], edgecolor="#fcfcfb", linewidth=1, label=sim)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlabel("realism error against held-out reference data")
        ax.set_ylabel("reference regret (half-spreads)")
        ax.legend(fontsize=8)
        save(fig, REP / "fig_realism_vs_regret")

    print(json.dumps(summary, indent=2))
    print(regret[["world", "size_frac", "recommender", "policy", "ref_best", "regret_bps", "ci_lo", "ci_hi", "regret_half_spreads", "realism_error", "rank_corr_with_reference"]].round(2).to_string())
