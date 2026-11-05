"""
Stage 2: tables and figures from the baseline and policy episodes.

Run from repo root (after stages 00 and 01):
    python experiments/simbench_02_analyze.py
"""

from itertools import combinations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simbench.analysis import baseline_index, boot_ci, displacement_paths, to_frame
from simbench.runner import RESULTS_DIR, load

FIG_DIR = RESULTS_DIR / "figures"
OBS = ["obs_spread_bps", "obs_depth5", "obs_vol10s_bps", "obs_ret_acf1", "obs_absret_acf1", "obs_kurtosis",
       "obs_volume", "obs_trade_size", "obs_ofi_std", "obs_ofi_acf1"]
COST = "cost_pen_bps"
B_CLIP = (-2.0, 8.0)


def b_of(policy: str) -> float:
    return float(policy.replace("exp_b", ""))


def paired_diff(df, world, size, pa, pb, seeds):
    """Per-seed cost difference pa - pb in one world (same seeds, so noise cancels)."""
    d = df[(df.world == world) & (df.size_frac == size) & (df.side == "sell") & df.seed.isin(seeds)]
    a = d[d.policy == pa].set_index("seed")[COST]
    b = d[d.policy == pb].set_index("seed")[COST]
    return (a - b).dropna().to_numpy()


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = baseline_index()
    recs = load("policies")
    timeouts = [(r["world"], r["seed"], r["policy"], r["size_frac"]) for r in recs if r.get("timeout")]
    base_timeouts = [(r["world"], r["seed"]) for r in load("baselines") if r.get("timeout")]
    n_run = len(recs) + len(load("baselines"))
    df = to_frame(recs, base)
    worlds = sorted(df.world.unique())
    sizes = sorted(df.size_frac.unique())
    # keep only seeds that are complete for every (world, size, policy, side) cell
    n_cells = df.groupby("seed").size()
    seeds = sorted(n_cells[n_cells == n_cells.max()].index)
    df = df[df.seed.isin(seeds)]
    recs = [r for r in recs if r["seed"] in set(seeds)]
    train, test = seeds[: len(seeds) // 2], seeds[len(seeds) // 2 :]
    findings = {"n_episodes_run": n_run, "timeouts_policies": timeouts, "timeouts_baselines": base_timeouts, "n_seeds_run": int(len(n_cells)),
                "n_seeds": len(seeds), "n_train": len(train), "n_test": len(test), "n_episodes": int(len(df)),
                "worlds": worlds, "sizes": sizes}
    sell = df[df.side == "sell"]
    sched = sorted([p for p in sell.policy.unique() if p.startswith("exp_b")], key=b_of)
    b_grid = np.array([b_of(p) for p in sched])
    policies = sched + ["immediate", "pov", "urgency"]

    # ---------------- 1. observables and world-to-world observational distance
    bdf = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in base.values()])
    obs = bdf.groupby("world")[OBS].mean().loc[worlds]
    stab = bdf.groupby("world")["mid_move_end_bps"].agg(move_sd_bps="std", move_gt1pct=lambda x: float((x.abs() > 100).mean()))
    obs.join(stab).round(4).to_csv(RESULTS_DIR / "table_observables.csv")
    # a world is "stable" if no unperturbed run moves the mid by more than 1% in the 30-minute window
    stable = [w for w in worlds if stab.loc[w, "move_gt1pct"] == 0.0]
    findings["stable_worlds"] = stable
    z = (obs - obs.mean()) / obs.std(ddof=0)
    dist = pd.DataFrame(
        [[np.sqrt(((z.loc[i] - z.loc[j]) ** 2).mean()) for j in worlds] for i in worlds], index=worlds, columns=worlds)
    dist.round(3).to_csv(RESULTS_DIR / "table_observational_distance.csv")

    # ---------------- 2. policy cost by world and size
    g = sell.groupby(["size_frac", "policy", "world"])
    cost = g[COST].mean().unstack("world").loc[(slice(None), policies), :]
    cost.round(2).to_csv(RESULTS_DIR / "table_policy_cost.csv")
    rows = []
    for (s, p, w), d in g:
        lo, hi = boot_ci(d[COST].to_numpy())
        rows.append({"size_frac": s, "policy": p, "world": w, "cost_bps": d[COST].mean(), "ci_lo": lo, "ci_hi": hi,
                     "cost_median_bps": d[COST].median(), "cost_sd_bps": d[COST].std(),
                     "raw_is_bps": d["is_bps"].mean(), "raw_is_sd": d["is_bps"].std(), "fill_frac": d["fill_frac"].mean(),
                     "slip_bps": d["slip_bps"].mean(), "disp_end_bps": d["disp_end_bps"].mean(),
                     "disp_post_bps": d["disp_post_bps"].mean(), "n": len(d)})
    long = pd.DataFrame(rows)
    long.round(3).to_csv(RESULTS_DIR / "table_policy_cost_long.csv", index=False)

    # ---------------- 3. policy rankings: agreement and significant reversals between worlds
    rank_rows = []
    for s in sizes:
        c = cost.loc[s]
        for wi, wj in combinations(worlds, 2):
            rho = spearmanr(c[wi], c[wj]).statistic
            flips = sig = 0
            for pa, pb in combinations(policies, 2):
                di, dj = paired_diff(df, wi, s, pa, pb, seeds), paired_diff(df, wj, s, pa, pb, seeds)
                if np.sign(di.mean()) != np.sign(dj.mean()):
                    flips += 1
                    (li, hi_), (lj, hj) = boot_ci(di), boot_ci(dj)
                    if (li > 0 or hi_ < 0) and (lj > 0 or hj < 0):
                        sig += 1
            rank_rows.append({"size_frac": s, "world_i": wi, "world_j": wj, "spearman": rho,
                              "pairs_flipped": flips, "pairs_flipped_significant": sig,
                              "n_policy_pairs": len(policies) * (len(policies) - 1) // 2})
    ranks = pd.DataFrame(rank_rows)
    ranks.round(3).to_csv(RESULTS_DIR / "table_rank_agreement.csv", index=False)
    best = cost.groupby(level=0).idxmin().apply(lambda col: col.map(lambda t: t[1]))
    best.to_csv(RESULTS_DIR / "table_best_policy.csv")

    # ---------------- 3b. each alternative against TWAP, paired by seed, all seeds
    v_rows = []
    for s in sizes:
        for w in worlds:
            for p in policies:
                if p == "exp_b0":
                    continue
                d = paired_diff(df, w, s, p, "exp_b0", seeds)
                lo, hi = boot_ci(d)
                v_rows.append({"size_frac": s, "world": w, "policy": p, "diff_bps": d.mean(), "ci_lo": lo, "ci_hi": hi,
                               "median_diff_bps": float(np.median(d)), "n": len(d)})
    vs_twap = pd.DataFrame(v_rows)
    vs_twap.round(3).to_csv(RESULTS_DIR / "table_vs_twap.csv", index=False)

    # ---------------- 4. schedule optimised in world i, evaluated in world j
    tr = sell[sell.seed.isin(train)].groupby(["size_frac", "world", "policy"])[COST].mean()
    te = sell[sell.seed.isin(test)].groupby(["size_frac", "world", "policy"])[COST].mean()
    bstar = {(s, w): min(sched, key=lambda p: tr[(s, w, p)]) for s in sizes for w in worlds}
    t_rows = []
    for s in sizes:
        for wi in worlds:
            for wj in worlds:
                d = paired_diff(df, wj, s, bstar[(s, wi)], bstar[(s, wj)], test)
                lo, hi = boot_ci(d) if wi != wj and bstar[(s, wi)] != bstar[(s, wj)] else (0.0, 0.0)
                t_rows.append({"size_frac": s, "train_world": wi, "eval_world": wj, "b_star_train": b_of(bstar[(s, wi)]),
                               "cost_bps": te[(s, wj, bstar[(s, wi)])], "regret_bps": float(d.mean()) if len(d) else 0.0,
                               "regret_ci_lo": lo, "regret_ci_hi": hi})
    transfer = pd.DataFrame(t_rows)
    transfer.round(3).to_csv(RESULTS_DIR / "table_transfer.csv", index=False)

    # ---------------- 5. MFG calibration per world and its prescribed schedule
    arr = {k: v["arrival_mid"] for k, v in base.items()}
    m_rows = []
    for s in sizes:
        for w in worlds:
            u, sl = [], []
            for r in recs:
                if r["world"] != w or r["size_frac"] != s or r["side"] != "sell" or r["seed"] not in train or not r["policy"].startswith("exp_b"):
                    continue
                fq = r["fill_qty"].astype(np.float64)
                ok = fq > 0
                px = r["fill_notional"][ok] / fq[ok]
                u.append(fq[ok] / (r["total_qty"] / len(fq)))
                sl.append((r["slice_mid"].astype(np.float64)[ok] - px) / arr[(w, r["seed"])] * 1e4)
            u, sl = np.concatenate(u), np.concatenate(sl)
            keep = np.isfinite(u) & np.isfinite(sl)  # a one-sided book has no mid
            a, c0 = np.polyfit(u[keep], sl[keep], 1)  # bps per unit of normalised trading rate; intercept ~ half spread
            d = sell[(sell.world == w) & (sell.size_frac == s) & (sell.policy == "exp_b0") & sell.seed.isin(train)]
            lam, lam_post = d["disp_end_bps"].mean(), d["disp_post_bps"].mean()
            lam_lo, lam_hi = boot_ci(d["disp_end_bps"].to_numpy())
            b_mfg = float(np.clip(lam / (2 * a), *B_CLIP)) if a > 0 else B_CLIP[1]
            m_rows.append({"size_frac": s, "world": w, "a_temp_bps": a, "half_spread_bps": c0, "lambda_perm_bps": lam, "lambda_lo": lam_lo, "lambda_hi": lam_hi,
                           "lambda_post_bps": lam_post, "decay_ratio": lam_post / lam if lam else np.nan,
                           "b_mfg": b_mfg, "b_ac_risk_neutral": 0.0, "b_star_sim": b_of(bstar[(s, w)])})
    mfg = pd.DataFrame(m_rows)

    def interp_cost(s, w, b):  # test cost of a schedule off the grid, by linear interpolation in b
        return float(np.interp(b, b_grid, [te[(s, w, p)] for p in sched]))

    mfg["cost_mfg_own"] = [interp_cost(r.size_frac, r.world, r.b_mfg) for r in mfg.itertuples()]
    mfg["cost_twap_own"] = [te[(r.size_frac, r.world, "exp_b0")] for r in mfg.itertuples()]
    mfg["cost_best_sched_own"] = [min(te[(r.size_frac, r.world, p)] for p in sched) for r in mfg.itertuples()]
    mfg["mfg_regret_own"] = mfg.cost_mfg_own - mfg.cost_best_sched_own
    mfg.round(3).to_csv(RESULTS_DIR / "table_mfg_calibration.csv", index=False)
    x_rows = []
    for s in sizes:
        for wi in worlds:
            bi = mfg[(mfg.size_frac == s) & (mfg.world == wi)].b_mfg.iloc[0]
            for wj in worlds:
                bj = mfg[(mfg.size_frac == s) & (mfg.world == wj)].b_mfg.iloc[0]
                x_rows.append({"size_frac": s, "calib_world": wi, "eval_world": wj, "b_mfg": bi,
                               "regret_bps": interp_cost(s, wj, bi) - interp_cost(s, wj, bj)})
    mfg_transfer = pd.DataFrame(x_rows)
    mfg_transfer.round(3).to_csv(RESULTS_DIR / "table_mfg_transfer.csv", index=False)

    # ---------------- 6. impact: size dependence, decay, buy/sell asymmetry (TWAP)
    tw = df[df.policy == "exp_b0"]
    imp = tw.groupby(["world", "side", "size_frac"])[[COST, "disp_end_bps", "disp_post_bps", "slip_bps", "fill_frac"]].mean()
    imp.round(3).to_csv(RESULTS_DIR / "table_impact_twap.csv")
    i_rows = []
    for w in worlds:
        row = {"world": w}
        for side in ("sell", "buy"):
            c = imp.loc[(w, side)]
            row[f"cost_exponent_{side}"] = np.polyfit(np.log(sizes), np.log(np.clip(c[COST].to_numpy(), 1e-6, None)), 1)[0]
        for s in sizes:
            d = (tw[(tw.world == w) & (tw.size_frac == s) & (tw.side == "buy")].set_index("seed")[COST]
                 - tw[(tw.world == w) & (tw.size_frac == s) & (tw.side == "sell")].set_index("seed")[COST]).dropna().to_numpy()
            lo, hi = boot_ci(d)
            row[f"buy_minus_sell_{s:g}"], row[f"bms_lo_{s:g}"], row[f"bms_hi_{s:g}"] = d.mean(), lo, hi
        i_rows.append(row)
    pd.DataFrame(i_rows).round(3).to_csv(RESULTS_DIR / "table_impact_shape.csv", index=False)

    # ---------------- 7. does observational distance predict transfer regret?
    off = transfer[transfer.train_world != transfer.eval_world].copy()
    off["obs_dist"] = [dist.loc[a, b] for a, b in zip(off.train_world, off.eval_world)]
    rq4 = {}
    rng = np.random.RandomState(0)
    for label, d in [(f"{s:g}", off[off.size_frac == s]) for s in sizes] + [("pooled", off.groupby(["train_world", "eval_world"], as_index=False)[["regret_bps", "obs_dist"]].mean())]:
        rho = spearmanr(d.obs_dist, d.regret_bps).statistic
        null = []
        for _ in range(5000):  # permute world labels of the distance matrix (pairs are not independent)
            perm = dict(zip(worlds, rng.permutation(worlds)))
            dd = [dist.loc[perm[a], perm[b]] for a, b in zip(d.train_world, d.eval_world)]
            null.append(spearmanr(dd, d.regret_bps).statistic)
        null = np.array(null)
        rq4[label] = {"spearman": float(rho), "perm_p_two_sided": float(np.mean(np.abs(null) >= abs(rho) - 1e-12)), "n_pairs": int(len(d))}
    off.round(3).to_csv(RESULTS_DIR / "table_distance_vs_regret.csv", index=False)
    so = off[off.train_world.isin(stable) & off.eval_world.isin(stable)].groupby(["train_world", "eval_world"], as_index=False)[["regret_bps", "obs_dist"]].mean()
    if len(so) >= 4:
        rq4["pooled_stable"] = {"spearman": float(spearmanr(so.obs_dist, so.regret_bps).statistic), "n_pairs": int(len(so))}

    findings.update({
        "rq4": rq4,
        "b_star": {f"{s:g}": {w: b_of(bstar[(s, w)]) for w in worlds} for s in sizes},
        "max_regret_bps": {f"{s:g}": float(transfer[transfer.size_frac == s].regret_bps.max()) for s in sizes},
        "mean_offdiag_regret_bps": {f"{s:g}": float(off[off.size_frac == s].regret_bps.mean()) for s in sizes},
        "rank_spearman_min": {f"{s:g}": float(ranks[ranks.size_frac == s].spearman.min()) for s in sizes},
        "rank_spearman_mean": {f"{s:g}": float(ranks[ranks.size_frac == s].spearman.mean()) for s in sizes},
        "significant_reversals_total": {f"{s:g}": int(ranks[ranks.size_frac == s].pairs_flipped_significant.sum()) for s in sizes},
    })
    (RESULTS_DIR / "findings.json").write_text(json.dumps(findings, indent=2))

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    color = dict(zip(worlds, HUES))
    short = {w: w.split("_", 1)[0] for w in worlds}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "legend.frameon": False})

    def save(fig, name):
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # F1: cost of each schedule, one panel per world (cost scales differ by orders of magnitude);
    # band = 95% bootstrap interval, ringed marker = schedule chosen on training seeds
    lk = long.set_index(["size_frac", "world", "policy"])
    size_hue = dict(zip(sizes, HUES))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), sharex=True)
    for ax, w in zip(axes.ravel(), worlds):
        for s in sizes:
            y = np.array([lk.loc[(s, w, p), "cost_bps"] for p in sched])
            lo = np.array([lk.loc[(s, w, p), "ci_lo"] for p in sched])
            hi = np.array([lk.loc[(s, w, p), "ci_hi"] for p in sched])
            ax.fill_between(b_grid, lo, hi, color=size_hue[s], alpha=0.13, lw=0)
            ax.plot(b_grid, y, color=size_hue[s], lw=2, label=f"{s:.0%} of volume")
            k = sched.index(bstar[(s, w)])
            ax.plot([b_grid[k]], [y[k]], "o", ms=8, color=size_hue[s], mec=SURF, mew=2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(w.replace("_", " ", 1))
        ax.set_xticks(b_grid)
    for ax in axes[-1]:
        ax.set_xlabel("front-loading b  (0 = TWAP)")
    for ax in axes[:, 0]:
        ax.set_ylabel("paired impact cost (bps)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, "fig_cost_vs_schedule")

    # F2: transfer regret matrices
    def slog(v):  # signed log scale for colour only
        return np.sign(v) * np.log10(1 + np.abs(v))

    vmax = slog(np.abs(transfer.regret_bps).max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.7 * len(sizes), 3.5))
    for ax, s in zip(np.atleast_1d(axes), sizes):
        d = transfer[transfer.size_frac == s]
        m = d.pivot(index="train_world", columns="eval_world", values="regret_bps").loc[worlds, worlds]
        lo = d.pivot(index="train_world", columns="eval_world", values="regret_ci_lo").loc[worlds, worlds]
        hi = d.pivot(index="train_world", columns="eval_world", values="regret_ci_hi").loc[worlds, worlds]
        ax.imshow(slog(m.to_numpy()), cmap=cmap, vmin=-vmax, vmax=vmax)
        for i in range(len(worlds)):
            for j in range(len(worlds)):
                v = m.iloc[i, j]
                sig = i != j and (lo.iloc[i, j] > 0 or hi.iloc[i, j] < 0)
                ax.text(j, i, f"{v:.1f}" + ("*" if sig else ""), ha="center", va="center", fontsize=7.5, color=INK,
                        fontweight="bold" if sig else "normal")
        ax.set_xticks(range(len(worlds)), [short[w] for w in worlds])
        ax.set_yticks(range(len(worlds)), [short[w] for w in worlds])
        ax.set_title(f"Size {s:.0%}: regret (bps)")
        ax.set_xlabel("evaluated in")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("schedule optimised in")
    fig.tight_layout()
    save(fig, "fig_transfer_regret")

    # F3: observational distance against transfer regret
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for s, mk in zip(sizes, ["o", "s", "^"]):
        d = off[off.size_frac == s]
        ax.scatter(d.obs_dist, d.regret_bps, s=34, marker=mk, color=HUES[sizes.index(s)], edgecolor=SURF, linewidth=1, label=f"size {s:.0%}")
    ax.set_xlabel("observational distance between worlds")
    ax.set_ylabel("transfer regret (bps)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(0, color=AXIS, lw=1)
    ax.legend(fontsize=8)
    save(fig, "fig_distance_vs_regret")

    # F4: adverse mid displacement during and after a TWAP sell (median across seeds, one panel per world)
    t = -600 + 30 * np.arange(len(recs[0]["mid30"]))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.0), sharex=True)
    for ax, w in zip(axes.ravel(), worlds):
        for s in sizes:
            sub = [r for r in recs if r["world"] == w and r["size_frac"] == s and r["side"] == "sell" and r["policy"] == "exp_b0" and not r.get("timeout")]
            ax.plot(t / 60, np.median(displacement_paths(sub, base), axis=0), color=size_hue[s], lw=2, label=f"{s:.0%} of volume")
        ax.axvspan(0, 30, color=GRID, alpha=0.45, lw=0)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(w.replace("_", " ", 1))
    for ax in axes[-1]:
        ax.set_xlabel("minutes from start (execution shaded)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median adverse mid shift (bps)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, "fig_impact_path")

    print(json.dumps(findings, indent=2))
    print(cost.round(1).to_string())
    print(mfg.round(2).to_string())
