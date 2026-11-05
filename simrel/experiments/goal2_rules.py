"""
Conservative decision rules and a decision-relevant validation profile (simrel/reports/lob_impact_design_addendum5.md).

Experiment A: deviate from TWAP only when confident. For strategy a, the ensemble's predicted gain over TWAP is
D_a = mean over simulators of pred_s(TWAP) - pred_s(a); its uncertainty combines between-simulator disagreement, the
seed error of the mean, and the strategy's historical error (root mean square error of the ensemble's gain over the
training markets), which captures error that all simulators share:

    sigma_a^2 = var_s(D_s,a) + mean_s(se_s,a^2) / n_s + h_a^2,     choose argmax_a (D_a - k sigma_a) if positive, else TWAP.

A per-simulator version uses sigma_s,a^2 = se_s,a^2 + h_ind_a^2 (h_ind: historical error of single simulators).
Variant "bc" also subtracts each strategy's historical bias b_a from D_a. Everything (h, b, k) is estimated on the
training markets only and frozen in rules_frozen.json before the test markets are read.

Experiment B: validation profile, each an error against the reference market (rank within market and size):
immediate-execution cost, TWAP impact, relaxation after TWAP (impact recovered in the 120 s after the deadline),
conditional price response (lag-one autocorrelation of 25-trade returns in free runs), and, in synthetic markets only,
cost-ranking accuracy (rank correlation of predicted and true costs over the 12 strategies).

    python simrel/experiments/goal2_rules.py train      # training markets -> rules_frozen.json
    python simrel/experiments/goal2_rules.py test       # the sampled markets, with the frozen values
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_cluster_bootstrap as B  # noqa: E402
from simrel.experiments import goal2_run as G  # noqa: E402

TRAIN = G.WORLDS + G.NEW_WORLDS
TEST = G.SAMPLED_WORLDS
K_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
K_FIXED = 1.0  # pre-specified sensitivity value
OUT = G.OUT / "rules"
TWAP = "P0_twap"
BLOCK = 25  # trades per return block for the conditional response


def cell_tables(worlds: list, names: list, tabs: dict) -> dict:
    """(world, size) -> {sim: (policies, mean cost per policy, seed s.e. of the gain over TWAP per policy, n seeds)}."""
    out = {}
    for w in worlds:
        for size in (0.05, 0.15):
            d = {}
            for n in names:
                if (w, n, size) not in tabs:
                    continue
                pols, cost, _ = tabs[(w, n, size)]
                j = pols.index(TWAP)
                gain = cost[:, [j]] - cost
                se = np.nanstd(gain, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(gain), axis=0))
                d[n] = (pols, np.round(np.nanmean(cost, axis=0), 9), se, len(cost))
            if d:
                out[(w, size)] = d
    return out


def truth_costs(worlds: list) -> dict:
    out = {}
    for w in worlds:
        t = G.truth(w)
        for size, g in t.groupby("size_frac"):
            out[(w, round(size, 2))] = g.set_index("policy")["true_cost"]
    return out


def ensemble_gain(cell: dict) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Policies, gains (simulators x policies), between-simulator variance and seed variance of the mean."""
    pols = next(iter(cell.values()))[0]
    gains = np.vstack([c[1][pols.index(TWAP)] - c[1] for c in cell.values()])
    se2 = np.vstack([c[2] ** 2 for c in cell.values()])
    return pols, gains, np.nanvar(gains, axis=0, ddof=1), np.nanmean(se2, axis=0) / len(cell)


def history(cells: dict, truth: dict) -> dict:
    """Per strategy: bias and root mean square error of the ensemble gain and of single-simulator gains."""
    ens, ind = {}, {}
    for key, cell in cells.items():
        pols, gains, _, _ = ensemble_gain(cell)
        tc = truth[key]
        tg = np.array([tc[TWAP] - tc[a] for a in pols])
        for i, a in enumerate(pols):
            ens.setdefault(a, []).append(np.nanmean(gains[:, i]) - tg[i])
            ind.setdefault(a, []).extend(list(gains[:, i] - tg[i]))
    f = lambda e: (float(np.nanmean(e)), float(np.sqrt(np.nanmean(np.square(e)))))
    return {a: {"bias": f(ens[a])[0], "h": f(ens[a])[1], "bias_ind": f(ind[a])[0], "h_ind": f(ind[a])[1]} for a in ens}


def choose(pols, gain, sigma, k) -> str:
    score = np.where(np.array(pols) == TWAP, -np.inf, gain - k * sigma)
    i = int(np.nanargmax(score)) if np.isfinite(score).any() else -1
    return pols[i] if i >= 0 and score[i] > 0 else TWAP


def regret_of(rule: str, k: float, cells: dict, truth: dict, hist: dict) -> dict:
    """Mean regret per cell of one rule: 'ensemble' (gated), 'ensemble_bc' (gated, bias-corrected), 'per_sim' (gated,
    averaged over simulators), 'ensemble_raw' and 'per_sim_raw' (ungated: the cheapest predicted strategy)."""
    out = {}
    for key, cell in cells.items():
        tc = truth[key]
        reg = lambda a: float(tc[a] - tc.min())
        pols, gains, var_b, var_seed = ensemble_gain(cell)
        h = np.array([hist[a]["h"] for a in pols])
        if rule in ("ensemble", "ensemble_bc", "ensemble_raw"):
            g = np.nanmean(gains, axis=0) - (np.array([hist[a]["bias"] for a in pols]) if rule == "ensemble_bc" else 0.0)
            sig = np.sqrt(var_b + var_seed + h ** 2)
            out[key] = reg(choose(pols, g, sig, 0.0 if rule == "ensemble_raw" else k) if rule != "ensemble_raw" else pols[int(np.nanargmax(np.where(np.array(pols) == TWAP, 0.0, g)))] if np.nanmax(g) > 0 else TWAP)
        else:
            h_ind = np.array([hist[a]["h_ind"] for a in pols])
            rs = []
            for c in cell.values():
                g = c[1][pols.index(TWAP)] - c[1]
                if rule == "per_sim_raw":
                    rs.append(reg(pols[int(np.nanargmin(c[1]))]))
                else:
                    rs.append(reg(choose(pols, g, np.sqrt(c[2] ** 2 + h_ind ** 2), k)))
            out[key] = float(np.mean(rs))
    return out


def train() -> None:
    names, tabs = B.seed_tables()
    cells = cell_tables(TRAIN, names, tabs)
    truth = truth_costs(TRAIN)
    hist = history(cells, truth)
    table, frozen = [], {"train_markets": TRAIN, "history": hist, "k": {}}
    for rule in ("ensemble", "ensemble_bc", "per_sim"):
        res = {k: float(np.mean(list(regret_of(rule, k, cells, truth, hist).values()))) for k in K_GRID}
        best = min(res.values())
        k_star = max(k for k, v in res.items() if v <= best + 1e-12)  # ties go to the more conservative value
        frozen["k"][rule] = k_star
        table += [{"rule": rule, "k": k, "train_mean_regret": v} for k, v in res.items()]
    for rule in ("ensemble_raw", "per_sim_raw"):
        table.append({"rule": rule, "k": None, "train_mean_regret": float(np.mean(list(regret_of(rule, 0.0, cells, truth, hist).values())))})
    tw = np.mean([float(truth[k][TWAP] - truth[k].min()) for k in cells])
    table.append({"rule": "always_twap", "k": None, "train_mean_regret": float(tw)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(OUT / "train_regret.csv", index=False)
    (OUT / "rules_frozen.json").write_text(json.dumps(frozen, indent=1))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(table).pivot_table(index="rule", columns="k", values="train_mean_regret", dropna=False).round(2).to_string())
    print("frozen k:", frozen["k"], f"; always TWAP {tw:.2f}")
    print(pd.DataFrame(hist).T.round(2).to_string())


# ------------------------------------------------------------------ experiment B: validation profile
def conditional_response(world: str, name: str) -> float:
    """Lag-one autocorrelation of 25-trade mid returns, pooled over a simulator's free runs (negative: reversal)."""
    from simrel.experiments import goal2_impact as GI

    x, y = [], []
    for side in ("buy", "sell"):
        for p in sorted((GI.BASE_LI / "rec" / world / name).glob(f"noins_{side}_*.pkl")):
            r = G.load(p)
            m = r.get("tmid", np.zeros(0))
            m = m[np.isfinite(m)][:: BLOCK]
            d = np.diff(np.log(m)) if len(m) > 3 else np.zeros(0)
            x += list(d[:-1]); y += list(d[1:])
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 10 and x.std() > 0 and y.std() > 0 else np.nan


def relaxation(rows: pd.DataFrame) -> float:
    """TWAP impact recovered in the 120 s after the deadline (bps): end displacement minus post displacement."""
    tw = rows[rows["policy"] == TWAP]
    return float((tw["disp_end_bps"] - tw["disp_post_bps"]).mean())


def profile_table(worlds: list) -> pd.DataFrame:
    """Per (market, size, simulator): the validation-profile errors against the reference market."""
    from scipy.stats import spearmanr

    t = pd.read_csv(G.OUT / "goal2_simulators.csv")
    t = t[t["world"].isin(worlds)].copy()
    rows = []
    for w in worlds:
        ref = pd.DataFrame(G.load(G.REF / f"ref_{w}" / "rows_reference.pkl"))
        ref = ref[ref["failed"].isna() | (ref["failed"] == False)]  # noqa: E712
        cr_ref = conditional_response(w, "reference")
        tr = G.truth(w)
        for size in (0.05, 0.15):
            tc = tr[tr["size_frac"].round(2) == size].set_index("policy")["true_cost"]
            relax_ref = relaxation(ref[ref["size_frac"].round(2) == size])
            for _, r in t[(t["world"] == w) & (t["size_frac"].round(2) == size)].iterrows():
                pol = []
                for s in range(G.POLICIES_SEEDS):
                    f = G.RAW / "policy" / f"{w}_{r['sim']}_{G.SEED_POL + s}.pkl"
                    if f.exists():
                        pol += [x for x in G.load(f) if not x.get("failed") and round(x["size_frac"], 2) == size]
                pol = pd.DataFrame(pol)
                pred = np.array([r.get(f"pred_{a}", np.nan) for a in tc.index])
                ok = np.isfinite(pred)
                rows.append({"world": w, "size_frac": size, "sim": r["sim"], "regret": r["regret"],
                             "immediate_error": abs(r["pred_P1_immediate"] - tc["P1_immediate"]),
                             "twap_impact_error": r["metaorder_impact_error"],
                             "relaxation_error": abs(relaxation(pol) - relax_ref) if len(pol) else np.nan,
                             "conditional_response_error": abs(conditional_response(w, r["sim"]) - cr_ref),
                             "ranking_accuracy_oracle": -spearmanr(pred[ok], tc.values[ok]).correlation})  # negated: lower is better
    d = pd.DataFrame(rows)
    dims = ["immediate_error", "twap_impact_error", "relaxation_error", "conditional_response_error"]
    ranks = d.groupby(["world", "size_frac"])[dims].rank()
    d["profile"] = ranks.mean(axis=1)
    return d


PROFILE_MEASURES = ["twap_impact_error", "immediate_error", "relaxation_error", "conditional_response_error", "profile", "ranking_accuracy_oracle",
                    "order_book"]


def test(n_boot: int = 2000) -> None:
    """The pre-registered tests of addendum 5 on the sampled markets, with the frozen training values."""
    frozen = json.loads((OUT / "rules_frozen.json").read_text())
    hist, ks = frozen["history"], frozen["k"]
    names, tabs = B.seed_tables()
    tabs = {k: v for k, v in tabs.items() if k[0] in TEST}
    truth = truth_costs(TEST)
    prof = profile_table(TEST)
    real = pd.read_csv(G.OUT / "goal2_simulators.csv")[["world", "size_frac", "sim", "order_book"]]
    prof = prof.merge(real, on=["world", "size_frac", "sim"], how="left")
    prof.to_csv(OUT / "test_profile.csv", index=False)
    fixed = prof.set_index(["world", "size_frac", "sim"])
    rng = np.random.default_rng(20261104)

    def once(boot: bool) -> dict:
        worlds = sorted({k[0] for k in tabs})
        worlds = list(rng.choice(worlds, len(worlds))) if boot else worlds
        designs = list(rng.choice(names, len(names))) if boot else names
        cells, tr, rows = {}, {}, []
        for wi, w in enumerate(worlds):
            for size in (0.05, 0.15):
                cell = {}
                for j, n in enumerate(designs):
                    if (w, n, size) not in tabs:
                        continue
                    pols, cost, _ = tabs[(w, n, size)]
                    idx = rng.integers(0, len(cost), len(cost)) if boot else np.arange(len(cost))
                    c = cost[idx]
                    gain = c[:, [pols.index(TWAP)]] - c
                    se = np.nanstd(gain, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(gain), axis=0))
                    mean = np.round(np.nanmean(c, axis=0), 9)
                    cell[f"{n}#{j}"] = (pols, mean, se, len(c))
                    tc = truth[(w, size)]
                    if (w, size, n) in fixed.index:
                        rows.append({"cell": f"{wi}_{size}", "regret": float(tc[pols[int(np.nanargmin(mean))]] - tc.min()), **fixed.loc[(w, size, n)].drop("regret").to_dict()})
                if cell:
                    cells[(f"{wi}", size)] = cell
                    tr[(f"{wi}", size)] = truth[(w, size)]
        res = {"always_twap": float(np.mean([v[TWAP] - v.min() for v in tr.values()]))}
        for rule in ("ensemble_raw", "per_sim_raw"):
            res[rule] = float(np.mean(list(regret_of(rule, 0.0, cells, tr, hist).values())))
        for rule in ("ensemble", "ensemble_bc", "per_sim"):
            res[f"{rule}_k{ks[rule]:g}"] = float(np.mean(list(regret_of(rule, ks[rule], cells, tr, hist).values())))
            res[f"{rule}_k{K_FIXED:g}"] = float(np.mean(list(regret_of(rule, K_FIXED, cells, tr, hist).values())))
        k = ks["ensemble"]
        res["A1_gated_minus_raw"] = res[f"ensemble_k{k:g}"] - res["ensemble_raw"]
        res["A2_gated_minus_twap"] = res[f"ensemble_k{k:g}"] - res["always_twap"]
        res["A3_persim_gated_minus_raw"] = res[f"per_sim_k{ks['per_sim']:g}"] - res["per_sim_raw"]
        df = pd.DataFrame(rows)
        for m in PROFILE_MEASURES:
            res[f"rho_{m}"] = B.spearman_within(df, m)
        res["B1_profile_minus_twap_impact"] = res["rho_profile"] - res["rho_twap_impact_error"]
        return res

    est = once(False)
    bs = pd.DataFrame([once(True) for _ in range(n_boot)])
    out = pd.DataFrame({"estimate": pd.Series(est), "lo": bs.quantile(0.025), "hi": bs.quantile(0.975)})
    out.to_csv(OUT / "test_results.csv")
    pd.set_option("display.width", 200)
    print(f"addendum 5 tests on the sampled markets ({len(TEST)} markets), three-level bootstrap, {n_boot} resamples")
    print(out.round(3).to_string())


if __name__ == "__main__":
    {"train": train, "test": test}[sys.argv[1]]()
