"""
Goal 2 with LOB-Impact (design: simrel/reports/lob_impact_design.md).

The LOB-Impact protocol is run on each reference market (which can be intervened on) and on each of its candidate
simulators; the errors of each candidate against its market join the Goal 2 table, and the pre-specified analyses
(B1-B4) compare every realism and counterfactual measure as a predictor and selector of good decisions.

    python simrel/experiments/goal2_impact.py calib
    python simrel/experiments/goal2_impact.py run [--worlds W1,W2] [--procs N] [--shard i/n]
    python simrel/experiments/goal2_impact.py analyse
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_run as G  # noqa: E402
from simrel.experiments import impact_01_run as I  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402
from simrel.metrics import lob_impact as LI  # noqa: E402

OUT = G.OUT / "lob_impact"  # result files
BASE_LI = G.BASE / "lob_impact"  # rollouts, calibration and ABIDES rates (shared by every analysis)
N_CAND, N_REF = 32, 64
N_DAYS = 8
SEED0 = 80_000
REF = "reference"
PENALTY = 10.0  # an undefined value where the reference has one counts as ten tolerances
TOL = {"gamma": 0.10, "H": 0.02, "drift": 5.0, "delta": 0.10, "Y": np.log(2.0), "ep": 0.10}


# ------------------------------------------------------------------ calibration
def stage_calib(worlds: list) -> None:
    """Child size, spacing and normalisation per reference market, from its held-out windows (rows of other markets kept)."""
    rows = []
    for w in worlds:
        ws = G.reference_windows(w)
        m = np.vstack([x.message for x in ws])
        mo = np.concatenate([I.market_orders(x.message) for x in ws])
        secs = sum(x.length_s for x in ws)
        phi = len(mo) / len(m)
        vol = float(sum(x.message[x.message[:, 1] == 4, 3].sum() for x in ws))
        r5 = []
        for x in ws:
            _, bid, ask = R.L.grid_quotes(x)
            mid = 0.5 * (bid + ask)
            ok = (ask < R.L.EMPTY) & (bid > -R.L.EMPTY)
            r5.append(np.diff(np.log(mid[ok])))
        sigma_d = float(np.concatenate(r5).std() * np.sqrt(23400 / 5))
        rows.append({"world": w, "phi": phi, "dN": int(round(9 / phi)), "child": int(np.median(mo)), "V": vol / secs * 23400,
                     "range": sigma_d * LI.KAPPA_P, "rate_per_s": len(m) / secs, "sigma_daily": sigma_d})
        print(f"  {w}: phi {phi:.4f}  dN {rows[-1]['dN']}  child {rows[-1]['child']}  rate {rows[-1]['rate_per_s']:.1f}/s  sigma_d {sigma_d:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if (BASE_LI / "calibration.csv").exists():
        old = pd.read_csv(BASE_LI / "calibration.csv")
        new = pd.concat([old[~old["world"].isin(worlds)], new])
    new.to_csv(BASE_LI / "calibration.csv", index=False)


def calibration() -> pd.DataFrame:
    return pd.read_csv(BASE_LI / "calibration.csv").set_index("world")


def daily(world: str) -> dict:
    c = calibration().loc[world]
    return {f"D{k}": {"V": float(c["V"]), "range": float(c["range"])} for k in range(N_DAYS)}


# ------------------------------------------------------------------ rollouts
def factory(world: str, name: str):
    if name == REF:
        from simrel.reference_worlds.data import make

        return make(world)
    return G.make_factory(world, name)


def abides_rate(world: str) -> float:
    path = BASE_LI / f"abides_rate_{world}.txt"
    if not path.exists():
        w = R.abides_window(G.abides_spec(world), G.SEED_SIM_WIN)
        path.write_text(str(len(w.message) / w.length_s))
    return float(path.read_text())


def rollout_job(args) -> str:
    world, name, arm, side, k = args
    path = BASE_LI / "rec" / world / name / f"{arm}_{side}_{k:03d}.pkl"
    if path.exists():
        return "cached"
    R._memory_guard(path, 6.0)
    c = calibration().loc[world]
    dN, child = int(c["dN"]), int(c["child"])
    n_ins, n_cool = I.ARMS[arm]
    is_buy, seed = side == "buy", SEED0 + k
    t0 = time.time()
    try:
        if name == "abides":
            msg, book, aggr = I.abides_rollout(G.abides_spec(world), seed, dN, child, n_ins, n_cool, is_buy, abides_rate(world))
        else:
            msg, book, aggr = I.eventsim_rollout(factory(world, name)(seed), dN, child, n_ins, n_cool, is_buy, True, G.WARMUP_S)
        rec = LI.reduce(I.to_rollout(msg, book, aggr, f"D{k % N_DAYS}", 1 if is_buy else -1), n_ins, n_cool)
        rec.update(world=world, sim=name, arm=arm, side=side, k=k, n_children=len(aggr), wall_s=time.time() - t0)
    except Exception as err:
        rec = {"world": world, "sim": name, "arm": arm, "side": side, "k": k, "failed": repr(err)[:300]}
    I.save(path, rec)
    return f"{world} {name} {arm} {side} {k}: " + (rec["failed"][:80] if "failed" in rec else f"{rec['n_rows']} rows, {rec['wall_s']:.0f}s")


def jobs(worlds: list) -> list:
    out = []
    for w in worlds:
        for name in [REF] + list(G.family()):
            n = N_REF if name == REF else N_CAND
            for arm in ("noins", "beta", "relax"):
                for side in (("buy",) if arm == "noins" else ("buy", "sell")):
                    out += [(w, name, arm, side, k) for k in range(n)]
    return [j for j in out if not (BASE_LI / "rec" / j[0] / j[1] / f"{j[2]}_{j[3]}_{j[4]:03d}.pkl").exists()]


# ------------------------------------------------------------------ scores
def scores(world: str, name: str) -> dict:
    def recs(arm):
        out = [I.load(p) for side in ("buy", "sell") for p in sorted((BASE_LI / "rec" / world / name).glob(f"{arm}_{side}_*.pkl"))]
        return [r for r in out if "failed" not in r]

    row = {f"p1_{k}": v for k, v in LI.phase1(recs("noins")).items()}
    beta = recs("beta")
    row.update({f"p2_{k}": v for k, v in LI.build_up(LI.points(beta), daily(world)).items()})
    K, _ = LI.k_clock(beta, 100, 0)
    row.update({f"p2_{k}": v for k, v in LI.peak(K, 100).items()})
    K, _ = LI.k_clock(recs("relax"), 10, 100)
    row.update({f"p3_{k}": v for k, v in LI.relaxation(K, 10).items()})
    return row


def _term(x, target, tol) -> float:
    if not np.isfinite(target):
        return np.nan
    return abs(x - target) / tol if np.isfinite(x) else PENALTY


def errors(s: dict, ref: dict) -> dict:
    """The pre-specified errors of one candidate against its reference market."""
    def law(d, y, ep):
        return float(np.mean([v for v in [_term(s["p2_delta"] if s.get("p2_identified") else np.nan, d, TOL["delta"]),
                                 _term(np.log(s["p2_Y"]) if s.get("p2_identified") and s["p2_Y"] > 0 else np.nan, y, TOL["Y"]),
                                 _term(s["p3_ep"] if s.get("p3_sig") else np.nan, ep, TOL["ep"])] if np.isfinite(v)] or [np.nan]))
    ref_delta = ref["p2_delta"] if ref.get("p2_identified") else np.nan
    ref_lnY = np.log(ref["p2_Y"]) if ref.get("p2_identified") and ref["p2_Y"] > 0 else np.nan
    ref_ep = ref["p3_ep"] if ref.get("p3_sig") else np.nan
    # literature: Y's target is the band [0.5, 1]; inside the band the Y term is zero
    lnY = np.log(s["p2_Y"]) if s.get("p2_identified") and s["p2_Y"] > 0 else np.nan
    y_band = (0.0 if np.log(0.5) <= lnY <= 0.0 else min(abs(lnY - np.log(0.5)), abs(lnY)) / TOL["Y"]) if np.isfinite(lnY) else PENALTY
    lit = float(np.mean([_term(s["p2_delta"] if s.get("p2_identified") else np.nan, 0.5, TOL["delta"]), y_band,
                         _term(s["p3_ep"] if s.get("p3_sig") else np.nan, 2 / 3, TOL["ep"])]))
    return {
        "li_flow_error": float(np.nanmean([_term(s.get("p1_gamma_flow", np.nan), ref.get("p1_gamma_flow", np.nan), TOL["gamma"]),
                                           _term(s.get("p1_H", np.nan), ref.get("p1_H", np.nan), TOL["H"]),
                                           _term(s.get("p1_drift_ticks", np.nan), ref.get("p1_drift_ticks", np.nan), TOL["drift"])])),
        "li_response_error": _term(s.get("p1_R100_over_R1", np.nan), ref.get("p1_R100_over_R1", np.nan), 1.0),
        "li_peak_error": _term(s.get("p2_peak_bps", np.nan), ref.get("p2_peak_bps", np.nan), 1.0),
        "li_law_error": law(ref_delta, ref_lnY, ref_ep),
        "li_law_error_literature": lit,
    }


# ------------------------------------------------------------------ analysis
OLD = ["order_book", "price", "impact", "trade_response_error", "metaorder_impact_error"]
NEW = ["li_flow_error", "li_response_error", "li_peak_error", "li_law_error", "li_law_error_literature"]
FLAVOUR = {"order_book": "realism (LOB-Bench)", "price": "realism (LOB-Bench)", "impact": "realism (LOB-Bench impact score)",
           "trade_response_error": "realism", "li_flow_error": "realism (LOB-Impact phase 1)", "li_response_error": "realism (LOB-Impact phase 1)",
           "li_peak_error": "counterfactual (LOB-Impact)", "li_law_error": "counterfactual (LOB-Impact)",
           "li_law_error_literature": "counterfactual (LOB-Impact, literature targets)", "metaorder_impact_error": "counterfactual (task-specific)"}


def spearman_ci(t: pd.DataFrame, m: str, rng) -> tuple[float, float, float]:
    from scipy.stats import spearmanr

    def rho(d):
        d = d.dropna(subset=[m])
        r = d.copy()
        r[m] = r.groupby(["world", "size_frac"])[m].rank()
        r["regret"] = r.groupby(["world", "size_frac"])["regret"].rank()
        return spearmanr(r[m], r["regret"]).correlation

    est = rho(t)
    groups = [g for _, g in t.groupby(["world", "size_frac"])]
    bs = [rho(pd.concat([g.sample(len(g), replace=True, random_state=rng) for g in groups])) for _ in range(2000)]
    return float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def analyse() -> None:
    sim = pd.read_csv(G.OUT / "goal2_simulators.csv")
    rows, ref_rows = [], []
    for w in G.worlds_with_results():
        if not (BASE_LI / "rec" / w).exists():
            continue
        ref = scores(w, REF)
        ref_rows.append({"world": w, **ref})
        for name in G.family():
            if not (BASE_LI / "rec" / w / name).exists():
                continue
            s = scores(w, name)
            rows.append({"world": w, "sim": name, **s, **errors(s, ref)})
    li = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ref_rows).to_csv(OUT / "reference_markets_lob_impact.csv", index=False)
    li.to_csv(OUT / "candidates_lob_impact.csv", index=False)
    t = sim.merge(li[["world", "sim"] + NEW], on=["world", "sim"], how="inner")
    t.to_csv(OUT / "goal2_with_lob_impact.csv", index=False)
    pd.set_option("display.width", 250)
    print("reference markets:")
    print(pd.DataFrame(ref_rows)[["world", "p1_gamma_flow", "p1_H", "p1_R100_over_R1", "p2_identified", "p2_delta", "p2_Y", "p2_peak_bps", "p3_ep"]].round(3).to_string(index=False))
    rng = np.random.RandomState(20261101)
    out = []
    for label, d in [("all", t), ("excluding ABIDES", t[t["sim"] != "abides"]), ("agent-mix markets only", t[t["world"].str.startswith("R2")]),
                     ("queue-reactive markets only", t[t["world"].str.startswith("R1")]),
                     ("original four markets", t[t["world"].isin(G.WORLDS)]), ("six added markets", t[t["world"].isin(G.NEW_WORLDS)]),
                     ("ten sampled markets", t[t["world"].isin(G.SAMPLED_WORLDS)])]:
        for m in OLD + NEW:
            if d[m].notna().sum() < 10:
                continue
            est, lo, hi = spearman_ci(d, m, rng)
            out.append({"subset": label, "measure": m, "flavour": FLAVOUR[m], "spearman": est, "lo": lo, "hi": hi})
    b1 = pd.DataFrame(out)
    b1.to_csv(OUT / "B1_correlations.csv", index=False)
    print("\nB1  rank correlation of regret with each measure (positive: a better score goes with lower regret)")
    print(b1.round(2).to_string(index=False))
    b2 = []
    for (w, s), g in t.groupby(["world", "size_frac"]):
        tr = G.truth(w)
        tc = tr[tr["size_frac"] == s].set_index("policy")["true_cost"]
        row = {"world": w, "size_frac": s, "always_twap": tc["P0_twap"] - tc.min(), "median_regret_all": g["regret"].median()}
        pred = [c for c in g.columns if c.startswith("pred_")]
        for m in OLD + NEW:
            ok = g.dropna(subset=[m])
            row[f"best_{m}"] = ok.loc[ok[m].idxmin(), "regret"] if len(ok) else np.nan
            row[f"top5_{m}"] = (tc[ok.nsmallest(5, m)[pred].mean().idxmin().removeprefix("pred_")] - tc.min()) if len(ok) >= 5 else np.nan
        b2.append(row)
    b2 = pd.DataFrame(b2)
    b2.to_csv(OUT / "B2_B3_selection.csv", index=False)
    print("\nB2/B3  mean regret (bps) over markets and sizes of the simulator, or five-simulator ensemble, chosen by each measure")
    print(b2.drop(columns=["world", "size_frac"]).mean().round(2).to_string())
    # a measure that is undefined in some markets must be compared with the others on the same markets
    fam = b2.assign(family=np.where(b2["world"].str.startswith("R1"), "queue-reactive", "agent-mix"))
    by = fam.drop(columns=["world", "size_frac"]).groupby("family").mean().T
    by.to_csv(OUT / "B2_B3_by_family.csv")
    print("\nB2/B3 by market family (same markets for every measure; NaN = undefined there)")
    print(by.round(2).to_string())
    detect = li.groupby("world")[["p2_detected", "p2_identified", "p2_n_rollouts"]].mean()
    detect.to_csv(OUT / "candidates_detection.csv")
    print("\nshare of candidates whose impact is detected (3 s.e.) and whose law is identified, and rollouts kept per candidate")
    print(detect.round(2).to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 4
    worlds = args[args.index("--worlds") + 1].split(",") if "--worlds" in args else G.WORLDS
    shard = tuple(int(x) for x in args[args.index("--shard") + 1].split("/")) if "--shard" in args else (0, 1)
    if args[0] == "calib":
        stage_calib(worlds)
    elif args[0] == "run":
        todo = jobs(worlds)
        if "--n" in args:
            todo = [j for j in todo if j[4] < int(args[args.index("--n") + 1])]
        if "--names" in args:
            todo = [j for j in todo if j[1] in args[args.index("--names") + 1].split(",")]
        R.pool_map(rollout_job, todo[shard[0]::shard[1]], procs, "goal2 lob-impact")
    elif args[0] == "analyse":
        analyse()
