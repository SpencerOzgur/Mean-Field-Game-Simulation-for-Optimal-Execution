"""
Calibration of the lambda pilot, on calibration seeds only. Everything here was run before the stopping rule
(simrel/reports/lambda_pilot_rule.md) was committed, and before any cost, regret or decision result existed.

First pass (stage "grid"), on the first six agent-mix draws and a grid of lambda values:
    vol_stage       the fund_vol at which the variance of 600 s mid returns in passive episodes equals that of
                    lambda = 0 (48 calibration seeds, 5% tolerance), and the same ratio on 48 check seeds
    persist_stage   TWAP impact persistence of the matched markets: mean displacement 120 s after the deadline over
                    mean displacement at the deadline (the ratio form of addendum 5, item 3), twin and TWAP only
The first pass fixed the levels (lambda_pilot.LEVELS) and showed that two of the six draws cannot be matched.

Final pass (stage "final"), which defines the pilot markets:
    vol_final       draws are taken in draw order, up to XLP_pilot_a20. Each is matched at the three levels with 192
                    calibration seeds and a 2% tolerance, then checked on the 48 check seeds. A draw is kept if it is
                    viable and, at every level, the bootstrap 95% interval of the check-seed variance ratio lies
                    entirely within [0.80, 1.25]. The search stops at six kept markets.
    finalise        writes the market list (pilot_markets.json), the arms (arms.json) and the table (calib_final.csv)
    persist_final   persistence of the final markets at the three levels, per market and pooled (description only;
                    the levels are not re-picked)

    python simrel/experiments/lambda_pilot_calib.py grid|final [procs]
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import reference as C  # noqa: E402
from simrel.experiments import lambda_pilot as P  # noqa: E402
from simrel.reference_worlds import lambda_world as LW  # noqa: E402

DESC = ["volume_10min", "n_trades", "vol5_ticks", "ret5_ac1", "ret60_ac1", "abs5_ac1", "sign_ac1", "end_spread", "end_depth"]
GRID_MARKETS = [f"{LW.PILOT_PREFIX}a{k:02d}" for k in range(1, 7)]  # first pass: the first six draws
CAP = 20  # the final search goes at most through XLP_pilot_a20
EQ_LO, EQ_HI = 0.80, 1.25  # equivalence bounds for the check-seed variance ratio
TRADER_LEVEL = "band_low"  # the trader-only comparison arm uses this level's lambda


def am_draws(last: int = CAP) -> list:
    """Agent-mix draws in draw order, up to number last."""
    return [f"{LW.PILOT_PREFIX}a{k:02d}" for k in range(1, last + 1)]


def volumes() -> dict:
    """Reference volume (lambda = 0, 40 viability twins) of every draw whose viability has been run."""
    return {k: v["volume"] for k, v in P.load_json("pilot_markets.json")["draw_block_viability"].items()}


def episodes(pool, market: str, kw: dict, seed0: int, n: int) -> list:
    return pool.map(P.vol_job, [(market, "", kw, seed0 + s) for s in range(n)], chunksize=4)


def _mids(rows: list) -> list:
    return [r["mid"].astype(float) for r in rows]


def _var(rows: list) -> float:
    return LW.return_var(_mids(rows))


def describe(rows: list) -> dict:
    return {k: float(np.nanmean([r[k] for r in rows])) for k in DESC}


def _boot_ratio(on: list, off: list, n_boot: int, seed: int) -> tuple[float, np.ndarray]:
    """Variance ratio on / off over non-wild windows, and its bootstrap draws (episodes resampled, same seeds in both arms)."""
    def stats(rows):
        ss, n = [], []
        for r in rows:
            d = LW.window_moves([r["mid"]])
            d = d[np.abs(d) <= LW.WILD_TICKS]
            ss.append((d**2).sum())
            n.append(len(d))
        return np.array(ss), np.array(n, float)

    (a, na), (b, nb) = stats(on), stats(off)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    boot = (a[idx].sum(1) / na[idx].sum(1)) / (b[idx].sum(1) / nb[idx].sum(1))
    return float((a.sum() / na.sum()) / (b.sum() / nb.sum())), boot


def ratio_se(on: list, off: list, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    r, boot = _boot_ratio(on, off, n_boot, seed)
    return r, float(boot.std())


def ratio_ci(on: list, off: list, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% interval of the variance ratio on / off."""
    _, boot = _boot_ratio(on, off, n_boot, seed)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---- persistence
def persistence(d: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Ratio of mean displacements (120 s after the deadline over at the deadline), pooled; bootstrap over runs."""
    e, p = d["disp_end_bps"].to_numpy(float), d["disp_post_bps"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    r = p[idx].mean(1) / e[idx].mean(1)
    return {"n": len(e), "disp_end_bps": e.mean(), "disp_end_se": e.std(ddof=1) / np.sqrt(len(e)), "disp_post_bps": p.mean(),
            "disp_post_se": p.std(ddof=1) / np.sqrt(len(p)), "recovered_bps": (e - p).mean(), "recovered_se": (e - p).std(ddof=1) / np.sqrt(len(e)),
            "persistence": p.mean() / e.mean(), "persistence_lo": float(np.percentile(r, 2.5)), "persistence_hi": float(np.percentile(r, 97.5))}


def persistence_table(rows: list, pooled_label: str = "pooled") -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d = d[d["disp_end_bps"].notna()]
    out = []
    for f in C.SIZES:
        for tag, g in d[d["size_frac"] == f].groupby("tag"):
            out.append({"lam_bp": float(tag), "size_frac": f, "market": pooled_label, **persistence(g)})
            for m, gm in g.groupby("market"):
                out.append({"lam_bp": float(tag), "size_frac": f, "market": m, **persistence(gm)})
    return pd.DataFrame(out).sort_values(["size_frac", "market", "lam_bp"])


# ---- first pass: grid of lambda values on the first six draws
def vol_stage(procs: int) -> dict:
    path = P.OUT / "calib_vol.json"
    if path.exists():
        return json.loads(path.read_text())
    V = volumes()
    out, table = {}, []
    with Pool(procs) as pool:
        for m in GRID_MARKETS:
            vals = P.ALL[m][1]
            cal0 = episodes(pool, m, {}, C.SEED_CALIB, P.N_VOL_CAL)
            chk0 = episodes(pool, m, {}, P.SEED_VOLCHECK, P.N_VOL_CHECK)
            target = _var(cal0)
            out[m] = {"target_var600_ticks2": target, "base_fund_vol": vals["fund_vol"], "levels": {}}
            table.append({"market": m, "lam_bp": 0.0, "fund_vol": vals["fund_vol"], "ratio_cal": 1.0, "ratio_check": 1.0, "ratio_check_se": 0.0,
                          "wild_share": LW.wild_share(_mids(cal0 + chk0)), "plain_var_ratio": 1.0, "feasible": True, "matched": True, **describe(cal0 + chk0)})
            for lam in P.GRID_BP:
                last = {}

                def var_of(fv, lam=lam, last=last):
                    rows = episodes(pool, m, {"lam_bp": lam, "reference_volume": V[m], "fund_vol": fv}, C.SEED_CALIB, P.N_VOL_CAL)
                    last[fv] = rows
                    return _var(rows)

                res = LW.match_fund_vol(var_of, vals["fund_vol"], target)
                chk = episodes(pool, m, {"lam_bp": lam, "reference_volume": V[m], "fund_vol": res["fund_vol"]}, P.SEED_VOLCHECK, P.N_VOL_CHECK)
                r_chk, r_se = ratio_se(chk, chk0)
                raw = episodes(pool, m, {"lam_bp": lam, "reference_volume": V[m]}, P.SEED_VOLCHECK, P.N_VOL_CHECK)
                both = last[res["fund_vol"]] + chk
                res.update({"ratio_check": r_chk, "ratio_check_se": r_se, "ratio_check_unmatched": _var(raw) / _var(chk0),
                            "wild_share": LW.wild_share(_mids(both)), "wild_share_unmatched": LW.wild_share(_mids(raw)),
                            "plain_var_ratio": LW.return_var(_mids(both), None) / LW.return_var(_mids(cal0 + chk0), None)})
                out[m]["levels"][str(lam)] = res
                table.append({"market": m, "lam_bp": lam, "fund_vol": res["fund_vol"], "ratio_cal": res["ratio"], "ratio_check": r_chk, "ratio_check_se": r_se,
                              "ratio_check_unmatched": res["ratio_check_unmatched"], "wild_share": res["wild_share"],
                              "wild_share_unmatched": res["wild_share_unmatched"], "plain_var_ratio": res["plain_var_ratio"], "feasible": res["feasible"],
                              "matched": res["matched"], "n_evals": len(res["evals"]), **describe(both)})
                print(f"  {m} lam={lam}: fund_vol {vals['fund_vol']:.3f} -> {res['fund_vol']:.3f}  ratio cal {res['ratio']:.3f} check {r_chk:.3f} "
                      f"(se {r_se:.3f}) feasible={res['feasible']} matched={res['matched']}", flush=True)
    pd.DataFrame(table).to_csv(P.OUT / "calib_vol.csv", index=False)
    path.write_text(json.dumps(out, indent=2))
    return out


def feasible_grid(vol: dict) -> list:
    """Grid values at which every first-pass market is feasible (the variance at fund_vol = 0 is not above target)."""
    return [lam for lam in P.GRID_BP if all(v["levels"][str(lam)]["feasible"] for v in vol.values())]


def persist_stage(procs: int, vol: dict) -> pd.DataFrame:
    from simrel.experiments.run_reference import pool_map

    V = volumes()

    def kw(m, lam):
        return {} if lam == 0.0 else {"lam_bp": lam, "reference_volume": V[m], "fund_vol": vol[m]["levels"][str(lam)]["fund_vol"]}

    jobs = [(m, str(lam), kw(m, lam), V[m], P.SEED_PERSIST + s) for m in GRID_MARKETS for lam in [0.0] + feasible_grid(vol) for s in range(P.N_PERSIST)]
    out = persistence_table(P.cached("calib_persist", lambda: pool_map(P.persist_job, jobs, procs, "persist")))
    out.to_csv(P.OUT / "calib_persist.csv", index=False)
    return out


# ---- final pass: matching at the three levels and market selection
def equivalent(level: dict) -> bool:
    """Matched: feasible, and the 95% interval of the check-seed variance ratio lies entirely within [0.80, 1.25]."""
    return bool(level["feasible"] and level["check_lo"] >= EQ_LO and level["check_hi"] <= EQ_HI)


def kept_markets(final: dict, n: int = P.N_AM) -> list:
    """In draw order, the first n draws that are viable and matched at every level."""
    via = P.load_json("pilot_markets.json")["draw_block_viability"]
    return [m for m in am_draws() if m in final and via[m]["viable"] and all(equivalent(r) for r in final[m]["levels"].values())][:n]


def vol_final(procs: int) -> dict:
    path = P.OUT / "calib_final.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    V = volumes()
    with Pool(procs) as pool:
        for m in am_draws():
            if len(kept_markets(out)) >= P.N_AM:
                break
            if m in out:
                continue
            vals = P.ALL[m][1]
            cal0 = episodes(pool, m, {}, C.SEED_CALIB, P.N_VOL_CAL_FINAL)
            chk0 = episodes(pool, m, {}, P.SEED_VOLCHECK, P.N_VOL_CHECK)
            target = _var(cal0)
            rec = {"reference_volume": V[m], "base_fund_vol": vals["fund_vol"], "target_var600_ticks2": target,
                   "lam0": {"wild_share": LW.wild_share(_mids(cal0 + chk0)), **describe(cal0 + chk0)}, "levels": {}}
            for name, lam in P.LEVELS.items():
                last = {}

                def var_of(fv, lam=lam, last=last):
                    rows = episodes(pool, m, {"lam_bp": lam, "reference_volume": V[m], "fund_vol": fv}, C.SEED_CALIB, P.N_VOL_CAL_FINAL)
                    last[fv] = rows
                    return _var(rows)

                res = LW.match_fund_vol(var_of, vals["fund_vol"], target, tol=P.VOL_TOL_FINAL, max_iter=10)
                cal = last[res["fund_vol"]]
                chk = episodes(pool, m, {"lam_bp": lam, "reference_volume": V[m], "fund_vol": res["fund_vol"]}, P.SEED_VOLCHECK, P.N_VOL_CHECK)
                r_chk, _ = ratio_se(chk, chk0)
                lo, hi = ratio_ci(chk, chk0)
                res.update({"lam_bp": lam, "ratio_check": r_chk, "check_lo": lo, "check_hi": hi, "interval_includes_1": bool(lo <= 1.0 <= hi),
                            "wild_share": LW.wild_share(_mids(cal + chk)), "desc": describe(cal + chk)})
                res["converged"] = res.pop("matched")
                rec["levels"][name] = res
                print(f"  {m} {name} lam={lam}: fund_vol {vals['fund_vol']:.3f} -> {res['fund_vol']:.3f}  cal {res['ratio']:.3f} ({len(res['evals'])} evals) "
                      f"check {r_chk:.3f} [{lo:.3f}, {hi:.3f}] wild {res['wild_share']:.4f} feasible={res['feasible']} converged={res['converged']}", flush=True)
            rec["includes_1_all_levels"] = bool(all(r["feasible"] and r["interval_includes_1"] for r in rec["levels"].values()))  # the replaced criterion
            rec["matched_all_levels"] = bool(all(equivalent(r) for r in rec["levels"].values()))
            print(f"  {m} matched at all levels={rec['matched_all_levels']}", flush=True)
            out[m] = rec
            path.write_text(json.dumps(out, indent=2))
    return out


def reason(rec: dict, viable: bool) -> str:
    if not viable:
        return "not viable at lambda = 0"
    bad = []
    for name, r in rec["levels"].items():
        if not r["feasible"]:
            bad.append(f"{r['lam_bp']}: infeasible (ratio {r['ratio_check']:.2f} with fund_vol = 0)")
        elif not equivalent(r):
            bad.append(f"{r['lam_bp']}: interval [{r['check_lo']:.2f}, {r['check_hi']:.2f}] not within [{EQ_LO}, {EQ_HI}]")
    return "; ".join(bad)


def final_table(final: dict) -> pd.DataFrame:
    via = P.load_json("pilot_markets.json")["draw_block_viability"]
    kept = kept_markets(final)
    rows = []
    for m in am_draws():
        if m not in final:
            continue
        rec = final[m]
        rows.append({"market": m, "level": "lam0", "lam_bp": 0.0, "fund_vol": rec["base_fund_vol"], "kept": m in kept, "viable": via[m]["viable"],
                     "wild_share": rec["lam0"]["wild_share"], **{k: rec["lam0"][k] for k in DESC}})
        for name, r in rec["levels"].items():
            rows.append({"market": m, "level": name, "lam_bp": r["lam_bp"], "fund_vol": r["fund_vol"], "ratio_cal": r["ratio"], "n_evals": len(r["evals"]),
                         "converged_2pct": r["converged"], "feasible": r["feasible"], "ratio_check": r["ratio_check"], "check_lo": r["check_lo"],
                         "check_hi": r["check_hi"], "equivalent": equivalent(r), "interval_includes_1": r["interval_includes_1"], "kept": m in kept,
                         "viable": via[m]["viable"], "drop_reason": "" if m in kept else reason(rec, via[m]["viable"]), "wild_share": r["wild_share"], **r["desc"]})
    return pd.DataFrame(rows)


def finalise(final: dict) -> dict:
    """Write the table, the market list and the arms of the main run."""
    final_table(final).to_csv(P.OUT / "calib_final.csv", index=False)
    P.write_markets(kept_markets(final))
    V = P.load_json("pilot_markets.json")["reference_volume"]
    kept = kept_markets(final)
    arms = {"lam0": {m: {} for m in kept}}
    for name, lam in P.LEVELS.items():
        arms[f"all_{name}"] = {m: {"lam_bp": lam, "reference_volume": V[m], "fund_vol": final[m]["levels"][name]["fund_vol"]} for m in kept}
    arms[f"trader_{TRADER_LEVEL}"] = {m: {"lam_bp": P.LEVELS[TRADER_LEVEL], "reference_volume": V[m], "mode": "trader"} for m in kept}
    n_drawn = max(int(m[-2:]) for m in final)
    doc = {"levels_bp_per_1pct_volume": P.LEVELS, "judged": ["all_band_low", "all_max_feasible"], "unjudged": ["all_dose", f"trader_{TRADER_LEVEL}"],
           "matching": {"calibration_seeds": [C.SEED_CALIB, C.SEED_CALIB + P.N_VOL_CAL_FINAL - 1], "tolerance": P.VOL_TOL_FINAL,
                        "check_seeds": [P.SEED_VOLCHECK, P.SEED_VOLCHECK + P.N_VOL_CHECK - 1], "equivalence_bounds": [EQ_LO, EQ_HI], "cap": f"a{CAP}"},
           "agent_mix_draws_examined": n_drawn, "kept": kept, "acceptance_rate": len(kept) / n_drawn, "arms": arms}
    (P.OUT / "arms.json").write_text(json.dumps(doc, indent=2))
    return doc


def persist_final(procs: int) -> pd.DataFrame:
    """Persistence of the final markets at lambda = 0 and the three levels (description only)."""
    from simrel.experiments.run_reference import pool_map

    A = P.arms()
    V = P.load_json("pilot_markets.json")["reference_volume"]
    cells = [("0.0", A["lam0"])] + [(str(lam), A[f"all_{name}"]) for name, lam in P.LEVELS.items()]
    jobs = [(m, tag, kw[m], V[m], P.SEED_PERSIST + s) for tag, kw in cells for m in kw for s in range(P.N_PERSIST)]
    out = persistence_table(P.cached("calib_persist_final", lambda: pool_map(P.persist_job, jobs, procs, "persist final")))
    out.to_csv(P.OUT / "calib_persist_final.csv", index=False)
    return out


def main(stage: str, procs: int) -> None:
    if stage == "grid":
        vol = vol_stage(procs)
        print("feasible grid:", feasible_grid(vol))
        out = persist_stage(procs, vol)
    else:
        final = vol_final(procs)
        doc = finalise(final)
        print(f"kept {doc['kept']}: {len(doc['kept'])} of {doc['agent_mix_draws_examined']} draws")
        out = persist_final(procs)
    print(out[out["market"] == "pooled"].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final", int(sys.argv[2]) if len(sys.argv) > 2 else 6)
