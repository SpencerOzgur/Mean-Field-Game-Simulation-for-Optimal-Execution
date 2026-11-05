"""
Stage 4: TRADES arm. Paired impact costs of each schedule in the retrained
TRADES market, next to the ABIDES worlds on the identical 10-minute task
(stage 3), plus the TRADES market's stylised facts against the real day it
was trained on.

Run from repo root (after run_trades_batch.py and stages 03 and 05):
    python experiments/simbench_04_trades_analyze.py
"""

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simbench.analysis import boot_ci
from simbench.runner import RESULTS_DIR, _acf1

RAW = RESULTS_DIR / "raw" / "trades"
NAME = {"exp-2": "exp_b-2", "exp0": "exp_b0", "exp2": "exp_b2", "exp8": "exp_b8", "immediate": "immediate"}

if __name__ == "__main__":
    base, runs = {}, []
    for p in sorted(RAW.glob("*.pkl")):
        r = pickle.load(open(p, "rb"))
        if p.name.startswith("base_"):
            base[r["seed"]] = r
        else:
            r["policy"] = NAME[p.name.split("_s")[0]]
            runs.append(r)

    rows, paths = [], {}
    for r in runs:
        b = base.get(r["seed"])
        if b is None or len(r["fills"]) == 0:
            continue
        H, n = r["horizon_s"], r["n_slices"]
        dt = H / n
        bm = np.full(n, np.nan)
        bm[b["slices"][:, 0].astype(int)] = b["slices"][:, 2]
        arrival = bm[0]
        k = np.clip((r["fills"][:, 0] // dt).astype(int), 0, n - 1)
        px, q = r["fills"][:, 1], r["fills"][:, 2]
        filled = q.sum()
        per_share = (bm[k] - px) / arrival * 1e4  # sell: positive = worse than the twin's mid
        ok = np.isfinite(per_share)  # the twin's book can be one-sided at a decision time
        cost = float((q[ok] * per_share[ok]).sum() / q[ok].sum())
        ff = min(filled / r["total_qty"], 1.0)
        cost_pen = ff * cost + (1 - ff) * max(float(per_share[ok].max()), cost)
        tb, tr = b["obs"], r["obs"]
        m = min(len(tb), len(tr))
        disp = -(0.5 * (tr[:m, 1] + tr[:m, 2]) - 0.5 * (tb[:m, 1] + tb[:m, 2])) / arrival * 1e4
        t = tr[:m, 0]
        pre = t < 0
        pre_same = bool(np.array_equal(tr[:m][pre, 1:], tb[:m][pre, 1:]))  # twin identical before the first order
        paths.setdefault(r["policy"], []).append(np.interp(np.arange(-60, H + 121, 5.0), t, disp))
        rows.append({"world": "T1_trades_intc", "policy": r["policy"], "seed": r["seed"], "pre_start_identical": pre_same,
                     "post_start_differs": bool(not np.array_equal(tr[:m][~pre, 1:], tb[:m][~pre, 1:])), "cost_bps": cost,
                     "cost_pen_bps": cost_pen, "fill_frac": filled / r["total_qty"],
                     "disp_end_bps": float(np.interp(H, t, disp)), "disp_post_bps": float(np.interp(H + 115, t, disp)),
                     "total_qty": r["total_qty"], "generated_orders": r["generated_orders"], "wall_s": r["wall_s"]})
    df = pd.DataFrame(rows)
    out = []
    for p, d in df.groupby("policy"):
        lo, hi = boot_ci(d["cost_pen_bps"].to_numpy())
        out.append({"world": "T1_trades_intc", "policy": p, "cost_bps": d["cost_pen_bps"].mean(), "ci_lo": lo, "ci_hi": hi,
                    "fill_frac": d["fill_frac"].mean(), "disp_end_bps": d["disp_end_bps"].mean(),
                    "disp_post_bps": d["disp_post_bps"].mean(), "n": len(d)})
    trades = pd.DataFrame(out)
    piv = df.pivot(index="seed", columns="policy", values="cost_pen_bps")
    vs = {}
    for p in piv.columns:
        if p != "exp_b0" and "exp_b0" in piv:
            d = (piv[p] - piv["exp_b0"]).dropna().to_numpy()
            lo, hi = boot_ci(d)
            vs[p] = {"diff_bps": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "n": int(len(d)), "n_positive": int((d > 0).sum())}
    trades.round(3).to_csv(RESULTS_DIR / "table_short_horizon_trades.csv", index=False)
    df.round(3).to_csv(RESULTS_DIR / "table_trades_episodes.csv", index=False)

    # combined 10-minute task table: ABIDES worlds + TRADES
    ab = pd.read_csv(RESULTS_DIR / "table_short_horizon_abides.csv")
    both = pd.concat([ab, trades], ignore_index=True)
    wide = both.pivot(index="policy", columns="world", values="cost_bps")
    wide.round(2).to_csv(RESULTS_DIR / "table_short_horizon_all.csv")
    sched = [p for p in ["exp_b-2", "exp_b0", "exp_b2", "exp_b8"] if p in wide.index]
    best = {w: wide.loc[sched, w].idxmin() for w in wide.columns}
    regret = pd.DataFrame({wj: {wi: wide.loc[best[wi], wj] - wide.loc[best[wj], wj] for wi in wide.columns} for wj in wide.columns})
    regret.round(2).to_csv(RESULTS_DIR / "table_short_horizon_regret.csv")

    # TRADES twin-market stylised facts over the task window, against the real day
    obs = []
    for b in base.values():
        t, bid, ask = b["obs"][:, 0], b["obs"][:, 1], b["obs"][:, 2]
        w = (t >= 0) & (t <= b["horizon_s"])
        mid = 0.5 * (bid + ask)[w]
        r10 = np.diff(np.log(mid[::2])) * 1e4  # observations are 5 s apart
        obs.append({"obs_spread_bps": float(np.nanmean((ask - bid)[w]) / mid[0] * 1e4), "obs_vol10s_bps": float(r10.std()),
                    "obs_ret_acf1": _acf1(r10), "obs_absret_acf1": _acf1(np.abs(r10)),
                    "obs_volume": float(np.sum(b["window_volume"])), "generated_orders": b["generated_orders"]})
    obs = pd.DataFrame(obs)
    # does the generated market drift? quoted spread early vs late in the window, across twins
    def spread_in(lo, hi):
        v = []
        for b in base.values():
            t, bid, ask = b["obs"][:, 0], b["obs"][:, 1], b["obs"][:, 2]
            w = (t >= lo) & (t < hi)
            v.append(np.nanmean((ask - bid)[w] / (0.5 * (ask + bid))[w]) * 1e4)
        return float(np.mean(v))

    spread_path = {"replay_last_minute": spread_in(-60, 0), "first_minute": spread_in(0, 60),
                   "minute_5": spread_in(240, 300), "last_minute": spread_in(540, 600)}
    real = json.loads((RESULTS_DIR / "real_reference.json").read_text())["REAL_INTC_0946_0956"]
    cmp = pd.DataFrame({"TRADES_retrained_mean": obs.mean(), "TRADES_retrained_sd": obs.std(),
                        "REAL_INTC_0946_0956": pd.Series({k: real.get(k, np.nan) for k in obs.columns})})
    cmp.round(3).to_csv(RESULTS_DIR / "table_trades_vs_real.csv")

    summary = {"n_twins": len(base), "n_runs": len(df), "best_schedule": best, "spread_path_bps": spread_path, "vs_twap": vs,
               "runs_per_policy": df.groupby("policy").size().to_dict(),
               "pre_start_identical_frac": float(df["pre_start_identical"].mean()) if len(df) else None,
               "post_start_differs_frac": float(df["post_start_differs"].mean()) if len(df) else None,
               "mean_wall_s": float(df["wall_s"].mean()) if len(df) else None,
               "parent_qty": int(df["total_qty"].iloc[0]) if len(df) else None}
    (RESULTS_DIR / "findings_trades.json").write_text(json.dumps(summary, indent=2))
    np.save(RESULTS_DIR / "trades_disp_paths.npy", {p: np.mean(v, axis=0) for p, v in paths.items()}, allow_pickle=True)
    print(json.dumps(summary, indent=2)); print(wide.round(2).to_string()); print(regret.round(2).to_string()); print(cmp.round(3).to_string())
