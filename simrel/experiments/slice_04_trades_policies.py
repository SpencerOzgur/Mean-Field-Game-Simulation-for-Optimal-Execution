"""
Execution policies in the TRADES simulator, compared with ABIDES and Hawkes in the slice TRADES covers.

    python simrel/experiments/slice_04_trades_policies.py <dir with pol_<date>_s<seed>_<policy>.pkl and gen_<date>_s<seed>.pkl>

Costs use the same accounting as the other simulators (simrel.metrics.cost.shortfall_bps): implementation shortfall in
bps of the arrival mid, positive = worse, with the unperturbed twin (same day and seed) supplying the arrival mid.
"""

from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel import slices as S  # noqa: E402
from simrel.metrics.cost import boot_ci, shortfall_bps  # noqa: E402

NAMES = {"exp:0": "P0_twap", "immediate": "P1_immediate", "exp:2": "P2_front_b2", "exp:-2": "P3_back_b-2"}
SLICE = "volH_voluH_trendH"


def mid_at(obs: np.ndarray, t: float) -> float:
    i = int(np.argmin(np.abs(obs[:, 0] - t)))
    return 0.5 * (obs[i, 1] + obs[i, 2])


if __name__ == "__main__":
    d = Path(sys.argv[1]).expanduser()
    rows = []
    for p in sorted(glob.glob(str(d / "pol_*.pkl"))):
        _, date, seed, pol = Path(p).stem.split("_", 3)
        rec = pickle.load(open(p, "rb"))
        twin = pickle.load(open(d / f"gen_{date}_{seed}.pkl", "rb"))
        arrival = mid_at(twin["obs"], 0.0)
        f = rec["fills"]
        n = rec["n_slices"]
        k = np.clip((f[:, 0] // (rec["horizon_s"] / n)).astype(int), 0, n - 1) if len(f) else np.array([], dtype=int)
        qty, notional = np.zeros(n), np.zeros(n)
        np.add.at(qty, k, f[:, 2])
        np.add.at(notional, k, f[:, 1] * f[:, 2])
        cost, unfilled = shortfall_bps(rec["side"], arrival, qty, notional, rec["total_qty"])
        tw_mid = np.array([mid_at(twin["obs"], t) for t in rec["slices"][:, 1]])
        ok = qty > 0
        impact = float((qty[ok] * (tw_mid[ok] - notional[ok] / qty[ok])).sum() / qty[ok].sum() / arrival * 1e4)
        ob = rec["obs"]
        w = (ob[:, 0] >= 0) & (ob[:, 0] <= rec["horizon_s"])
        rows.append({"sim": "trades", "slice": SLICE, "date": date, "seed": f"{date}_{seed}", "policy": NAMES[pol], "is_bps": cost, "unfilled_frac": unfilled,
                     "impact_bps": impact, "spread_bps": float(np.nanmean((ob[w, 2] - ob[w, 1])) / arrival * 1e4),
                     "twin_move_bps": float((mid_at(twin["obs"], rec["horizon_s"]) - arrival) / arrival * 1e4),
                     "twins_identical_before_start": bool(np.allclose(rec["obs"][rec["obs"][:, 0] < 0], twin["obs"][twin["obs"][:, 0] < 0], equal_nan=True))})
    t = pd.DataFrame(rows)
    out = S.out_dir()
    t.to_csv(out / "trades_policy_episodes.csv", index=False)
    other = pd.read_csv(out / "policy_episodes.csv")
    other = other[(other["slice"] == SLICE) & other["sim"].isin(["abides", "hawkes"]) & other["policy"].isin(NAMES.values())]
    allr = pd.concat([t[["sim", "seed", "policy", "is_bps", "impact_bps"]], other[["sim", "seed", "policy", "is_bps", "impact_bps"]]])
    res = []
    for sim, g in allr.groupby("sim"):
        wide = g.pivot(index="seed", columns="policy", values="is_bps").dropna()
        best = wide.mean().idxmin()
        for pol in wide.columns:
            lo, hi = boot_ci(wide[pol].to_numpy())
            dlo, dhi = boot_ci((wide[pol] - wide[best]).to_numpy()) if pol != best else (0.0, 0.0)
            res.append({"sim": sim, "policy": pol, "n": len(wide), "mean_is_bps": wide[pol].mean(), "ci_lo": lo, "ci_hi": hi,
                        "impact_bps": g[g["policy"] == pol]["impact_bps"].mean(), "excess_over_best": wide[pol].mean() - wide[best].mean(), "excess_lo": dlo, "excess_hi": dhi})
    res = pd.DataFrame(res)
    res.to_csv(out / "trades_policy_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(2).to_string(index=False))
    print(t.groupby("policy")[["is_bps", "impact_bps", "unfilled_frac", "spread_bps", "twin_move_bps"]].mean().round(2).to_string())
    print("twins identical before the task start in", int(t["twins_identical_before_start"].sum()), "of", len(t), "episodes")
    print(t.pivot(index="seed", columns="policy", values="is_bps").round(1).to_string())
