"""
Design addendum 11: TRADES on three development markets, a descriptive robustness comparison.

Every design is scored on the same reduced decision: the cheapest predicted of four schedules (TWAP, sell at once,
back-loaded, front-loaded) at 15% of reference volume; regret is the chosen schedule's true cost minus the cheapest true
cost among the four (truth: the market's 300 reference seeds). TRADES costs use the accounting of the Goal 1 TRADES case
study (simrel.metrics.cost.shortfall_bps, arrival mid from the seed's unperturbed twin).

    python simrel/experiments/goal2_trades.py ~/simdata/trades_syn
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

from simrel.experiments import goal2_addendum8 as A  # noqa: E402
from simrel.experiments import goal2_budget as GB  # noqa: E402
from simrel.metrics.cost import shortfall_bps  # noqa: E402

G = A.G
MARKETS = ["R1a_qr_resilient", "R2a_am_anchored", "W6_hcfm_R2a"]
SCHED = {"exp:0": "P0_twap", "immediate": "P1_immediate", "exp:-2": "P3_back_b-2", "exp:4": "P2_front_b4"}
FOUR = sorted(SCHED.values())
SIZE = 0.15
DESIGNS = ["h_k1", "h_c_f_m_k1", "abides", "qr", "chawkes", "h_anchor", "h_k1_r24", "h_c_f_m_k1_r24", "oracle"]


def mid_at(obs: np.ndarray, t: float) -> float:
    i = int(np.argmin(np.abs(obs[:, 0] - t)))
    return 0.5 * (obs[i, 1] + obs[i, 2])


def trades_costs(d: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(d / "pol_*.pkl"))):
        _, date, seed, pol = Path(p).stem.split("_", 3)
        rec = pickle.load(open(p, "rb"))
        twin = pickle.load(open(d / f"gen_{date}_{seed}.pkl", "rb"))
        arrival = mid_at(np.asarray(twin["obs"]), 0.0)
        f = np.asarray(rec["fills"], float).reshape(-1, 3)
        n = rec["n_slices"]
        k = np.clip((f[:, 0] // (rec["horizon_s"] / n)).astype(int), 0, n - 1) if len(f) else np.array([], dtype=int)
        qty, notional = np.zeros(n), np.zeros(n)
        np.add.at(qty, k, f[:, 2])
        np.add.at(notional, k, f[:, 1] * f[:, 2])
        cost, unfilled = shortfall_bps(rec["side"], arrival, qty, notional, rec["total_qty"])
        rows.append({"seed": int(seed[1:]), "policy": SCHED[pol], "is_bps": cost, "unfilled_frac": unfilled})
    return pd.DataFrame(rows)


def main() -> None:
    base = Path(sys.argv[1]).expanduser()
    truth = A.truth_tables(MARKETS)
    rows, gains = [], []
    for w in MARKETS:
        tp, tmean = truth[(w, SIZE)][0], truth[(w, SIZE)][1]
        tc = pd.Series(tmean, index=tp)[FOUR]
        best = tc.min()
        regret = lambda s: float(tc[s] - best)  # noqa: E731
        rows.append({"market": w, "design": "always TWAP", "seeds": "-", "chosen": "P0_twap", "regret": regret("P0_twap")})
        preds, extra = {}, {}
        d = base / w
        if d.exists() and glob.glob(str(d / "pol_*.pkl")):
            t = trades_costs(d)
            preds[("TRADES", 20)] = t.groupby("policy").is_bps.mean().reindex(FOUR)
            extra = {"unfilled": float(t.unfilled_frac.mean()), "n_episodes": len(t)}
        for des in DESIGNS:
            m = GB.matrix(w, des, SIZE, 100)
            if m is None:
                continue
            pols, cost = m
            c = pd.DataFrame(cost, columns=pols)[FOUR]
            for n in (20, 100):
                preds[(des, n)] = c.iloc[:n].mean()
        for (des, n), pm in preds.items():
            ch = pm.round(9).sort_index().idxmin()
            rows.append({"market": w, "design": des, "seeds": n, "chosen": ch, "regret": regret(ch), **(extra if des == "TRADES" else {})})
            for a in FOUR:
                if a != "P0_twap":
                    gains.append({"market": w, "design": des, "seeds": n, "strategy": a, "predicted gain": float(pm["P0_twap"] - pm[a]),
                                  "true gain": float(tc["P0_twap"] - tc[a])})
    t, g = pd.DataFrame(rows), pd.DataFrame(gains)
    g["error"] = g["predicted gain"] - g["true gain"]
    out = G.BASE / "addendum11"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "trades_comparison.csv", index=False)
    g.to_csv(out / "trades_gains.csv", index=False)
    pd.set_option("display.width", 200)
    print("reduced decision (four schedules, 15%): regret in bp; chosen schedule")
    show = t[t.seeds.isin(["-", 20])].pivot(index="design", columns="market", values="regret")
    show["mean"] = show.mean(axis=1)
    print(show.sort_values("mean").round(2).to_string())
    print("\nchosen schedule (20 seeds)")
    print(t[t.seeds.isin(["-", 20])].pivot(index="design", columns="market", values="chosen").to_string())
    print("\ntrue cost of the four schedules (bp)")
    print(pd.DataFrame({w: pd.Series(truth[(w, SIZE)][1], index=truth[(w, SIZE)][0])[FOUR] for w in MARKETS}).round(2).to_string())
    print("\nmean absolute error in predicted gain over TWAP (bp), 20 seeds")
    print(g[g.seeds == 20].assign(abs_error=lambda x: x.error.abs()).pivot_table(index="design", columns="market", values="abs_error").round(2).to_string())


if __name__ == "__main__":
    main()
