"""
Real-market reference for the stylised-fact table.

Computes the same observables as simbench.runner._observables on the one
public LOBSTER sample day that ships with DeepMarket (INTC, 2012-06-21), for
the same clock window the ABIDES task uses (10:00-10:30) and for the TRADES
task window (09:46-09:56). One stock on one day is a reference point, not a
benchmark.

    python experiments/simbench_05_real_reference.py /path/to/DeepMarket
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simbench.runner import RESULTS_DIR, _acf1


def observables(msg, ob, t0, horizon):
    t = msg["time"].to_numpy()
    bid, ask = ob["buy1"].to_numpy(float), ob["sell1"].to_numpy(float)
    grid = t0 + np.arange(0, horizon + 1, 1.0)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    mid, spr = 0.5 * (bid + ask)[idx], (ask - bid)[idx]
    d5 = 0.5 * (ob[[f"vbuy{i}" for i in range(1, 6)]].to_numpy().sum(1) + ob[[f"vsell{i}" for i in range(1, 6)]].to_numpy().sum(1))[idx]
    r10 = np.diff(np.log(mid[::10])) * 1e4
    ex = msg[msg["event_type"].isin([4, 5]) & (msg["time"] >= t0) & (msg["time"] <= t0 + horizon)]
    # LOBSTER direction is the resting order's side: a resting buy that executes is a sell-initiated trade
    signed = -ex["direction"].to_numpy() * ex["size"].to_numpy()
    ofi = np.histogram(ex["time"].to_numpy() - t0, bins=np.arange(0, horizon + 1, 30), weights=signed)[0]
    sd = r10.std()
    return {
        "obs_spread_bps": float(spr.mean() / mid[0] * 1e4), "obs_depth5": float(d5.mean()), "obs_vol10s_bps": float(sd),
        "obs_ret_acf1": _acf1(r10), "obs_absret_acf1": _acf1(np.abs(r10)),
        "obs_kurtosis": float(((r10 - r10.mean()) ** 4).mean() / sd**4 - 3.0), "obs_volume": float(ex["size"].sum()),
        "obs_n_trades": float(len(ex)), "obs_trade_size": float(ex["size"].mean()), "obs_ofi_std": float(ofi.std()),
        "obs_ofi_acf1": _acf1(ofi),
    }


if __name__ == "__main__":
    d = Path(sys.argv[1]) / "data" / "INTC" / "INTC_2012-06-21_2012-06-21"
    msg = pd.read_csv(d / "INTC_2012-06-21_34200000_57600000_message_10.csv", names=["time", "event_type", "order_id", "size", "price", "direction"])
    cols = [f"{s}{i}" for i in range(1, 11) for s in ("sell", "vsell", "buy", "vbuy")]
    ob = pd.read_csv(d / "INTC_2012-06-21_34200000_57600000_orderbook_10.csv", names=cols)
    out = {
        "REAL_INTC_1000_1030": observables(msg, ob, 10 * 3600.0, 1800),
        "REAL_INTC_0946_0956": observables(msg, ob, 9 * 3600.0 + 46 * 60, 600),
    }
    (RESULTS_DIR / "real_reference.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(out).round(3).to_string())
