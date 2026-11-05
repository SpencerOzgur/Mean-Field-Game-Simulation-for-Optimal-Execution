"""
Cross-check of the ported LOB-Impact estimators against the reference tool, on the same CSV rollouts.

impact_01_run.py writes a few INTC rollouts (Hawkes-original and ABIDES-original, six per arm and side) as LOBSTER
CSVs in the reference tool's layout under results/simrel/lob_impact/INTC/csv. This script computes our numbers on
exactly those files; the reference tool is run on them separately (commands printed below) and the two outputs
should agree: number of points, delta (binned), gamma_flow and H.

    python simrel/experiments/impact_03_crosscheck.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.metrics import lob_impact as LI  # noqa: E402

BASE = ROOT / "results" / "simrel" / "lob_impact" / "INTC"
CSV = BASE / "csv_unique"  # one placeholder day per rollout (impact_05_unique_days.py)
DATE = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


def read(msg_file: str, side: str, day: str, aggr) -> dict:
    m = pd.read_csv(msg_file, header=None).to_numpy(float)
    ob = pd.read_csv(msg_file.replace("message", "orderbook"), header=None).to_numpy(float)
    ask, bid = ob[:, 0].copy(), ob[:, 2].copy()
    bad = (ask >= 2147483647) | (bid >= 2147483647) | (ask <= 0) | (bid <= 0)
    ask[bad], bid[bad] = np.nan, np.nan
    return {"msg": m, "ask": ask, "bid": bid, "aggr": np.asarray(aggr, int), "day": day, "sign": 1 if side == "buy" else -1}


def rollouts(root: Path, sides=("buy", "sell"), n_ins=100, n_cool=0) -> list:
    recs = []
    for side in sides:
        for f in sorted(glob.glob(str(root / side / "**" / "data_gen" / "*message*gen*.csv"), recursive=True)):
            day = DATE.search(os.path.basename(f)).group(1)
            ai = Path(f).parent.parent / f"aggressive_indices_{day}.csv"
            aggr = np.loadtxt(ai, dtype=int, ndmin=1) if ai.exists() else []
            recs.append(LI.reduce(read(f, side, day, aggr), n_ins, n_cool))
    return recs


def main() -> None:
    ds = pd.read_csv(BASE / "daily_stats_unique.csv").set_index("day")
    daily = {d: {"V": float(r["execution_sum"]), "range": float(np.log(r["highest_price"] / r["lowest_price"]))} for d, r in ds.iterrows()}
    for model in ("hawkes", "abides"):
        beta = rollouts(CSV / "grid" / f"INTC-{model}-beta")
        pts = LI.points(beta)
        b = LI.build_up(pts, daily)
        K, _ = LI.k_clock(beta, 100, 0)
        free = rollouts(CSV / "controls" / "noins" / f"INTC-{model}-beta", sides=("buy",), n_ins=0)
        p1 = LI.phase1(free)
        col = K[:, 100][np.isfinite(K[:, 100])]
        print(f"{model}: {len(pts['Q'])} points, delta {b['delta']:.3f} [{b['delta_lo']:.3f}, {b['delta_hi']:.3f}], "
              f"k-clock at k=100 {col.mean():.2f} +/- {col.std(ddof=1) / np.sqrt(len(col)):.2f} bps (n={len(col)}); "
              f"free runs {p1.get('n_runs')}: gamma_flow {p1.get('gamma_flow', np.nan):.3f}, H {p1.get('H', np.nan):.3f}")
    print("\nReference tool, run from its own checkout (after `pip install -e .` in a separate environment):")
    print(f"  export LOB_IMPACT_CONTROLS={CSV}/controls LOB_IMPACT_GRID={CSV}/grid LOB_IMPACT_RESULTS={CSV}/tool_results")
    print("  lob_impact points --stock INTC --models hawkes,abides")
    print(f"  lob_impact delta --stock INTC --models hawkes,abides --daily {BASE}/daily_stats_unique.csv")
    print("  lob_impact flow_balance --stock INTC --models hawkes,abides --ref hawkes --no_plot")


if __name__ == "__main__":
    main()
