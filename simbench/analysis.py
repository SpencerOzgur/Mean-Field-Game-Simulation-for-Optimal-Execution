"""
Cost accounting shared by the experiment scripts.

All costs are in basis points of the arrival mid and signed so that positive
means worse for the trader, for both buys and sells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .runner import load

GRID30_START_S = -600  # mid30 starts 10 minutes before the execution start


def baseline_index(name: str = "baselines") -> dict:
    return {(r["world"], r["seed"]): r for r in load(name) if not r.get("timeout")}


def to_frame(recs: list[dict], base: dict) -> pd.DataFrame:
    """One row per episode with raw and paired costs.

    is_bps       raw implementation shortfall against the arrival mid
    cost_bps     paired impact cost: each fill is compared with the mid the
                 unperturbed twin episode (same world, same seed) had at that
                 decision time, so background drift and noise cancel
    cost_pen_bps cost_bps with unfilled shares charged at the episode's worst
                 per-share slice cost (equal to cost_bps when fully filled)
    slip_bps     volume-weighted fill price against the trader's own pre-trade
                 mid in each slice (spread crossing plus book walking)
    disp_end_bps adverse displacement of the mid at the deadline vs the twin
    disp_post_bps same, 5 minutes after the deadline
    """
    rows = []
    for r in recs:
        b = base.get((r["world"], r["seed"]))
        if b is None or r.get("timeout"):  # runaway episode (or its twin) was cut off
            continue
        sgn = 1.0 if r["side"] == "sell" else -1.0
        arr = b["arrival_mid"]
        fq = r["fill_qty"].astype(np.float64)
        filled = fq.sum()
        row = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        bm = b["slice_mid"].astype(np.float64)
        ok = (fq > 0) & np.isfinite(bm)  # skip slices where the twin's book was one-sided
        if ok.any():
            row["cost_bps"] = sgn * float((fq[ok] * bm[ok] - r["fill_notional"][ok]).sum()) / max(fq[ok].sum(), 1.0) / arr * 1e4
            om = r["slice_mid"].astype(np.float64)
            fin = ok & np.isfinite(om)
            row["slip_bps"] = sgn * float((fq[fin] * om[fin] - r["fill_notional"][fin]).sum()) / max(fq[fin].sum(), 1.0) / arr * 1e4
            # shares the book could not absorb are charged at the worst slice's per-share cost
            worst = float(np.max(sgn * (bm[ok] - r["fill_notional"][ok] / fq[ok]) / arr * 1e4))
            ff = min(filled / r["total_qty"], 1.0)
            row["cost_pen_bps"] = ff * row["cost_bps"] + (1.0 - ff) * max(worst, row["cost_bps"])
        d = -sgn * (r["mid30"] - b["mid30"]) / arr * 1e4
        i_end = int((r.get("horizon_s", 1800) - GRID30_START_S) // 30)
        row["disp_end_bps"] = float(d[i_end])
        row["disp_post_bps"] = float(d[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def displacement_paths(recs: list[dict], base: dict) -> np.ndarray:
    """Adverse mid displacement vs the twin episode on the 30 s grid, in bps."""
    out = []
    for r in recs:
        b = base.get((r["world"], r["seed"]))
        if b is None or r.get("timeout"):
            continue
        sgn = 1.0 if r["side"] == "sell" else -1.0
        out.append(-sgn * (r["mid30"] - b["mid30"]) / b["arrival_mid"] * 1e4)
    return np.array(out)


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, q=(2.5, 97.5)) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rs = np.random.RandomState(seed)
    m = x[rs.randint(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return tuple(np.percentile(m, q))
