"""
Execution-cost accounting, identical for every simulator.

Sign convention: every cost is in basis points of the arrival mid and positive
means worse for the trader, for buys and sells alike.

    is_bps       implementation shortfall: arrival mid against the volume-weighted
                 fill price. This is the decision metric.
    impact_bps   the same fills measured against the unperturbed twin's mid at each
                 decision time: the part of the shortfall the order itself caused.
    disp_end_bps adverse shift of the mid at the deadline relative to the twin.

Shares the market could not absorb by the deadline are charged at the worst
slice price of the episode and reported as `unfilled_frac`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EpisodeRecord


def side_sign(side: str) -> float:
    return 1.0 if side == "sell" else -1.0


def shortfall_bps(side: str, arrival_mid: float, fill_qty: np.ndarray, fill_notional: np.ndarray, total_qty: float) -> tuple[float, float]:
    """Implementation shortfall with unfilled shares charged at the worst slice price. Returns (cost, unfilled fraction)."""
    sgn = side_sign(side)
    ok = fill_qty > 0
    filled = float(fill_qty[ok].sum())
    if filled <= 0:
        return np.nan, 1.0
    px = fill_notional[ok] / fill_qty[ok]
    per_slice = sgn * (arrival_mid - px) / arrival_mid * 1e4
    cost = float((fill_qty[ok] * per_slice).sum() / filled)
    unfilled = max(1.0 - filled / total_qty, 0.0)
    return (1.0 - unfilled) * cost + unfilled * max(float(per_slice.max()), cost), unfilled


def episode_metrics(rec: EpisodeRecord, twin: EpisodeRecord) -> dict:
    t = rec.task
    row = {"world": rec.world, "seed": rec.seed, "policy": rec.policy, "side": t.side, "total_qty": t.total_qty,
           "failed": rec.failed or twin.failed, "wall_s": rec.wall_s}
    if row["failed"] or rec.fill_qty is None:
        return row
    sgn = side_sign(t.side)
    arr = twin.arrival_mid
    row["is_bps"], row["unfilled_frac"] = shortfall_bps(t.side, arr, rec.fill_qty, rec.fill_notional, t.total_qty)
    ok = (rec.fill_qty > 0) & np.isfinite(twin.slice_mid)
    if ok.any():
        px = rec.fill_notional[ok] / rec.fill_qty[ok]
        row["impact_bps"] = float((rec.fill_qty[ok] * sgn * (twin.slice_mid[ok] - px)).sum() / rec.fill_qty[ok].sum() / arr * 1e4)
    m = min(len(rec.grid_mid), len(twin.grid_mid))
    d = -sgn * (rec.grid_mid[:m] - twin.grid_mid[:m]) / arr * 1e4
    gt = rec.grid_t[:m]
    row["disp_end_bps"] = float(np.interp(t.horizon_s, gt, d))
    row["disp_post_bps"] = float(d[-1])
    win = (gt >= 0) & (gt <= t.horizon_s)
    row["spread_bps"] = float(np.nanmean(rec.grid_spread[:m][win]) / arr * 1e4)
    row["twin_spread_bps"] = float(np.nanmean(twin.grid_spread[:m][win]) / arr * 1e4)
    row["twin_move_bps"] = float((np.interp(t.horizon_s, gt, twin.grid_mid[:m]) - arr) / arr * 1e4)
    row["background_volume"] = twin.background_volume
    done = np.nonzero(np.cumsum(rec.fill_qty) >= t.total_qty)[0]
    row["completion_frac"] = float((done[0] + 1) / t.n_slices) if len(done) else np.nan
    return row


def boot_ci(x, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rs = np.random.RandomState(seed)
    m = x[rs.randint(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(df: pd.DataFrame, value: str = "is_bps") -> pd.DataFrame:
    rows = []
    for keys, d in df.groupby(["world", "size_frac", "policy"]):
        v = d[value].dropna().to_numpy()
        lo, hi = boot_ci(v)
        rows.append({"world": keys[0], "size_frac": keys[1], "policy": keys[2], "n": len(v), "mean": v.mean() if len(v) else np.nan,
                     "ci_lo": lo, "ci_hi": hi, "median": np.median(v) if len(v) else np.nan,
                     "p90": np.percentile(v, 90) if len(v) else np.nan, "p95": np.percentile(v, 95) if len(v) else np.nan,
                     "unfilled_frac": d["unfilled_frac"].mean(), "impact_bps": d["impact_bps"].mean(), "disp_end_bps": d["disp_end_bps"].mean()})
    return pd.DataFrame(rows)
