"""
Observational realism: how closely a simulator's unperturbed market resembles a target market.

Both sides are summarised from the same thing every adapter records for an
unperturbed twin: the 5-second quote grid over the task window and the traded
volume. Prices are converted to basis points of the arrival mid so markets
with different price levels can be compared.

Per-episode statistics:
    spread_bps         mean quoted spread
    vol_bps            standard deviation of 5-second mid returns
    kurtosis           excess kurtosis of those returns
    ret_acf1           their lag-1 autocorrelation
    absret_acf1        lag-1 autocorrelation of absolute returns (volatility clustering)
    move_bps_abs       absolute mid change over the window
    volume_rel         traded volume relative to the target's mean (only when units are shared)

The realism error of a simulator is the mean, over statistics, of
|mean_sim - mean_target| / scale, where scale is the dispersion of the
statistic across the target's own episodes, floored at a quarter of its mean
so that a statistic the target pins down exactly (a one-tick spread, say) does
not dominate. An error near or below 1 means the simulator is within the
target's own episode-to-episode variation.
"""

from __future__ import annotations

import numpy as np

STATS = ["spread_bps", "vol_bps", "kurtosis", "ret_acf1", "absret_acf1", "move_bps_abs"]


def _acf1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else 0.0


def episode_stats(rec) -> dict:
    t = rec.task
    w = (rec.grid_t >= 0) & (rec.grid_t <= t.horizon_s)
    mid, spr = rec.grid_mid[w], rec.grid_spread[w]
    r = np.diff(mid) / rec.arrival_mid * 1e4
    sd = float(r.std())
    return {
        "spread_bps": float(np.nanmean(spr) / rec.arrival_mid * 1e4),
        "vol_bps": sd,
        "kurtosis": float(((r - r.mean()) ** 4).mean() / sd**4 - 3.0) if sd > 0 else 0.0,
        "ret_acf1": _acf1(r),
        "absret_acf1": _acf1(np.abs(r)),
        "move_bps_abs": float(abs(mid[-1] - mid[0]) / rec.arrival_mid * 1e4),
        "volume": float(rec.background_volume),
    }


def realism_error(sim_stats: list[dict], target_stats: list[dict], with_volume: bool = False) -> tuple[float, dict]:
    keys = STATS + (["volume"] if with_volume else [])
    parts = {}
    for k in keys:
        s = np.array([d[k] for d in sim_stats], dtype=float)
        g = np.array([d[k] for d in target_stats], dtype=float)
        scale = max(np.nanstd(g), 0.25 * abs(np.nanmean(g)), 1e-9)  # floor: a statistic that barely varies in the target is not held to its noise
        parts[k] = float(abs(np.nanmean(s) - np.nanmean(g)) / scale)
    return float(np.mean(list(parts.values()))), parts
