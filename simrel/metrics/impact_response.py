"""
Price response to aggressive orders, measured identically in real and simulated LOBSTER windows.

An aggressive order is the set of executions with the same timestamp against the same resting side (in LOBSTER the
direction field is the resting side, so executions against resting buy orders are a sell). For each one, the response
is the change in mid price from just before it to `h` seconds later, in bps, signed so that positive means the price
moved in the direction of the trade. This uses only the market's own trades, so it can be computed for real data,
which has no controlled interventions, and for every simulator in the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.lobster import EMPTY, NS

HORIZONS = (1.0, 10.0, 60.0)
SIZE_BINS = (0, 100, 500, 2000, np.inf)  # shares


def responses(message: np.ndarray, book: np.ndarray, book_before: np.ndarray) -> pd.DataFrame:
    prev = np.vstack([book_before[None, :], book[:-1]])
    ok = (book[:, 0] < EMPTY) & (book[:, 2] > -EMPTY)
    mid = np.where(ok, 0.5 * (book[:, 0] + book[:, 2]), np.nan)
    mid = pd.Series(mid).ffill().to_numpy()
    t = message[:, 0]
    ex = np.nonzero(message[:, 1] == 4)[0]
    if len(ex) == 0:
        return pd.DataFrame(columns=["size", "sign"] + [f"resp_{h:g}s" for h in HORIZONS])
    key = t[ex] * 4 + (message[ex, 5] + 1)
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    rows = []
    for s, e in zip(starts, np.r_[starts[1:], len(ex)]):
        i = ex[s]
        size = float(message[ex[s:e], 3].sum())
        sign = -1.0 if message[i, 5] == 1 else 1.0  # resting buy executed = seller-initiated
        before = prev[i]
        if not (before[0] < EMPTY and before[2] > -EMPTY):
            continue
        m0 = 0.5 * (before[0] + before[2])
        row = {"size": size, "sign": sign}
        for h in HORIZONS:
            j = np.searchsorted(t, t[i] + int(h * NS), side="right") - 1
            row[f"resp_{h:g}s"] = np.nan if t[i] + int(h * NS) > t[-1] else sign * (mid[j] - m0) / m0 * 1e4
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["size_bin"] = pd.cut(df["size"], SIZE_BINS, labels=["<=100", "101-500", "501-2000", ">2000"])
    g = df.groupby("size_bin", observed=False)
    out = g[[f"resp_{h:g}s" for h in HORIZONS]].mean()
    out["n"] = g.size()
    return out
