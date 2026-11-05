"""
Scenario slices of real order data, and the pieces each simulator needs from a slice.

A slice is a set of ten-minute windows that share a label such as "high
volatility, low volume, high trend". Windows from the fit days are what a
simulator may learn from; windows from the test days are what its realism is
scored against.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .configs import slices as C
from .data import lobster as L


def visible_window(w: L.Window) -> L.Window:
    m, b = L.visible(w.message, w.book, w.book_before)
    return L.Window(w.symbol, w.date, w.start_s, w.length_s, m, b, w.book_before)


def load_windows(days: list | None = None) -> dict:
    """All ten-minute windows of the symbol, reduced to the ten visible levels, keyed by window key."""
    out = {}
    for date in days or (C.FIT_DAYS + C.TEST_DAYS):
        m, b = L.load_day(C.DATA_DIR, C.SYMBOL, date)
        for w in L.cut_windows(C.SYMBOL, date, m, b, C.WINDOW_S):
            v = visible_window(w)
            out[v.key] = v
    return out


def feature_table(windows: dict) -> pd.DataFrame:
    f = pd.DataFrame([L.features(w) for w in windows.values()])
    f["role"] = np.where(f["date"].isin(C.FIT_DAYS), "fit", "test")
    return f


def assign_slices(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label every window H or L on each slice feature (split at the median over all windows)."""
    f = f.copy()
    cuts = {k: float(f[k].median()) for k in C.SLICE_FEATURES}
    parts = [np.where(f[k] > cuts[k], short + "H", short + "L") for k, short in C.SLICE_FEATURES.items()]
    f["slice"] = ["_".join(x) for x in zip(*parts)]
    return f, cuts


def slice_names() -> list:
    return ["_".join(x) for x in itertools.product(*[[s + "H", s + "L"] for s in C.SLICE_FEATURES.values()])]


def book_orders(snapshot: np.ndarray, chunk: int = 100) -> tuple[tuple, int]:
    """Turn a ten-level snapshot into resting orders of at most `chunk` shares (prices in ticks) and the mid in ticks."""
    rows = []
    for lvl in range(10):
        for is_buy, pc in ((False, 4 * lvl), (True, 4 * lvl + 2)):
            price, qty = int(snapshot[pc]), int(snapshot[pc + 1])
            if abs(price) > L.EMPTY:
                continue
            while qty > 0:
                rows.append((is_buy, price // L.TICK, min(qty, chunk)))
                qty -= chunk
    return tuple(rows), int((snapshot[0] + snapshot[2]) // (2 * L.TICK))


def moments(feature_rows: list) -> dict:
    return {k: (float(np.mean([r[k] for r in feature_rows])), float(np.std([r[k] for r in feature_rows]))) for k in C.CALIB_MOMENTS}


def moment_error(sim_rows: list, target: dict) -> tuple[float, dict]:
    parts = {}
    for k, (mean, sd) in target.items():
        parts[k] = abs(float(np.mean([r[k] for r in sim_rows])) - mean) / max(sd, 0.25 * abs(mean), 1e-9)
    return float(np.mean(list(parts.values()))), parts


def out_dir() -> Path:
    p = Path(__file__).resolve().parents[1] / "results" / "simrel" / ("slices" if C.SYMBOL == "INTC" else f"slices_{C.SYMBOL}")
    (p / "raw").mkdir(parents=True, exist_ok=True)
    return p
