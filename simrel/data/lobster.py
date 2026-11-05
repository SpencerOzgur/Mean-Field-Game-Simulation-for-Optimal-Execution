"""
Working with LOBSTER-style arrays produced by simrel.data.itch.

    message  N x 6   time (ns after midnight), type, order id, size, price, direction
    book     N x 40  ask price 1, ask size 1, bid price 1, bid size 1, ... (snapshot after the event)

Functions here cut days into windows, describe each window (volatility,
volume, trend), write LOBSTER CSV files for tools that expect them, and turn a
window into the event format the Hawkes fit consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NS = 10**9
EMPTY = 9_000_000_000
TICK = 100  # one cent in LOBSTER price units


@dataclass
class Window:
    symbol: str
    date: str
    start_s: int  # seconds after midnight
    length_s: int
    message: np.ndarray
    book: np.ndarray
    book_before: np.ndarray  # snapshot just before the first event of the window

    @property
    def key(self) -> str:
        return f"{self.symbol}_{self.date}_{self.start_s}"


def load_day(data_dir, symbol: str, date: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(Path(data_dir).expanduser() / f"{symbol}_{date}.npz")
    return d["message"], d["book"]


def cut_windows(symbol: str, date: str, message: np.ndarray, book: np.ndarray, length_s: int = 600,
                first_s: int = 9 * 3600 + 45 * 60, last_s: int = 15 * 3600 + 45 * 60) -> list[Window]:
    """Non-overlapping windows between first_s and last_s (the open and close auctions' surroundings are left out)."""
    t = message[:, 0]
    out = []
    for s in range(first_s, last_s, length_s):
        a, b = np.searchsorted(t, s * NS, side="left"), np.searchsorted(t, (s + length_s) * NS, side="left")
        if b - a < 100 or a == 0:
            continue
        out.append(Window(symbol, date, s, length_s, message[a:b], book[a:b], book[a - 1]))
    return out


def grid_quotes(w: Window, dt_s: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best bid and ask on a regular grid (prices in LOBSTER units); the quote in force at each grid time."""
    g = w.start_s * NS + (np.arange(0, w.length_s + 1e-9, dt_s) * NS).astype(np.int64)
    idx = np.searchsorted(w.message[:, 0], g, side="right") - 1
    ask = np.where(idx >= 0, w.book[np.clip(idx, 0, None), 0], w.book_before[0]).astype(float)
    bid = np.where(idx >= 0, w.book[np.clip(idx, 0, None), 2], w.book_before[2]).astype(float)
    return (g - w.start_s * NS) / NS, bid, ask


def features(w: Window) -> dict:
    """Descriptors used to define scenario slices. All are computed from the window itself."""
    _, bid, ask = grid_quotes(w)
    ok = (ask < EMPTY) & (bid > -EMPTY)
    mid = 0.5 * (bid + ask)[ok]
    r = np.diff(np.log(mid)) * 1e4
    ex = w.message[(w.message[:, 1] == 4) | (w.message[:, 1] == 5)]
    signed = -ex[:, 5] * ex[:, 3]  # the resting side is given; the aggressor is the opposite side
    vol = float(ex[:, 3].sum())
    return {
        "key": w.key, "symbol": w.symbol, "date": w.date, "start_s": w.start_s,
        "volatility_bps": float(r.std()),  # s.d. of 5-second mid returns
        "volume": vol,  # shares executed on Nasdaq in the window
        "trend_bps": float(abs(mid[-1] - mid[0]) / mid[0] * 1e4),  # absolute mid change over the window
        "spread_bps": float(np.mean((ask - bid)[ok] / (0.5 * (ask + bid)[ok])) * 1e4),
        "imbalance": float(signed.sum() / vol) if vol > 0 else 0.0,
        "volume_visible": float(w.message[w.message[:, 1] == 4, 3].sum()),
        "events": int(len(w.message)), "mid": float(mid[0] / 1e4),
        "depth10": float(0.5 * (w.book[:, 1::4].sum(1).mean() + w.book[:, 3::4].sum(1).mean())),
    }


def write_lobster_csv(out_dir, symbol: str, date: str, message: np.ndarray, book: np.ndarray, levels: int = 10) -> tuple[Path, Path]:
    """Write the two files of a LOBSTER day: <SYM>_<DATE>_34200000_57600000_{message,orderbook}_<levels>.csv."""
    import pandas as pd

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{date}_34200000_57600000"
    m = pd.DataFrame(message, columns=["time", "type", "id", "size", "price", "direction"])
    sec = m["time"] // NS
    frac = m["time"] % NS
    m["time"] = sec.astype(str) + "." + frac.astype(str).str.zfill(9)
    mp, bp = out / f"{stem}_message_{levels}.csv", out / f"{stem}_orderbook_{levels}.csv"
    m.to_csv(mp, header=False, index=False)
    pd.DataFrame(book[:, : 4 * levels]).to_csv(bp, header=False, index=False)
    return mp, bp


def visible(message: np.ndarray, book: np.ndarray, book_before: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the events a ten-level LOBSTER file would contain: those that change the visible book, and all executions.

    The ITCH feed (and a simulator's order tape) also carries orders placed and cancelled deeper than level ten.
    """
    prev = np.vstack([book_before[None, :], book[:-1]])
    keep = (book != prev).any(axis=1) | (message[:, 1] == 4) | (message[:, 1] == 5)
    return message[keep], book[keep]


def _cut(sb, t_a: int, t_b: int, symbol: str, date: str) -> Window:
    message, book = sb.arrays()
    a, b = np.searchsorted(message[:, 0], t_a, side="left"), np.searchsorted(message[:, 0], t_b, side="left")
    if a == 0:
        raise ValueError("window starts before the first recorded event")
    return Window(symbol, date, t_a // NS, (t_b - t_a) // NS, message[a:b], book[a:b], book[a - 1])


def tape_to_lobster(tape: list, start_s: float, length_s: float, clock0_s: int = 10 * 3600, price_unit: int = TICK,
                    symbol: str = "SIM", date: str = "sim") -> Window:
    """Rebuild LOBSTER arrays from a simulator's order tape and cut out [start_s, start_s + length_s).

    tape rows are (time in seconds, kind, order id, is_buy, price in ticks, shares) with kind "A" (order rests),
    "E" (resting order executed) or "X" (resting order cancelled), as written by simrel.lob.Book. Simulator time 0 is
    placed at clock0_s after midnight.
    """
    from .itch import SymbolBook

    sb = SymbolBook(symbol)
    for t, kind, oid, is_buy, price, qty in tape:
        ts = int(round((clock0_s + t) * NS))
        if kind == "A":
            sb.add(ts, oid, bool(is_buy), int(price) * price_unit, int(qty))
        elif kind == "E":
            sb.execute(ts, oid, int(qty))
        elif kind == "M":  # the order's quantity is now qty
            o = sb.orders.get(oid)
            if o is not None and o[2] > int(qty):
                sb.cancel(ts, oid, o[2] - int(qty))
        else:
            sb.cancel(ts, oid, int(qty))
    t_a = int((clock0_s + start_s) * NS)
    return _cut(sb, t_a, t_a + int(length_s * NS), symbol, date)


def abides_to_lobster(history: list, start_ns: int, start_s: float, length_s: float, price_unit: int = TICK,
                      symbol: str = "ABIDES", date: str = "sim") -> Window:
    """Rebuild LOBSTER arrays from an ABIDES exchange's order history and cut out [start_s, start_s + length_s).

    start_ns is the ABIDES timestamp that start_s = 0 refers to. ABIDES prices are in cents.
    """
    from .itch import SymbolBook

    day0 = start_ns - start_ns % (86400 * NS)
    sb = SymbolBook(symbol)
    for h in history:
        ts, kind = int(h["time"]) - day0, h["type"]
        if kind == "LIMIT":
            sb.add(ts, h["order_id"], h["side"] == "BID", int(h["price"]) * price_unit, int(h["quantity"]))
        elif kind == "EXEC":
            sb.execute(ts, h["order_id"], int(h["quantity"]))
        elif kind == "CANCEL":
            sb.cancel(ts, h["order_id"])
        elif kind == "CANCEL_PARTIAL":
            sb.cancel(ts, h["order_id"], int(h["quantity"]))
        elif kind == "MODIFY":
            o = sb.orders.get(h["order_id"])
            if o is not None and o[2] > int(h["new_quantity"]):
                sb.cancel(ts, h["order_id"], o[2] - int(h["new_quantity"]))
        elif kind == "REPLACE":
            o = sb.orders.get(h["old_order_id"])
            if o is not None:
                is_buy = o[0]
                sb.cancel(ts, h["old_order_id"])
                sb.add(ts, h["new_order_id"], is_buy, int(h["price"]) * price_unit, int(h["quantity"]))
    t_a = (start_ns - day0) + int(start_s * NS)
    return _cut(sb, t_a, t_a + int(length_s * NS), symbol, date)


def to_observed(w: Window, lot: int = 1):
    """Convert a window to the event log the Hawkes fit expects (simrel.reference_worlds.data.Observed).

    Limit orders keep their side; a cancellation or deletion becomes a cancel on its side; executions with the same
    timestamp against the same side are merged into one market order of the opposite side. Prices are in ticks.
    """
    from ..reference_worlds.data import Observed

    m, b = w.message, w.book
    prev = np.vstack([w.book_before[None, :], b[:-1]])  # book before each event
    ask0, bid0 = prev[:, 0].astype(float), prev[:, 2].astype(float)
    spread = np.where((ask0 < EMPTY) & (bid0 > -EMPTY), (ask0 - bid0) / TICK, np.nan)
    dep_a, dep_b = prev[:, 1::4].sum(1).astype(float), prev[:, 3::4].sum(1).astype(float)
    q1_a, q1_b = prev[:, 1].astype(float), prev[:, 3].astype(float)
    t = (m[:, 0] - w.start_s * NS) / NS
    rows = []
    i, n = 0, len(m)
    while i < n:
        typ, size, price, d = int(m[i, 1]), int(m[i, 3]), int(m[i, 4]), int(m[i, 5])
        if typ == 1:
            best = bid0[i] if d == 1 else ask0[i]
            level = int(round((best - price) / TICK)) if d == 1 else int(round((price - best) / TICK))
            if abs(best) > EMPTY:
                level = 0
            rows.append((t[i], 0 if d == 1 else 1, size, level, price / TICK, spread[i], dep_b[i], dep_a[i], q1_b[i], q1_a[i]))
            i += 1
        elif typ in (2, 3):
            best = bid0[i] if d == 1 else ask0[i]
            level = int(round((best - price) / TICK)) if d == 1 else int(round((price - best) / TICK))
            rows.append((t[i], 4 if d == 1 else 5, size, max(level, 0), price / TICK, spread[i], dep_b[i], dep_a[i], q1_b[i], q1_a[i]))
            i += 1
        elif typ == 4:  # executions against hidden orders (type 5) are left out: they do not touch the visible book
            j, tot = i, 0
            while j < n and m[j, 0] == m[i, 0] and m[j, 1] == 4 and m[j, 5] == d:
                tot += int(m[j, 3])
                j += 1
            rows.append((t[i], 3 if d == 1 else 2, tot, 0, price / TICK, spread[i], dep_b[i], dep_a[i], q1_b[i], q1_a[i]))  # resting buy hit = market sell
            i = j
        else:
            i += 1
    ev = np.array(rows, dtype=float).reshape(-1, 10)  # columns 8, 9: best bid and ask queue before the event
    ev[:, 2] = np.maximum(np.round(ev[:, 2] / lot), 1) * lot
    gt, bid, ask = grid_quotes(w)
    idx = np.clip(np.searchsorted(m[:, 0], w.start_s * NS + (gt * NS).astype(np.int64), side="right") - 1, 0, None)
    ex = m[m[:, 1] == 4]
    trades = np.column_stack([(ex[:, 0] - w.start_s * NS) / NS, -ex[:, 5] * ex[:, 3]]).astype(float)
    return Observed(seed=0, duration=float(w.length_s), events=ev, grid_t=gt[:-1], bid=bid[:-1] / TICK, ask=ask[:-1] / TICK,
                    bid_depth10=b[idx[:-1], 3::4].sum(1).astype(float), ask_depth10=b[idx[:-1], 1::4].sum(1).astype(float), trades=trades)
