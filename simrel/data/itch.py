"""
Nasdaq TotalView-ITCH 5.0 -> LOBSTER-style order book data for chosen symbols.

Nasdaq publishes full-day ITCH sample files at https://emi.nasdaq.com/ITCH/.
A file is a stream of length-prefixed binary messages for every Nasdaq-listed
stock. This module scans the stream once, keeps only the requested symbols,
rebuilds each symbol's visible limit order book from the order-level messages,
and records, for every book event in regular trading hours, a LOBSTER-format
message row and a ten-level book snapshot taken after the event.

LOBSTER conventions used:
    message: time (seconds after midnight), type, order id, size, price, direction
        type 1 new limit order        4 execution of a visible order
             2 partial cancellation   5 execution of a hidden order (book unchanged)
             3 deletion
        direction +1 buy / -1 sell, always the side of the resting order
        price in units of 1e-4 dollars
    book: ask price 1, ask size 1, bid price 1, bid size 1, ... to level 10;
        empty levels are 9999999999 / -9999999999 with size 0
    An ITCH replace message becomes a deletion followed by a new order.
"""

from __future__ import annotations

import numpy as np
from sortedcontainers import SortedDict

OPEN_NS, CLOSE_NS = 34_200 * 10**9, 57_600 * 10**9
EMPTY_ASK, EMPTY_BID = 9_999_999_999, -9_999_999_999
LEVELS = 10

_A, _F, _E, _C, _X, _D, _U, _P, _R = (ord(c) for c in "AFECXDUPR")
_WANTED = frozenset((_A, _F, _E, _C, _X, _D, _U, _P))


class SymbolBook:
    """Visible limit order book of one symbol, rebuilt from order-level messages."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.orders: dict[int, list] = {}  # reference -> [is_buy, price, shares]
        self.bids: SortedDict = SortedDict()  # price -> shares
        self.asks: SortedDict = SortedDict()
        self.msg: list[tuple] = []
        self.book: list[tuple] = []
        self.n_crossed = 0
        self.n_unknown_ref = 0

    # ---- book mutations
    def _add(self, ref: int, is_buy: bool, price: int, shares: int) -> None:
        self.orders[ref] = [is_buy, price, shares]
        side = self.bids if is_buy else self.asks
        side[price] = side.get(price, 0) + shares

    def _reduce(self, ref: int, shares: int):
        """Remove shares from an order. Returns (is_buy, price, removed, emptied) or None for an unknown reference."""
        o = self.orders.get(ref)
        if o is None:
            self.n_unknown_ref += 1
            return None
        take = min(shares, o[2])
        o[2] -= take
        side = self.bids if o[0] else self.asks
        left = side[o[1]] - take
        if left > 0:
            side[o[1]] = left
        else:
            del side[o[1]]
        emptied = o[2] == 0
        if emptied:
            del self.orders[ref]
        return o[0], o[1], take, emptied

    def _snapshot(self) -> tuple:
        row = []
        ask_p, ask_q = self.asks.keys(), self.asks.values()
        bid_p, bid_q = self.bids.keys(), self.bids.values()
        na, nb = len(ask_p), len(bid_p)
        for k in range(LEVELS):
            if k < na:
                row += [ask_p[k], ask_q[k]]
            else:
                row += [EMPTY_ASK, 0]
            if k < nb:
                row += [bid_p[nb - 1 - k], bid_q[nb - 1 - k]]
            else:
                row += [EMPTY_BID, 0]
        if na and nb and ask_p[0] <= bid_p[nb - 1]:
            self.n_crossed += 1
        return tuple(row)

    def _emit(self, ts: int, kind: int, ref: int, size: int, price: int, is_buy: bool) -> None:
        if OPEN_NS <= ts <= CLOSE_NS:
            self.msg.append((ts, kind, ref, size, price, 1 if is_buy else -1))
            self.book.append(self._snapshot())

    # ---- one ITCH message (m is a memoryview of the message body, starting at the type byte)
    def handle(self, t: int, ts: int, m) -> None:
        ib = int.from_bytes
        if t == _A or t == _F:
            ref, is_buy, shares, price = ib(m[11:19], "big"), m[19] == 66, ib(m[20:24], "big"), ib(m[32:36], "big")
            self._add(ref, is_buy, price, shares)
            self._emit(ts, 1, ref, shares, price, is_buy)
        elif t == _E or t == _C:
            ref, shares = ib(m[11:19], "big"), ib(m[19:23], "big")
            r = self._reduce(ref, shares)
            if r:
                price = ib(m[32:36], "big") if t == _C else r[1]
                self._emit(ts, 4, ref, r[2], price, r[0])
        elif t == _X:
            ref, shares = ib(m[11:19], "big"), ib(m[19:23], "big")
            r = self._reduce(ref, shares)
            if r:
                self._emit(ts, 3 if r[3] else 2, ref, r[2], r[1], r[0])
        elif t == _D:
            ref = ib(m[11:19], "big")
            o = self.orders.get(ref)
            r = self._reduce(ref, o[2]) if o else self._reduce(ref, 0)
            if r:
                self._emit(ts, 3, ref, r[2], r[1], r[0])
        elif t == _U:
            old, new, shares, price = ib(m[11:19], "big"), ib(m[19:27], "big"), ib(m[27:31], "big"), ib(m[31:35], "big")
            o = self.orders.get(old)
            if o is None:
                self.n_unknown_ref += 1
                return
            is_buy = o[0]
            r = self._reduce(old, o[2])
            self._emit(ts, 3, old, r[2], r[1], is_buy)
            self._add(new, is_buy, price, shares)
            self._emit(ts, 1, new, shares, price, is_buy)
        elif t == _P:  # execution against a non-displayed order: the visible book does not change
            ref, is_buy, shares, price = ib(m[11:19], "big"), m[19] == 66, ib(m[20:24], "big"), ib(m[32:36], "big")
            self._emit(ts, 5, ref, shares, price, is_buy)

    # ---- simulator-facing API (times in ns after midnight; prices in LOBSTER units)
    def add(self, ts: int, ref: int, is_buy: bool, price: int, shares: int) -> None:
        self._add(ref, is_buy, price, shares)
        self._emit(ts, 1, ref, shares, price, is_buy)

    def execute(self, ts: int, ref: int, shares: int) -> None:
        r = self._reduce(ref, shares)
        if r:
            self._emit(ts, 4, ref, r[2], r[1], r[0])

    def cancel(self, ts: int, ref: int, shares: int | None = None) -> None:
        o = self.orders.get(ref)
        if o is None:
            self.n_unknown_ref += 1
            return
        r = self._reduce(ref, o[2] if shares is None else shares)
        self._emit(ts, 3 if r[3] else 2, ref, r[2], r[1], r[0])

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array(self.msg, dtype=np.int64).reshape(-1, 6), np.array(self.book, dtype=np.int64).reshape(-1, 4 * LEVELS))


def parse_stream(stream, symbols: list[str], chunk: int = 1 << 26, progress=None) -> dict[str, SymbolBook]:
    """Scan an uncompressed ITCH 5.0 byte stream and rebuild the books of `symbols`."""
    want = {s.ljust(8).encode(): s for s in symbols}
    books: dict[int, SymbolBook] = {}  # stock locate code -> book
    buf = b""
    n_msgs = 0
    while True:
        data = stream.read(chunk)
        if not data:
            break
        buf = buf + data if buf else data
        mv = memoryview(buf)
        i, n = 0, len(buf)
        while i + 2 <= n:
            ln = (buf[i] << 8) | buf[i + 1]
            j = i + 2
            if j + ln > n:
                break
            t = buf[j]
            n_msgs += 1
            if t in _WANTED:
                loc = (buf[j + 1] << 8) | buf[j + 2]
                b = books.get(loc)
                if b is not None:
                    b.handle(t, int.from_bytes(buf[j + 5:j + 11], "big"), mv[j:j + ln])
            elif t == _R:
                sym = buf[j + 11:j + 19]
                if sym in want:
                    books[(buf[j + 1] << 8) | buf[j + 2]] = SymbolBook(want[sym])
            i = j + ln
        mv.release()
        buf = buf[i:]
        if progress:
            progress(n_msgs)
    return {b.symbol: b for b in books.values()}
