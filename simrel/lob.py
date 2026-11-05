"""
A small price-time-priority limit order book.

Used by the Hawkes simulator and by the reference worlds. Prices are integer
ticks, sizes integer shares. Each price level is a FIFO queue. Marketable
orders execute at the resting orders' prices.
"""

from __future__ import annotations

from collections import deque


class Book:
    def __init__(self) -> None:
        self.bids: dict[int, deque] = {}  # price -> deque of [order_id, qty, owner]
        self.asks: dict[int, deque] = {}
        self._next_id = 0
        self.last_price: int | None = None
        self.tape: list | None = None  # optional order-level record: (time, kind, order id, is_buy, price, qty), kind in "AEX"
        self.now: float = 0.0  # set by the caller; stamped on tape rows

    # ---- queries
    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    def level_qty(self, side: str, price: int) -> int:
        q = (self.bids if side == "bid" else self.asks).get(price)
        return sum(o[1] for o in q) if q else 0

    def depth(self, side: str, levels: int) -> int:
        book = self.bids if side == "bid" else self.asks
        prices = sorted(book, reverse=(side == "bid"))[:levels]
        return sum(o[1] for p in prices for o in book[p])

    def prices(self, side: str, levels: int) -> list[int]:
        book = self.bids if side == "bid" else self.asks
        return sorted(book, reverse=(side == "bid"))[:levels]

    # ---- updates
    def _match(self, taker_is_buy: bool, qty: int, limit: int | None) -> list[tuple[int, int, object]]:
        """Consume resting liquidity; returns fills as (price, qty, passive_owner)."""
        book = self.asks if taker_is_buy else self.bids
        fills = []
        while qty > 0 and book:
            p = min(book) if taker_is_buy else max(book)
            if limit is not None and ((taker_is_buy and p > limit) or (not taker_is_buy and p < limit)):
                break
            queue = book[p]
            while qty > 0 and queue:
                o = queue[0]
                take = min(qty, o[1])
                o[1] -= take
                qty -= take
                fills.append((p, take, o[2]))
                if self.tape is not None:
                    self.tape.append((self.now, "E", o[0], not taker_is_buy, p, take))
                if o[1] == 0:
                    queue.popleft()
            if not queue:
                del book[p]
        if fills:
            self.last_price = fills[-1][0]
        return fills

    def market(self, is_buy: bool, qty: int) -> list[tuple[int, int, object]]:
        """Market order: fills what the book holds; any remainder is dropped."""
        return self._match(is_buy, qty, None)

    def limit(self, is_buy: bool, price: int, qty: int, owner: object = None) -> tuple[list[tuple[int, int, object]], int | None]:
        """Limit order: matches if marketable, then rests. Returns (fills, resting order id or None)."""
        fills = self._match(is_buy, qty, price)
        rest = qty - sum(f[1] for f in fills)
        oid = None
        if rest > 0:
            oid = self._next_id
            self._next_id += 1
            (self.bids if is_buy else self.asks).setdefault(price, deque()).append([oid, rest, owner])
            if self.tape is not None:
                self.tape.append((self.now, "A", oid, is_buy, price, rest))
        return fills, oid

    def cancel(self, side: str, price: int, qty: int) -> int:
        """Cancel up to qty shares at a price, newest orders first. Returns shares cancelled."""
        book = self.bids if side == "bid" else self.asks
        queue = book.get(price)
        done = 0
        while queue and done < qty:
            o = queue[-1]
            take = min(qty - done, o[1])
            o[1] -= take
            done += take
            if self.tape is not None:
                self.tape.append((self.now, "X", o[0], side == "bid", price, take))
            if o[1] == 0:
                queue.pop()
        if queue is not None and not queue:
            del book[price]
        return done

    def cancel_order(self, side: str, price: int, u: float) -> int:
        """Cancel one whole resting order at a price, chosen by a uniform u in [0, 1). Returns shares cancelled."""
        book = self.bids if side == "bid" else self.asks
        queue = book.get(price)
        if not queue:
            return 0
        k = min(int(u * len(queue)), len(queue) - 1)
        o = queue[k]
        del queue[k]
        if self.tape is not None:
            self.tape.append((self.now, "X", o[0], side == "bid", price, o[1]))
        if not queue:
            del book[price]
        return o[1]
