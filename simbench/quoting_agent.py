"""
A quoting agent for ABIDES: places passive limit orders near the best quote and cancels them.

Each agent wakes at Poisson times, asks for the current spread and places one limit order on a random side, some
ticks behind the best quote on that side (never crossing the book), with the number of ticks and the size drawn from
tables. Each quote is cancelled when its lifetime, drawn from a table at placement, runs out (if it has not been
executed by then); with lifetime_table empty, each resting quote is instead cancelled at a wake-up with probability
1 - exp(-dt / lifetime_s). It never trades aggressively. A population of these supplies the quoting and
cancelling that ABIDES's default agents lack; rates, lifetimes and tables are set from data by the caller.
"""

from __future__ import annotations

import numpy as np
from abides_core import NanosecondTime
from abides_markets.agents import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import LimitOrder, Side

NS = 1_000_000_000


class QuotingAgent(TradingAgent):
    def __init__(self, id: int, symbol: str, starting_cash: int, rate_per_s: float, lifetime_s: float,
                 level_table: tuple, size_table: tuple, random_state: np.random.RandomState, name: str | None = None,
                 lifetime_table: tuple = ()) -> None:
        super().__init__(id, name=name or f"QUOTING_AGENT_{id}", type="QuotingAgent", starting_cash=starting_cash,
                         log_orders=False, random_state=random_state)
        self.symbol, self.rate, self.lifetime = symbol, rate_per_s, lifetime_s
        self.levels, self.sizes = np.asarray(level_table, dtype=np.int64), np.asarray(size_table, dtype=np.int64)
        self.last_wake: NanosecondTime | None = None
        self.awaiting = False
        self.scheduled: NanosecondTime | None = None  # the one wake-up this agent's own Poisson chain is waiting for
        # lifetime_table is either one table, or one table per placement level (ticks behind the best; the last
        # table is used for all deeper levels)
        tabs = lifetime_table if (len(lifetime_table) and isinstance(lifetime_table[0], (tuple, list))) else (lifetime_table,)
        self.lifetime_tables = [np.asarray(t, dtype=float) for t in tabs]
        self.lifetimes = self.lifetime_tables[0]
        self.expiry: dict = {}  # order id -> time at which the agent cancels it

    def _schedule(self, t: NanosecondTime) -> None:
        self.scheduled = t
        self.set_wakeup(t)

    def _next(self, now: NanosecondTime) -> NanosecondTime:
        return now + int(self.random_state.exponential(1.0 / self.rate) * NS) + 1

    def wakeup(self, current_time: NanosecondTime) -> None:
        # The parent class also schedules wake-ups (market-hours discovery); only the agent's own chain acts, so the
        # quoting rate is exactly rate_per_s.
        super().wakeup(current_time)
        if self.mkt_closed:
            return
        if not self.mkt_open or not self.mkt_close:
            return
        if self.scheduled is None:
            self._schedule(max(current_time, self.mkt_open) + int(self.random_state.exponential(1.0 / self.rate) * NS) + 1)
            return
        if self.expiry:  # cancel quotes whose lifetime has run out
            for oid, t_exp in list(self.expiry.items()):
                if t_exp <= current_time:
                    del self.expiry[oid]
                    order = self.orders.get(oid)
                    if isinstance(order, LimitOrder):
                        self.cancel_order(order)
        if current_time != self.scheduled:
            return
        if not len(self.lifetimes) and self.last_wake is not None and self.orders:
            p = 1.0 - np.exp(-(current_time - self.last_wake) / NS / self.lifetime)
            for order in list(self.orders.values()):
                if isinstance(order, LimitOrder) and self.random_state.rand() < p:
                    self.cancel_order(order)
        self.last_wake = current_time
        self.get_current_spread(self.symbol)
        self.awaiting = True
        self._schedule(self._next(current_time))

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message) -> None:
        super().receive_message(current_time, sender_id, message)
        if not (self.awaiting and isinstance(message, QuerySpreadResponseMsg)) or self.mkt_closed:
            return
        self.awaiting = False
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        if bid is None or ask is None:
            return
        is_bid = self.random_state.rand() < 0.5
        level = int(self.levels[self.random_state.randint(len(self.levels))])
        size = int(self.sizes[self.random_state.randint(len(self.sizes))])
        price = bid - level if is_bid else ask + level
        price = min(price, ask - 1) if is_bid else max(price, bid + 1)  # passive only
        if size > 0 and price > 0:
            self.place_limit_order(self.symbol, size, Side.BID if is_bid else Side.ASK, int(price))
            if len(self.lifetimes):
                oid = max(self.orders) if self.orders else None
                if oid is not None:
                    tab = self.lifetime_tables[min(max(int(bid - price) if is_bid else int(price - ask), 0), len(self.lifetime_tables) - 1)]
                    t_exp = current_time + max(int(tab[self.random_state.randint(len(tab))] * NS), 1)
                    self.expiry[oid] = t_exp
                    self.set_wakeup(t_exp)

    def get_wake_frequency(self) -> NanosecondTime:
        return int(self.random_state.uniform(0, 1.0 / self.rate) * NS)
