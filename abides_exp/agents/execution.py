"""
Execution agents for ABIDES.

ExecutionAgent handles everything that is the same for every execution strategy:
learning market hours, waking up on a fixed grid, requesting the spread, recording
the arrival price and fills. A strategy only implements `slice_qty`.

To add a strategy (e.g. MFG on Sunday): subclass ExecutionAgent, override
`slice_qty`, and register the class in agents/__init__.py.
"""
from typing import List, Optional, Tuple

import numpy as np

from abides_markets.agents import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side


class ExecutionAgent(TradingAgent):
    def __init__(
        self,
        id: int,
        symbol: str,
        total_qty: int,
        side: str,                 # "sell" or "buy"
        start_delay: int,          # ns after market open
        duration: int,             # ns
        n_slices: int,
        starting_cash: int = 10_000_000,
        random_state: Optional[np.random.RandomState] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            id,
            name=name or f"{type(self).__name__}_{id}",
            type=type(self).__name__,
            random_state=random_state,
            starting_cash=starting_cash,
            log_orders=False,
        )
        if side not in ("sell", "buy"):
            raise ValueError("side must be 'sell' or 'buy'")
        self.symbol = symbol
        self.total_qty = int(total_qty)
        self.side = side
        self.start_delay = int(start_delay)
        self.duration = int(duration)
        self.n_slices = int(n_slices)
        self.interval = self.duration // self.n_slices

        self.remaining = self.total_qty
        self.slices_sent = 0
        self.arrival_mid: Optional[float] = None     # cents
        self.fills: List[Tuple[int, int, int]] = []  # (time ns, qty, price cents)
        self.state = "AWAITING_WAKEUP"

    # --- strategy hook ----------------------------------------------------
    def slice_qty(self, current_time: int, bid: Optional[int], ask: Optional[int]) -> int:
        """Shares to trade this slice. Override in subclasses."""
        raise NotImplementedError

    # --- ABIDES plumbing --------------------------------------------------
    def kernel_starting(self, start_time: int) -> None:
        super().kernel_starting(start_time)
        self.set_wakeup(start_time)  # first wakeup only learns market hours

    def get_wake_frequency(self) -> int:
        # TradingAgent schedules a wakeup at mkt_open + this once hours are known.
        return self.start_delay

    def wakeup(self, current_time: int) -> None:
        can_trade = super().wakeup(current_time)
        if not can_trade or self.slices_sent >= self.n_slices:
            return
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_SPREAD"

    def receive_message(self, current_time, sender_id, message) -> None:
        super().receive_message(current_time, sender_id, message)
        if self.state != "AWAITING_SPREAD" or not isinstance(message, QuerySpreadResponseMsg):
            return

        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        if self.arrival_mid is None and bid and ask:
            self.arrival_mid = (bid + ask) / 2

        qty = max(0, min(int(self.slice_qty(current_time, bid, ask)), self.remaining))
        if qty > 0:
            side = Side.ASK if self.side == "sell" else Side.BID
            self.place_market_order(self.symbol, quantity=qty, side=side)
            self.remaining -= qty

        self.slices_sent += 1
        self.state = "AWAITING_WAKEUP"
        if self.slices_sent < self.n_slices:
            self.set_wakeup(current_time + self.interval)

    def order_executed(self, order) -> None:
        super().order_executed(order)  # keeps holdings and cash correct
        self.fills.append((self.current_time, order.quantity, order.fill_price))


class TWAPAgent(ExecutionAgent):
    """Equal slices; any rounding remainder is spread over later slices."""

    def slice_qty(self, current_time, bid, ask) -> int:
        slices_left = self.n_slices - self.slices_sent
        return int(round(self.remaining / slices_left))
