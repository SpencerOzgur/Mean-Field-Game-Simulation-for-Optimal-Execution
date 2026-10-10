"""
Execution agents for ABIDES.

ExecutionAgent handles everything that is the same for every execution strategy:
learning market hours, waking up on a fixed grid, requesting market data, recording
the arrival price and fills. A strategy implements `slice_qty` (market-order
strategies) or overrides `act` (strategies that manage limit orders).

All constructors share the harness signature
    (id, symbol, total_qty, side, start_delay, duration, n_slices, random_state, **kwargs)
so any registered strategy can be swept by harness.py and battery.py.
"""
from typing import List, Optional, Tuple

import numpy as np

from abides_markets.agents import TradingAgent
from abides_markets.messages.query import (
    QuerySpreadResponseMsg,
    QueryTransactedVolResponseMsg,
)
from abides_markets.orders import Side


class ExecutionAgent(TradingAgent):
    needs_volume = False  # set True to receive market volume over the last interval

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

        self.remaining = self.total_qty       # shares not yet sent as market orders
        self.slices_sent = 0
        self.arrival_mid: Optional[float] = None     # cents
        self.fills: List[Tuple[int, int, int]] = []  # (time ns, qty, price cents)
        self.state = "AWAITING_WAKEUP"
        self._pending = set()

    # --- helpers ------------------------------------------------------------
    @property
    def order_side(self) -> Side:
        return Side.ASK if self.side == "sell" else Side.BID

    @property
    def filled(self) -> int:
        return int(sum(q for _, q, _ in self.fills))

    def t_frac(self, current_time: int) -> float:
        """Fraction of the execution window elapsed (0 at first slice)."""
        return self.slices_sent / self.n_slices

    # --- strategy hooks -------------------------------------------------------
    def slice_qty(self, current_time: int, bid: Optional[int], ask: Optional[int]) -> int:
        """Shares to send as a market order this slice. Override in subclasses."""
        raise NotImplementedError

    def act(self, current_time: int, bid: Optional[int], ask: Optional[int]) -> None:
        """Default action: one market order of slice_qty shares."""
        qty = max(0, min(int(self.slice_qty(current_time, bid, ask)), self.remaining))
        if qty > 0:
            self.place_market_order(self.symbol, quantity=qty, side=self.order_side)
            self.remaining -= qty

    # --- ABIDES plumbing --------------------------------------------------------
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
        self._pending = {"spread"}
        self.get_current_spread(self.symbol)
        if self.needs_volume:
            self._pending.add("volume")
            self.get_transacted_volume(self.symbol, lookback_period=f"{self.interval // 10**9}s")
        self.state = "AWAITING_DATA"

    def receive_message(self, current_time, sender_id, message) -> None:
        super().receive_message(current_time, sender_id, message)
        if self.state != "AWAITING_DATA":
            return
        if isinstance(message, QuerySpreadResponseMsg):
            self._pending.discard("spread")
        elif isinstance(message, QueryTransactedVolResponseMsg):
            self._pending.discard("volume")
        if self._pending:
            return

        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        if self.arrival_mid is None and bid and ask:
            self.arrival_mid = (bid + ask) / 2

        if self.total_qty > 0:
            self.act(current_time, bid, ask)

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


class AlmgrenChrissAgent(ExecutionAgent):
    """
    Almgren-Chriss optimal trajectory x(t) = X sinh(k(T - t)) / sinh(kT).
    kappa_T = k*T controls urgency: ~0 is TWAP, larger values front-load.
    """

    def __init__(self, *args, kappa_T: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa_T = float(kappa_T)

    def target_remaining(self, frac: float) -> float:
        if self.kappa_T < 1e-6:
            return self.total_qty * (1 - frac)
        return self.total_qty * np.sinh(self.kappa_T * (1 - frac)) / np.sinh(self.kappa_T)

    def slice_qty(self, current_time, bid, ask) -> int:
        frac_next = (self.slices_sent + 1) / self.n_slices
        return int(round(self.remaining - self.target_remaining(frac_next)))


class POVAgent(ExecutionAgent):
    """
    Percentage of volume: each slice trades `pov` x market volume of the last interval.
    Does not force completion: any remainder at the end stays unfilled (report fill rate).
    """
    needs_volume = True

    def __init__(self, *args, pov: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.pov = float(pov)

    def slice_qty(self, current_time, bid, ask) -> int:
        bid_vol, ask_vol = self.transacted_volume.get(self.symbol, (0, 0))
        return int(round(self.pov * (bid_vol + ask_vol)))


class PassiveThenCrossAgent(ExecutionAgent):
    """
    Each slice: cancel working orders and rest the shares due so far at the near touch
    (best ask when selling, best bid when buying). On the final slice, cancel and cross
    the spread with a market order for whatever is still unfilled.
    """

    def act(self, current_time, bid, ask) -> None:
        self.cancel_all_orders()
        unfilled = self.total_qty - self.filled
        if unfilled <= 0:
            return
        last = self.slices_sent == self.n_slices - 1
        if last:
            # Cancels are in flight; a later fill of a cancelled order could overshoot by
            # at most one resting slice. Acceptable for a benchmark.
            self.place_market_order(self.symbol, quantity=unfilled, side=self.order_side)
            return
        due = int(round(self.total_qty * (self.slices_sent + 1) / self.n_slices)) - self.filled
        qty = max(0, min(due, unfilled))
        price = ask if self.side == "sell" else bid
        if qty > 0 and price:
            self.place_limit_order(self.symbol, quantity=qty, side=self.order_side, limit_price=price)
