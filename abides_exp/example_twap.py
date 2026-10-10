"""
ABIDES walkthrough: add a custom execution agent to the standard rmsc04 market,
run it against a baseline with the same seed, and measure what it did.

Run from the repo root:  python abides_exp/example_twap.py
"""
import logging

import numpy as np
import pandas as pd

from abides_core import abides
from abides_core.utils import str_to_ns, fmt_ts
from abides_markets.agents import TradingAgent
from abides_markets.configs import rmsc04
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.utils import config_add_agents


# ---------------------------------------------------------------------------
# 1. A custom agent. Every agent is an event handler:
#    wakeup()          -> the kernel woke you up; ask the exchange for data
#    receive_message() -> the exchange answered; decide and place orders
#    set_wakeup()      -> schedule your next wakeup
#    order_executed()  -> a fill came back
# ---------------------------------------------------------------------------
class TWAPSeller(TradingAgent):
    def __init__(self, id, symbol, total_qty, start_delay, duration, n_slices,
                 starting_cash=10_000_000, random_state=None):
        super().__init__(id, name="TWAP_SELLER", type="TWAPSeller",
                         random_state=random_state, starting_cash=starting_cash,
                         log_orders=True)
        self.symbol = symbol
        self.total_qty = total_qty
        self.start_delay = start_delay          # ns after the open
        self.interval = duration // n_slices    # ns between slices
        self.n_slices = n_slices
        self.slices_sent = 0
        self.remaining = total_qty
        self.arrival_mid = None                 # benchmark for shortfall
        self.fills = []                         # (time, qty, price in cents)
        self.state = "AWAITING_WAKEUP"

    def kernel_starting(self, start_time):
        super().kernel_starting(start_time)
        # First wakeup only learns market hours from the exchange.
        self.set_wakeup(start_time)

    def get_wake_frequency(self):
        # Once market hours are known, TradingAgent schedules a wakeup at mkt_open + this.
        return self.start_delay

    def wakeup(self, current_time):
        can_trade = super().wakeup(current_time)   # also requests mkt_open/close
        if not can_trade:
            return
        if self.slices_sent >= self.n_slices:
            return
        self.get_current_spread(self.symbol)       # async: answer arrives in receive_message
        self.state = "AWAITING_SPREAD"

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)
        if self.state != "AWAITING_SPREAD" or not isinstance(message, QuerySpreadResponseMsg):
            return
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        if self.arrival_mid is None and bid and ask:
            self.arrival_mid = (bid + ask) / 2

        slices_left = self.n_slices - self.slices_sent
        qty = int(round(self.remaining / slices_left))
        if qty > 0:
            self.place_market_order(self.symbol, quantity=qty, side=Side.ASK)
            self.remaining -= qty
        self.slices_sent += 1
        self.state = "AWAITING_WAKEUP"
        if self.slices_sent < self.n_slices:
            self.set_wakeup(current_time + self.interval)

    def order_executed(self, order):
        super().order_executed(order)              # keeps holdings/cash correct
        self.fills.append((self.current_time, order.quantity, order.fill_price))


# ---------------------------------------------------------------------------
# 2. Build a config, optionally add the agent, run.
# ---------------------------------------------------------------------------
SEED = 0
END = "11:00:00"
TICKER = "ABM"


def run(qty: int):
    """qty=0 gives the baseline. The seller is still added (and does nothing) so both
    runs have the same agent count, latency model and random draws."""
    cfg = rmsc04.build_config(seed=SEED, end_time=END, ticker=TICKER,
                              stdout_log_level="WARNING")
    seller = TWAPSeller(
        id=len(cfg["agents"]), symbol=TICKER, total_qty=qty,
        start_delay=str_to_ns("00:10:00"), duration=str_to_ns("00:30:00"),
        n_slices=30, random_state=np.random.RandomState(SEED + 1),
    )
    cfg = config_add_agents(cfg, [seller])   # also rebuilds the latency model
    end_state = abides.run(cfg)
    return end_state, seller


def mid_series(end_state) -> pd.Series:
    """L1 book history -> mid-price series (dollars) indexed by timestamp."""
    book = end_state["agents"][0].order_books[TICKER]
    l1 = book.get_L1_snapshots()
    t = l1["best_bids"][:, 0].astype("int64")
    bid = pd.to_numeric(pd.Series(l1["best_bids"][:, 1]), errors="coerce")
    ask = pd.to_numeric(pd.Series(l1["best_asks"][:, 1]), errors="coerce")
    mid = ((bid + ask) / 2 / 100).values
    return pd.Series(mid, index=pd.to_datetime(t)).dropna()


if __name__ == "__main__":
    logging.getLogger("abides").setLevel(logging.WARNING)

    base_state, _ = run(qty=0)
    exec_state, seller = run(qty=20_000)

    # 3. Results --------------------------------------------------------------
    if not seller.fills:
        raise SystemExit("seller got no fills")
    fills = pd.DataFrame(seller.fills, columns=["time", "qty", "price"])
    filled = fills.qty.sum()
    vwap = (fills.qty * fills.price).sum() / filled / 100
    arrival = seller.arrival_mid / 100
    shortfall_bps = (arrival - vwap) / arrival * 1e4

    print(f"\nTWAP seller: {filled:,} of {seller.total_qty:,} shares filled in {len(fills)} fills")
    print(f"arrival mid ${arrival:.2f}, VWAP ${vwap:.2f}, shortfall {shortfall_bps:.1f} bps")

    base_mid = mid_series(base_state).resample("1min").last()
    exec_mid = mid_series(exec_state).resample("1min").last()
    diff = (exec_mid - base_mid).dropna()
    print("\nmid-price, with seller minus baseline (same seed), $:")
    print(diff.iloc[::5].round(3).to_string())

    # Every agent's event log as one DataFrame (useful for debugging):
    #   from abides_core.utils import parse_logs_df
    #   logs = parse_logs_df(exec_state)
    #   logs[logs.agent_type == "TWAPSeller"].EventType.value_counts()