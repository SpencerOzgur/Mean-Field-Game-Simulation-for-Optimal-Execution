"""
ScriptedTrader: an agent that executes a timed list of actions relative to the
current best bid/ask. Adversaries (spoofing, momentum ignition, predatory trading)
and exogenous order-flow events (flash-crash dumps) are all scripts.

A script is a list of (offset_s, action) where offset_s is seconds after the market
open and action is one of:
    {"type": "market", "side": "buy"|"sell", "qty": int}
    {"type": "limit",  "side": "buy"|"sell", "qty": int, "ticks": int}
        ticks is the distance from the near touch, measured away from the spread:
        0 joins the best bid (buy) / best ask (sell), 1 is one tick behind, etc.
    {"type": "cancel_all"}
    {"type": "market_to", "side": "buy"|"sell", "cum_qty": int}
        market order for whatever is needed to reach cum_qty shares filled on that side
        so far; re-sends quantity that a thin book failed to fill earlier.

With active=False the agent keeps the same wakeup schedule but does nothing, so a
baseline run has the same agents and the same event timing as the treated run.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from abides_markets.agents import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side

NS = 10**9


class ScriptedTrader(TradingAgent):
    def __init__(
        self,
        id: int,
        symbol: str,
        script: List[Tuple[float, Dict]],
        active: bool = True,
        name: Optional[str] = None,
        starting_cash: int = 10_000_000,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(id, name=name or f"Scripted_{id}", type=name or "ScriptedTrader",
                         random_state=random_state, starting_cash=starting_cash, log_orders=False)
        self.symbol = symbol
        self.active = active
        by_time = defaultdict(list)
        for offset_s, action in script:
            by_time[int(round(offset_s * NS))].append(action)
        self.times = sorted(by_time)
        self.actions = by_time
        self.k = 0
        self.state = "AWAITING_WAKEUP"
        self.fills: List[Tuple[int, int, int, str]] = []   # (time, qty, price, side)
        self.log_rows: List[Dict] = []                     # what it saw and did
        self.market_requested = 0                          # shares in plain market actions
        self.cum_target: Dict[str, int] = {}               # largest market_to target per side

    def kernel_starting(self, start_time):
        super().kernel_starting(start_time)
        self.set_wakeup(start_time)

    def get_wake_frequency(self):
        return self.times[0] if self.times else 0

    def wakeup(self, current_time):
        can_trade = super().wakeup(current_time)
        if not can_trade or not self.active or self.k >= len(self.times):
            return
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_SPREAD"

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)
        if self.state != "AWAITING_SPREAD" or not isinstance(message, QuerySpreadResponseMsg):
            return
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        for a in self.actions[self.times[self.k]]:
            self._do(a, bid, ask)
            self.log_rows.append({"t": current_time, "bid": bid, "ask": ask, **a})
        self.k += 1
        self.state = "AWAITING_WAKEUP"
        if self.k < len(self.times):
            self.set_wakeup(self.mkt_open + self.times[self.k])

    def _do(self, a, bid, ask):
        kind = a["type"]
        if kind == "cancel_all":
            self.cancel_all_orders()
            return
        side = Side.BID if a["side"] == "buy" else Side.ASK
        if kind == "market_to":
            done = sum(q for _, q, _, sd in self.fills if sd == a["side"])
            self.cum_target[a["side"]] = max(self.cum_target.get(a["side"], 0), int(a["cum_qty"]))
            qty = max(0, int(a["cum_qty"]) - done)
            if qty > 0:
                self.place_market_order(self.symbol, quantity=qty, side=side)
            return
        qty = int(a["qty"])
        if qty <= 0:
            return
        if kind == "market":
            self.market_requested += qty
            self.place_market_order(self.symbol, quantity=qty, side=side)
        elif kind == "limit":
            ref = bid if a["side"] == "buy" else ask
            if not ref:
                return
            ticks = int(a.get("ticks", 0))
            price = ref - ticks if a["side"] == "buy" else ref + ticks
            self.place_limit_order(self.symbol, quantity=qty, side=side, limit_price=price)

    def order_executed(self, order):
        super().order_executed(order)
        self.fills.append((self.current_time, order.quantity, order.fill_price,
                           "buy" if order.side.is_bid() else "sell"))

    # --- accounting ---------------------------------------------------------
    def position(self) -> int:
        return int(sum(q if s == "buy" else -q for _, q, _, s in self.fills))

    def market_fill_rate(self) -> float:
        """Filled / requested for market orders. ABIDES drops the unfilled remainder of a
        market order when it empties the opposite side of the book."""
        target = self.market_requested + sum(self.cum_target.values())
        if not target:
            return float("nan")
        return min(1.0, sum(q for _, q, _, _ in self.fills) / target)

    def pnl_cents(self, mark: float) -> float:
        """Trading PnL: cash flows from fills plus open position marked at `mark` (cents)."""
        cash = sum((-q * p) if s == "buy" else (q * p) for _, q, p, s in self.fills)
        return cash + self.position() * mark


# ---------------------------------------------------------------------------
# Script builders. Times are seconds after the open.
# ---------------------------------------------------------------------------
def spoof_script(start_s, duration_s, cycle_s=30, spoof_qty=5000, trade_qty=300, ticks=1):
    """
    Alternating spoof cycles. Cycle k (even): post a large fake bid `ticks` behind the
    best bid to suggest buying pressure, then half a cycle later market-SELL trade_qty
    and pull the fake bid. Odd cycles mirror it (fake ask, then BUY), so inventory
    stays near flat and profit only comes from moving the price.
    """
    script, t, k = [], start_s, 0
    while t + cycle_s <= start_s + duration_s:
        fake, real = ("buy", "sell") if k % 2 == 0 else ("sell", "buy")
        script.append((t, {"type": "limit", "side": fake, "qty": spoof_qty, "ticks": ticks}))
        script.append((t + cycle_s / 2, {"type": "market", "side": real, "qty": trade_qty}))
        script.append((t + cycle_s / 2, {"type": "cancel_all"}))
        t += cycle_s
        k += 1
    return script


def ignition_script(start_s, burst_qty=3000, burst_s=30, n_burst=6, hold_s=60, unwind_s=600, n_unwind=20):
    """Buy aggressively in a short burst, wait, then sell the position back slowly."""
    script = [(start_s + i * burst_s / n_burst, {"type": "market", "side": "buy", "qty": burst_qty // n_burst})
              for i in range(n_burst)]
    t0 = start_s + burst_s + hold_s
    qty = (burst_qty // n_burst) * n_burst
    for i in range(n_unwind):
        q = qty // n_unwind + (1 if i < qty % n_unwind else 0)
        script.append((t0 + i * unwind_s / n_unwind, {"type": "market", "side": "sell", "qty": q}))
    return script


def dump_script(start_s, qty, duration_s=120, n=12, side="sell", n_catchup=6):
    """
    Fire sale: aggressive one-sided selling of `qty` over `duration_s`. Uses cumulative
    targets, so shares a thin book could not absorb are re-sent at the next step; a few
    catch-up steps follow at the same cadence.
    """
    step = duration_s / n
    return [(start_s + i * step, {"type": "market_to", "side": side, "cum_qty": int(round(qty * (i + 1) / n))})
            for i in range(n)] + \
           [(start_s + (n + i) * step, {"type": "market_to", "side": side, "cum_qty": qty})
            for i in range(n_catchup)]


def predator_script(victim_start_s, victim_duration_s, qty, frontrun_s=120, n_front=6,
                    buyback_start_frac=0.8, buyback_s=None, n_back=10, victim_side="sell"):
    """
    Trade ahead of a known execution: sell `qty` right as the victim starts selling,
    then buy it back near the end of the victim's window, when the price is most
    depressed. Mirrored if the victim is buying.
    """
    first, second = ("sell", "buy") if victim_side == "sell" else ("buy", "sell")
    script = [(victim_start_s + i * frontrun_s / n_front, {"type": "market", "side": first, "qty": qty // n_front})
              for i in range(n_front)]
    total = (qty // n_front) * n_front
    b0 = victim_start_s + buyback_start_frac * victim_duration_s
    span = buyback_s if buyback_s is not None else victim_start_s + victim_duration_s - b0
    for i in range(n_back):
        q = total // n_back + (1 if i < total % n_back else 0)
        script.append((b0 + i * span / n_back, {"type": "market", "side": second, "qty": q}))
    return script
