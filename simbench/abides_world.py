"""
ABIDES market configurations and a common execution harness.

A "world" is an ABIDES background-agent configuration (see WORLDS). The same
ExecutionAgent, task definition, and accounting are used in every world, so
differences in outcomes come from the background population only.

Prices are in cents (ABIDES convention); reported costs are in basis points
of the arrival mid price.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import pomegranate_shim  # noqa: F401  (must precede abides_markets)

from abides_core import Kernel
from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.agents import (
    AdaptiveMarketMakerAgent,
    ExchangeAgent,
    MomentumAgent,
    NoiseAgent,
    TradingAgent,
    ValueAgent,
)
from abides_markets.messages.query import (
    QuerySpreadResponseMsg,
    QueryTransactedVolResponseMsg,
)
from abides_markets.models import OrderSizeModel
from abides_markets.oracles import SparseMeanRevertingOracle
from abides_markets.orders import Side
from abides_markets.utils import generate_latency_model

TICKER = "ABM"
DATE = "20210205"
NS = 1_000_000_000


# ---------------------------------------------------------------------
# World (background configuration) registry
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class WorldSpec:
    name: str
    description: str
    num_noise_agents: int = 1000
    num_value_agents: int = 102
    num_momentum_agents: int = 12
    num_mm: int = 2
    mm_pov: float = 0.025
    mm_wake_up_freq: str = "60S"
    fund_vol: float = 5e-5
    lambda_a: float = 5.7e-12
    r_bar: int = 100_000  # mean fundamental in cents; price-scaled parameters below follow it
    # optional population of quoting agents (simbench/quoting_agent.py); 0 = none, as in the original configurations
    num_quoting: int = 0
    quoting_rate: float = 0.0  # wake-ups per second per agent (one limit order each)
    quoting_lifetime_s: float = 1.0  # mean lifetime of a resting quote before the agent cancels it
    quoting_levels: tuple = (0,)  # ticks behind the best quote, sampled uniformly from this table
    quoting_sizes: tuple = (100,)  # shares, sampled uniformly from this table
    quoting_lifetimes: tuple = ()  # seconds; if given, each quote's lifetime is drawn from this table


WORLDS: dict[str, WorldSpec] = {
    w.name: w
    for w in [
        WorldSpec("A1_rmsc04", "RMSC04 reference: 1000 noise, 102 value, 12 momentum, 2 market makers"),
        WorldSpec("A2_no_mm", "RMSC04 without market makers", num_mm=0),
        WorldSpec("A3_momentum", "RMSC04 with 3x momentum traders", num_momentum_agents=36),
        WorldSpec("A4_few_value", "RMSC04 with 40 value traders (weaker anchoring, less flow)", num_value_agents=40),
        WorldSpec("A5_deep_mm", "RMSC04 with 4 market makers at 4x participation", num_mm=4, mm_pov=0.1),
        WorldSpec("A6_volatile", "RMSC04 with 4x fundamental volatility", fund_vol=2e-4),
    ]
}


# ---------------------------------------------------------------------
# Execution task and agent
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    side: str = "sell"  # "sell" or "buy"
    total_qty: int = 0  # shares; 0 = no intervention (paired baseline)
    start: str = "10:00:00"
    horizon_s: int = 1800
    n_slices: int = 60
    post_s: int = 300  # observation window after the deadline


@dataclass
class SliceState:
    k: int
    n: int
    remaining: int
    total: int
    spread: Optional[int]
    near_depth: int  # shares at the best level we would hit
    mkt_vol_last: int  # shares transacted in the market over the last slice
    mid: Optional[float]
    bids: Optional[list] = None  # visible book, best first, as (price, shares)
    asks: Optional[list] = None
    buy_vol_last: int = 0  # the two components of mkt_vol_last, as reported by the exchange
    sell_vol_last: int = 0


Policy = Callable[[SliceState], int]


class ExecutionAgent(TradingAgent):
    """Executes a parent order with market orders at fixed decision times.

    At each decision time it queries the book and recent volume, asks the policy
    for a child size, and sends one market order. The last slice always sends the
    full remainder so every policy completes the same parent order.
    """

    def __init__(self, id, symbol, starting_cash, task: Task, policy: Policy, mkt_open_ns: int, start_ns: int, random_state):
        super().__init__(id, name="EXECUTION_AGENT", type="ExecutionAgent", starting_cash=starting_cash, log_orders=False, random_state=random_state)
        self.symbol = symbol
        self.task = task
        self.policy = policy
        self.start_ns = start_ns
        self._offset = start_ns - mkt_open_ns
        self.dt_ns = int(task.horizon_s * NS / task.n_slices)
        self.k = 0
        self.filled = 0
        self.fills: list[tuple[int, int, int]] = []  # (time_ns, price, qty)
        self.slices: list[dict[str, Any]] = []
        self._pending: set[str] = set()

    def get_wake_frequency(self):
        return self._offset

    def wakeup(self, current_time):
        ready = super().wakeup(current_time)
        if not ready or current_time < self.start_ns or self.k >= self.task.n_slices:
            return
        self._pending = {"spread", "vol"}
        self.get_current_spread(self.symbol, depth=10)
        self.get_transacted_volume(self.symbol, lookback_period=f"{max(self.dt_ns // NS, 1)}s")

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)
        if not self._pending:
            return
        if isinstance(message, QuerySpreadResponseMsg):
            self._pending.discard("spread")
        elif isinstance(message, QueryTransactedVolResponseMsg):
            self._pending.discard("vol")
        else:
            return
        if not self._pending:
            self._act(current_time)

    def _act(self, current_time):
        sell = self.task.side == "sell"
        bids = self.known_bids.get(self.symbol) or []
        asks = self.known_asks.get(self.symbol) or []
        near = bids if sell else asks
        spread = asks[0][0] - bids[0][0] if bids and asks else None
        mid = 0.5 * (asks[0][0] + bids[0][0]) if bids and asks else None
        bv, sv = self.transacted_volume.get(self.symbol, (0, 0))
        vol = bv + sv
        remaining = self.task.total_qty - self.filled
        state = SliceState(self.k, self.task.n_slices, remaining, self.task.total_qty, spread, near[0][1] if near else 0, vol, mid,
                           bids=list(bids), asks=list(asks), buy_vol_last=bv, sell_vol_last=sv)

        if remaining <= 0:
            qty = 0
        elif self.k == self.task.n_slices - 1:
            qty = remaining
        else:
            qty = int(np.clip(round(self.policy(state)), 0, remaining))

        self.slices.append({"k": self.k, "t": current_time, "mid": mid, "spread": spread, "qty": qty, "remaining": remaining})
        if qty > 0:
            self.place_market_order(self.symbol, qty, Side.ASK if sell else Side.BID)

        self.k += 1
        if self.k < self.task.n_slices:
            self.set_wakeup(self.start_ns + self.k * self.dt_ns)

    def order_executed(self, order):
        super().order_executed(order)
        self.filled += order.quantity
        self.fills.append((self.current_time, order.fill_price, order.quantity))


# ---------------------------------------------------------------------
# Config builder (mirrors rmsc04's seeding order, plus the execution agent)
# ---------------------------------------------------------------------


def build_config(world: WorldSpec, seed: int, task: Task, policy: Policy) -> dict[str, Any]:
    np.random.seed(seed)
    starting_cash = 10_000_000_000
    r_bar = world.r_bar
    scale = r_bar / 100_000  # RMSC04's absolute price parameters are specified for r_bar = 100,000
    kappa = 1.67e-15

    date = int(pd.to_datetime(DATE).to_datetime64())
    mkt_open = date + str_to_ns("09:30:00")
    start_ns = date + str_to_ns(task.start)
    mkt_close = start_ns + (task.horizon_s + task.post_s) * NS
    noise_open = mkt_open - str_to_ns("00:30:00")
    noise_close = date + str_to_ns("16:00:00")

    def rs():
        return np.random.RandomState(seed=np.random.randint(low=0, high=2**32, dtype="uint64"))

    oracle = SparseMeanRevertingOracle(
        mkt_open,
        noise_close,
        {
            TICKER: {
                "r_bar": r_bar,
                "kappa": 1.67e-16,
                "sigma_s": 0,
                "fund_vol": world.fund_vol * scale,
                "megashock_lambda_a": 2.77778e-18,
                "megashock_mean": 1000 * scale,
                "megashock_var": 50_000 * scale * scale,
                "random_state": rs(),
            }
        },
    )
    size_model = OrderSizeModel()

    agents: list[Any] = [
        ExchangeAgent(
            id=0,
            name="EXCHANGE_AGENT",
            type="ExchangeAgent",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[TICKER],
            book_logging=True,
            book_log_depth=10,
            log_orders=None,
            pipeline_delay=0,
            computation_delay=0,
            stream_history=500,
            random_state=rs(),
        )
    ]
    n = 1
    for j in range(n, n + world.num_noise_agents):
        agents.append(
            NoiseAgent(id=j, name=f"NoiseAgent {j}", type="NoiseAgent", symbol=TICKER, starting_cash=starting_cash, wakeup_time=get_wake_time(noise_open, noise_close), log_orders=False, order_size_model=size_model, random_state=rs())
        )
    n += world.num_noise_agents
    for j in range(n, n + world.num_value_agents):
        agents.append(
            ValueAgent(id=j, name=f"Value Agent {j}", type="ValueAgent", symbol=TICKER, starting_cash=starting_cash, sigma_n=r_bar / 100, r_bar=r_bar, kappa=kappa, lambda_a=world.lambda_a, log_orders=False, order_size_model=size_model, random_state=rs())
        )
    n += world.num_value_agents
    for j in range(n, n + world.num_mm):
        agents.append(
            AdaptiveMarketMakerAgent(
                id=j,
                name=f"ADAPTIVE_POV_MARKET_MAKER_AGENT_{j}",
                type="AdaptivePOVMarketMakerAgent",
                symbol=TICKER,
                starting_cash=starting_cash,
                pov=world.mm_pov,
                min_order_size=1,
                window_size="adaptive",
                num_ticks=10,
                wake_up_freq=str_to_ns(world.mm_wake_up_freq),
                poisson_arrival=True,
                cancel_limit_delay=50,
                skew_beta=0,
                price_skew_param=4,
                level_spacing=5,
                spread_alpha=0.75,
                backstop_quantity=0,
                log_orders=False,
                random_state=rs(),
            )
        )
    n += world.num_mm
    for j in range(n, n + world.num_momentum_agents):
        agents.append(
            MomentumAgent(id=j, name=f"MOMENTUM_AGENT_{j}", type="MomentumAgent", symbol=TICKER, starting_cash=starting_cash, min_size=1, max_size=10, wake_up_freq=str_to_ns("37s"), poisson_arrival=True, log_orders=False, order_size_model=size_model, random_state=rs())
        )
    n += world.num_momentum_agents
    if world.num_quoting:
        from simbench.quoting_agent import QuotingAgent

        for j in range(n, n + world.num_quoting):
            agents.append(QuotingAgent(j, TICKER, starting_cash, world.quoting_rate, world.quoting_lifetime_s,
                                       world.quoting_levels, world.quoting_sizes, rs(), lifetime_table=world.quoting_lifetimes))
        n += world.num_quoting

    kernel_rs = rs()
    # The execution agent is always present (with total_qty=0 for baselines) and is
    # seeded last, so a baseline and an intervention run with the same seed share
    # every background random stream and the same latency matrix.
    agents.append(ExecutionAgent(n, TICKER, starting_cash, task, policy, mkt_open, start_ns, rs()))
    n += 1
    latency_model = generate_latency_model(n)

    return {
        "start_time": date,
        "stop_time": mkt_close + str_to_ns("1s"),
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": 50,
        "custom_properties": {"oracle": oracle},
        "random_state": kernel_rs,
        "skip_log": True,
        "_meta": {"start_ns": start_ns, "mkt_open": mkt_open, "mkt_close": mkt_close},
    }


# ---------------------------------------------------------------------
# Running one episode and extracting standardized outputs
# ---------------------------------------------------------------------


@dataclass
class Episode:
    world: str
    seed: int
    policy: str
    task: dict
    scalars: dict = field(default_factory=dict)
    grid_s: np.ndarray = None  # seconds relative to execution start
    mid: np.ndarray = None  # mid price on grid (cents)
    spread: np.ndarray = None
    bid_depth: np.ndarray = None  # top-5 depth
    ask_depth: np.ndarray = None
    trades: np.ndarray = None  # (t_s, signed volume) of all market trades
    fills: np.ndarray = None  # (t_s, price, qty) of our fills
    slices: list = None
    history: list = None  # the exchange's order-level history (only kept on request)
    start_ns: int = 0


def _l1_series(book_log2, start_ns, grid_s):
    t = np.array([r["QuoteTime"] for r in book_log2], dtype=np.int64)
    bb = np.array([r["bids"][0][0] if len(r["bids"]) else np.nan for r in book_log2], dtype=np.float64)
    ba = np.array([r["asks"][0][0] if len(r["asks"]) else np.nan for r in book_log2], dtype=np.float64)
    bd = np.array([r["bids"][:5, 1].sum() if len(r["bids"]) else 0 for r in book_log2], dtype=np.float64)
    ad = np.array([r["asks"][:5, 1].sum() if len(r["asks"]) else 0 for r in book_log2], dtype=np.float64)
    idx = np.searchsorted(t, start_ns + (grid_s * NS).astype(np.int64), side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    mid = pd.Series(0.5 * (bb + ba)).ffill().to_numpy()
    spr = pd.Series(ba - bb).ffill().to_numpy()
    return mid[idx], spr[idx], bd[idx], ad[idx]


def run_episode(world_name: str, seed: int, task: Task, policy: Policy, policy_name: str, warmup_s: int = 1200, keep_history: bool = False) -> Episode:
    world = WORLDS[world_name]
    cfg = build_config(world, seed, task, policy)
    meta = cfg.pop("_meta")
    # ABIDES always writes a small summary under ./log/<log_dir>; keep it unique per
    # episode (parallel workers) and out of the repository root.
    scratch = os.path.join(tempfile.gettempdir(), "simbench_abides_log")
    os.makedirs(scratch, exist_ok=True)
    os.chdir(scratch)
    kernel = Kernel(log_dir=f"{world_name}_{seed}_{policy_name}_{task.side}_{task.total_qty}_{os.getpid()}", **cfg)
    end_state = kernel.run()

    agents = end_state["agents"]
    book = agents[0].order_books[TICKER]
    ex: ExecutionAgent = agents[-1]
    start_ns = meta["start_ns"]

    grid_s = np.arange(-warmup_s, task.horizon_s + task.post_s + 1, 1.0)
    mid, spr, bd, ad = _l1_series(book.book_log2, start_ns, grid_s)

    # signed volume: + for the book's buy_transactions list, - for sell_transactions
    tr = [(t, v) for t, v in book.buy_transactions] + [(t, -v) for t, v in book.sell_transactions]
    trades = np.array(sorted(tr), dtype=np.float64).reshape(-1, 2)
    if len(trades):
        trades[:, 0] = (trades[:, 0] - start_ns) / NS
    fills = np.array(ex.fills, dtype=np.float64).reshape(-1, 3)
    if len(fills):
        fills[:, 0] = (fills[:, 0] - start_ns) / NS

    i0 = int(warmup_s)
    iT = i0 + task.horizon_s
    arrival = mid[i0]
    sgn = 1.0 if task.side == "sell" else -1.0
    in_win = (trades[:, 0] >= 0) & (trades[:, 0] <= task.horizon_s) if len(trades) else np.array([], dtype=bool)
    win_vol = float(np.abs(trades[in_win, 1]).sum()) if len(trades) else 0.0
    filled = float(fills[:, 2].sum()) if len(fills) else 0.0
    vwap = float((fills[:, 1] * fills[:, 2]).sum() / filled) if filled > 0 else np.nan

    r1 = np.diff(np.log(mid[i0 : iT + 1 : 60]))  # 1-minute log returns in the window
    sc = {
        "arrival_mid": arrival,
        "end_mid": mid[iT],
        "post_mid": mid[-1],
        "filled": filled,
        "fill_frac": filled / task.total_qty if task.total_qty else np.nan,
        "vwap": vwap,
        "is_bps": sgn * (arrival - vwap) / arrival * 1e4 if filled > 0 else np.nan,
        "mid_move_end_bps": (mid[iT] - arrival) / arrival * 1e4,
        "mid_move_post_bps": (mid[-1] - arrival) / arrival * 1e4,
        "win_volume": win_vol,
        "bg_volume": win_vol - filled,
        "mean_spread_bps": float(np.nanmean(spr[i0:iT])) / arrival * 1e4,
        "mean_depth5": float(np.mean(0.5 * (bd[i0:iT] + ad[i0:iT]))),
        "vol_1min_bps": float(np.std(r1)) * 1e4,
        "n_book_events": len(book.book_log2),
    }
    return Episode(world_name, seed, policy_name, asdict(task), sc, grid_s, mid, spr, bd, ad, trades, fills, ex.slices,
                   book.history if keep_history else None, start_ns)
