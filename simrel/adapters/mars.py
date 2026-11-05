"""
MarS adapter: the common execution task inside a MarS-generated market.

Timeline of one seed (all in MarS's own exchange engine, mlib):

    09:30-10:00   noise agent builds an initial book (MarS's official demo set-up)
    10:00-10:20   MarS order model generates all flow (burn-in; the market is unstable
                  for roughly the first 15 minutes after hand-over)
    10:20-        task window, then a post window

At the task start the whole environment is deep-copied once per policy, so the
unperturbed twin and every policy continue from the identical market state and
random streams. A child order is sent as a marketable limit order priced at
the deepest visible level it needs and sized to the visible depth (ten
levels), so it fills immediately and never rests in the book.

MarS conventions are kept: prices in units of 1e-4 (tick = 100), lots of 100
shares. The released models were trained on Chinese equities.
"""

from __future__ import annotations

import copy
import logging
import random
import signal
import time
from typing import Optional

import numpy as np

from ..core import EpisodeRecord, Observation, Policy, Task, child_quantity
from .mars_local import DEFAULT_REPO, LocalModelClient

LOT, TICK, GRID_DT = 100, 100, 5.0
SYMBOL = "000000"
PREFIX_LIMIT_S, BRANCH_LIMIT_S = 420, 150  # wall-clock limits; a runaway rollout can otherwise generate orders for hours


class RolloutTimeout(BaseException):
    """Raised by the wall-clock limit (BaseException so nothing inside MarS can swallow it)."""


def _alarm(signum, frame):
    raise RolloutTimeout()


class time_limit:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return False


def run_twin_set(seed: int, task: Task, policies: dict[str, Optional[Policy]], repo_id: str = DEFAULT_REPO,
                 burn_in_s: int = 1200, pre_s: int = 300, world: str = "mars", sizes: Optional[dict[str, int]] = None,
                 init_observed=None) -> dict[str, EpisodeRecord]:
    """Requires setup_mars() to have been called in this process.

    `sizes` optionally gives a parent size per policy name (default task.total_qty), so several
    order sizes can branch from the same expensive burn-in.

    `init_observed` (a simrel.reference_worlds.data.Observed) replaces the noise-agent
    initialisation: the first 30 minutes replay that market's recorded order flow, so the MarS
    model starts from, and is conditioned on, another market's book and recent orders.
    """
    from pandas import Timedelta, Timestamp

    from market_simulation.agents.background_agent import BackgroundAgent
    from market_simulation.agents.noise_agent import NoiseAgent
    from market_simulation.conf import C
    from market_simulation.states.order_state import Converter, OrderState
    from market_simulation.states.trade_info_state import TradeInfoState
    from mlib.core.action import Action
    from mlib.core.base_agent import BaseAgent
    from mlib.core.env import Env
    from mlib.core.event import create_exchange_events
    from mlib.core.exchange import Exchange
    from mlib.core.exchange_config import create_exchange_config_without_call_auction
    from mlib.core.limit_order import LimitOrder
    from pathlib import Path

    logging.disable(logging.CRITICAL)
    random.seed(seed)
    np.random.seed(seed)

    start = Timestamp("2024-01-01 09:30:00")
    init_end = Timestamp("2024-01-01 10:00:00")
    t0 = init_end + Timedelta(seconds=burn_in_s)
    end = t0 + Timedelta(seconds=task.horizon_s + task.post_s)
    n = task.n_slices
    steps_per_slice = int(round(task.dt / GRID_DT))

    class ExecAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__(init_cash=0, communication_delay=0, computation_delay=0)
            self.policy: Optional[Policy] = None
            self.active = False  # False for the unperturbed twin
            self.grid: list[tuple[float, float, float]] = []
            self.step = 0  # grid steps since the task start
            self.k = 0
            self.total = task.total_qty
            self.remaining = task.total_qty
            self.state: dict = {}
            self.s_t, self.s_mid, self.s_spr, self.s_req, self.f_qty, self.f_not = (np.zeros(n) for _ in range(6))
            self.arrival = None
            self.prev_mid = None
            self.tape_i = 0
            self.own_last = 0.0
            self.fill_slice = 0
            self.last_quote = (np.nan, np.nan)
            self.vol_at_start = None
            self.vol_at_end = None

        def _trade_infos(self):
            return self.symbol_states[SYMBOL][TradeInfoState.__name__].trade_infos

        def _lob(self):
            ti = self._trade_infos()
            return ti[-1].lob_snapshot if ti else None

        @staticmethod
        def _traded(infos) -> tuple[float, float]:
            tot = sgn = 0.0
            for x in infos:
                if x.order.type in "BS" and x.transactions:
                    v = sum(tr.volume for tr in x.transactions if tr.type != "C")
                    tot += v
                    sgn += v if x.order.type == "B" else -v
            return tot, sgn

        def get_action(self, observation) -> Action:
            now = observation.time
            if now < t0 - Timedelta(seconds=pre_s):
                return Action(agent_id=self.agent_id, orders=[], time=now, next_wakeup_time=t0 - Timedelta(seconds=pre_s))
            lob = self._lob()
            if lob is not None and lob.bid_prices and lob.ask_prices:
                self.last_quote = (float(lob.bid_prices[0]), float(lob.ask_prices[0]))
            bid, ask = self.last_quote
            rel = (now - t0).total_seconds()
            self.grid.append((rel, bid, ask))
            orders = []
            if rel >= -1e-9:
                if self.step % steps_per_slice == 0 and self.k < n:
                    orders = self._decide(now, rel, lob, bid, ask)
                if self.step == n * steps_per_slice:
                    self.vol_at_end = self._traded(self._trade_infos())[0]
                self.step += 1
            nxt = now + Timedelta(seconds=GRID_DT)
            return Action(agent_id=self.agent_id, orders=orders, time=now, next_wakeup_time=nxt if nxt <= end else None)

        def _decide(self, now, rel, lob, bid, ask):
            infos = self._trade_infos()
            if self.k == 0:
                self.vol_at_start = self._traded(infos)[0]
            tot, sgn = self._traded(infos[self.tape_i:]) if self.k else (0.0, 0.0)
            self.tape_i = len(infos)
            mid = 0.5 * (bid + ask)
            self.arrival = mid if self.arrival is None else self.arrival
            near_p = (lob.bid_prices if task.side == "sell" else lob.ask_prices) if lob else []
            near_v = (lob.bid_volumes if task.side == "sell" else lob.ask_volumes) if lob else []
            obs = Observation(
                k=self.k, n=n, t=rel, bid=bid, ask=ask,
                bid_depth1=float(lob.bid_volumes[0]) if lob and lob.bid_volumes else 0.0,
                ask_depth1=float(lob.ask_volumes[0]) if lob and lob.ask_volumes else 0.0,
                bid_depth5=float(sum(lob.bid_volumes[:5])) if lob else 0.0, ask_depth5=float(sum(lob.ask_volumes[:5])) if lob else 0.0,
                market_volume_last=tot, signed_volume_last=sgn,
                mid_change_last=0.0 if self.prev_mid is None else mid - self.prev_mid,
                arrival_mid=self.arrival, total_qty=self.total, side=task.side,
            )
            qty = child_quantity(self.policy, obs, self.remaining, self.state, LOT) if self.active else 0
            qty = min(qty, int(sum(near_v)))  # never more than the visible book can absorb
            self.s_t[self.k], self.s_mid[self.k], self.s_spr[self.k], self.s_req[self.k] = rel, mid, ask - bid, qty
            self.fill_slice = self.k
            self.prev_mid = mid
            self.k += 1
            if qty <= 0:
                return []
            need, price = qty, near_p[0]
            for p, v in zip(near_p, near_v):
                price = p
                need -= v
                if need <= 0:
                    break
            return [LimitOrder(time=now, type="S" if task.side == "sell" else "B", price=int(price), volume=int(qty), symbol=SYMBOL,
                               agent_id=self.agent_id, order_id=-1, cancel_type="None", cancel_id=-1, tag="exec")]

        def on_order_executed(self, time_, transaction, trans_order_id_to_notify):
            out = super().on_order_executed(time_, transaction, trans_order_id_to_notify)
            if transaction.type != "C":
                v = transaction.volume if transaction.order_matched_volume is None else transaction.order_matched_volume[trans_order_id_to_notify]
                self.f_qty[self.fill_slice] += v
                self.f_not[self.fill_slice] += v * transaction.price
                self.remaining -= v
            return out

    cfg = create_exchange_config_without_call_auction(market_open=start, market_close=end, symbols=[SYMBOL])
    exchange = Exchange(cfg)
    converter = Converter(Path(C.directory.input_root_dir) / C.order_model.converter_dir)
    class ReplayAgent(BaseAgent):
        """Replays a recorded order stream (one tick = TICK price units) and owns the resulting book."""

        def __init__(self, obs) -> None:
            super().__init__(init_cash=0, communication_delay=0, computation_delay=0)
            self.ev = obs.events[obs.events[:, 0] < (init_end - start).total_seconds() - 1.0]
            self.i = -1  # -1: the opening book has not been posted yet
            self.open = (float(obs.bid[0]), float(obs.ask[0]), float(obs.bid_depth10[0]), float(obs.ask_depth10[0]))

        def _lob(self):
            ti = self.symbol_states[SYMBOL][TradeInfoState.__name__].trade_infos
            return ti[-1].lob_snapshot if ti else None

        def _order(self, now, kind, price, volume):
            return LimitOrder(time=now, type=kind, price=int(price), volume=int(volume), symbol=SYMBOL, agent_id=self.agent_id,
                              order_id=-1, cancel_type="None", cancel_id=-1, tag="replay")

        def get_action(self, observation) -> Action:
            now = observation.time
            if observation.is_market_open_wakup or now < start + Timedelta(seconds=1):
                return Action(agent_id=self.agent_id, orders=[], time=now, next_wakeup_time=start + Timedelta(seconds=1))
            orders = []
            if self.i < 0:  # opening book: ten levels a side, sized from the first depth observation
                bid, ask, db, da = self.open
                for k in range(10):
                    orders.append(self._order(now, "B", (bid - k) * TICK, max(int(db / 10) // LOT * LOT, LOT)))
                    orders.append(self._order(now, "S", (ask + k) * TICK, max(int(da / 10) // LOT * LOT, LOT)))
                self.i = 0
            else:
                t_ev, kind, size, _, price = self.ev[self.i][:5]
                kind, size = int(kind), max(int(size) // LOT * LOT, LOT)
                lob = self._lob()
                if kind in (0, 1):
                    orders.append(self._order(now, "B" if kind == 0 else "S", price * TICK, size))
                elif kind in (2, 3) and lob is not None:
                    ps, vs = (lob.ask_prices, lob.ask_volumes) if kind == 2 else (lob.bid_prices, lob.bid_volumes)
                    size = min(size, int(sum(vs)))
                    need, px = size, None
                    for p_, v_ in zip(ps, vs):
                        px = p_
                        need -= v_
                        if need <= 0:
                            break
                    if px is not None and size > 0:
                        orders.append(self._order(now, "B" if kind == 2 else "S", px, size))
                else:
                    orders = list(self.construct_valid_orders(time=now, symbol=SYMBOL, type="C", price=int(price * TICK), volume=size))
                self.i += 1
            nxt = None
            if 0 <= self.i < len(self.ev):
                nxt = max(start + Timedelta(seconds=float(self.ev[self.i][0])), now + Timedelta(microseconds=1))
            return Action(agent_id=self.agent_id, orders=orders, time=now, next_wakeup_time=nxt)

    if init_observed is not None:
        init_agent = ReplayAgent(init_observed)
    else:
        init_agent = NoiseAgent(symbol=SYMBOL, init_price=100000, interval_seconds=1, start_time=start, end_time=init_end, seed=seed)
    bg = BackgroundAgent(symbol=SYMBOL, converter=converter, start_time=init_end, end_time=end,
                         model_client=LocalModelClient(repo_id=repo_id, seed=seed), init_agent=init_agent)
    exchange.register_state(OrderState(num_max_orders=C.order_model.seq_len, num_bins_price_level=converter.price_level.num_bins,
                                       num_bins_pred_order_volume=converter.pred_order_volume.num_bins,
                                       num_bins_order_interval=converter.order_interval.num_bins, converter=converter))
    exchange.register_state(TradeInfoState())
    env = Env(exchange=exchange, description=f"mars seed {seed}")
    ex = ExecAgent()
    for a in (init_agent, bg, ex):
        env.register_agent(a)
    env.push_events(create_exchange_events(cfg))

    def drive(e, first_obs=None, stop_at_task_start=False):
        """Run an environment; optionally stop when the execution agent reaches the task start."""
        if first_obs is not None:
            e.step(first_obs.agent.get_action(first_obs))
        for obs in e.env():
            if stop_at_task_start and isinstance(obs.agent, ExecAgent) and obs.time >= t0:
                return obs
            e.step(obs.agent.get_action(obs))
        return None

    w_prefix = time.time()
    names = ["none", *policies]
    try:
        with time_limit(PREFIX_LIMIT_S):
            pending = drive(env, stop_at_task_start=True)
    except RolloutTimeout:
        return {k: EpisodeRecord(world, seed, k, task, failed=f"burn-in exceeded {PREFIX_LIMIT_S}s (runaway order generation)") for k in names}
    except Exception as err:  # e.g. MarS empties its own book during burn-in and its engine raises
        import traceback

        why = f"burn-in failed: {type(err).__name__}: {err} @ {traceback.extract_tb(err.__traceback__)[-1].name}"[:200]
        return {k: EpisodeRecord(world, seed, k, task, failed=why) for k in names}
    prefix_s = time.time() - w_prefix
    out: dict[str, EpisodeRecord] = {}
    if pending is None:
        return {k: EpisodeRecord(world, seed, k, task, failed="simulation ended before the task start") for k in names}
    np_state, py_state = np.random.get_state(), random.getstate()
    ex_id = ex.agent_id
    for name, policy in {"none": None, **policies}.items():
        w0 = time.time()
        total = (sizes or {}).get(name, task.total_qty) if name != "none" else 0
        btask = Task(side=task.side, total_qty=total, horizon_s=task.horizon_s, n_slices=task.n_slices, post_s=task.post_s)
        rec = EpisodeRecord(world, seed, name, btask)
        try:
            e2 = copy.deepcopy(env)
            np.random.set_state(np_state)
            random.setstate(py_state)
            a = e2.agents[ex_id]
            a.policy, a.active = policy, (name != "none" and total > 0)
            a.total = a.remaining = total
            obs2 = type(pending)(time=pending.time, agent=a, is_market_open_wakup=pending.is_market_open_wakup)
            with time_limit(BRANCH_LIMIT_S):
                drive(e2, first_obs=obs2)
            g = np.array(a.grid)
            rec.arrival_mid, rec.slice_t, rec.slice_mid, rec.slice_spread, rec.slice_req = a.arrival, a.s_t, a.s_mid, a.s_spr, a.s_req
            rec.fill_qty, rec.fill_notional = a.f_qty, a.f_not
            rec.grid_t, rec.grid_mid, rec.grid_spread = g[:, 0], 0.5 * (g[:, 1] + g[:, 2]), g[:, 2] - g[:, 1]
            traded = (a.vol_at_end if a.vol_at_end is not None else np.nan) - (a.vol_at_start or 0.0)
            rec.background_volume = float(traded - a.f_qty.sum())
            rec.extra = {"prefix_s": prefix_s, "model": repo_id, "model_calls": e2.agents[bg.agent_id].model_client.n_calls}
        except RolloutTimeout:
            rec.failed = f"rollout exceeded {BRANCH_LIMIT_S}s (runaway order generation)"
        except Exception as err:  # a MarS rollout can break (e.g. an emptied book); record it instead of crashing the batch
            import traceback

            rec.failed = f"{type(err).__name__}: {err} @ {traceback.extract_tb(err.__traceback__)[-1].name}"[:200]
        rec.wall_s = time.time() - w0
        out[name] = rec
    return out
