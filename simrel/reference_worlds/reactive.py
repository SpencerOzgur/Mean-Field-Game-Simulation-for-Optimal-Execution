"""
Reactive reference family (design addendum 11, development stage): a base world plus a predator.

The predator watches the public trade tape. When it sees a run of same-side trades that is regular in time and in
size (the footprint of a scheduled metaorder), it sends market orders on the same side just before each predicted
child, up to an inventory cap, and buys the position back once the pattern stops. It acts only when such a pattern is
present, so unperturbed episodes should contain almost none of its flow.

Structure

    TradeLog        the predator's view of the market: rows of (time, side, size) and nothing else. Times carry
                    a measurement error (ts_noise_s) standing in for latency and timestamp noise. Background trades
                    and injected (external) fills are written through the same call, Predator.observe(t, side, size),
                    so the detector cannot tell them apart.
    detect()        a pure function of the rows of a TradeLog.
    Predator        the state machine (idle, tracking, unwinding), its inventory and cash, and its own random stream.
    ReactiveWorld   a FlowModel that wraps an unmodified AgentMix or LatentLiquidity world. Predator orders hit the
                    shared book, go on the public tape and into the base world's own bookkeeping exactly as a
                    background market order does, so trend followers and liquidity withdrawal react to them.

With aggression 0 (or an inventory cap below one lot) the predator never trades and the wrapper never splits a call
to the base world's advance(), so every run is bit-identical to the base world. The detector still runs "in shadow",
which gives detection statistics without any market feedback.

Round 2 (simrel/reports/addendum11_round2.md) adds a "rational" variant, switched on by exit="passive":

    passive exit    the position is bought back with resting limit orders (owner tag PRED_TAG) at the best same-side
                    quote, re-posted when the quote moves away, and only what is left at a hard deadline is taken with
                    market orders. The orders rest from the moment of entry, so the trader's next child can fill them.
    profit gate     an entry is sent only if (estimated remaining children) x (estimated mid move per child) is at
                    least gate_mult x (half-spread + exit_allowance x spread). Both estimates use public data only:
                    the trade rows above and the best quotes (Predator.observe_quote). See Predator._gate.
    fills           passive fills are read from the book's own order-level record (Book.tape), which also tells a fill
                    from a background cancellation of the predator's order (the base worlds cancel resting shares
                    newest first, whoever posted them). The engine leaves that record off and never reads it; the
                    wrapper switches it on for an active predator.
    attribution     which passive fills were taken by the trader's own children is counted in ReactiveWorld.ledger for
                    reporting. The Predator object holds no reference to it and no decision reads it.

The aggressive predator (exit="aggressive", the pilot's) is unchanged, bit for bit.

Nothing in simrel/eventsim.py or in the base worlds is changed. run_twin_set deep-copies the model and returns only
episode records; a Recorder that survives deep copies collects the predator of every branch (see Recorder).
"""

from __future__ import annotations

import copy
from bisect import bisect_left
from dataclasses import dataclass, replace

import numpy as np

from ..eventsim import GRID_DT, FlowModel
from ..lob import Book
from . import latent, sampled
from .agent_mix import AgentMix
from .base import CB, CS, LB, LS, MB, MS, book_state

IDLE, TRACK, UNWIND = 0, 1, 2
PILOT_SEED = 20261120  # parameter stream of the pilot markets (XRP_pilot_*); used once, never reused
PILOT_PREFIX = "XRP_pilot_"
PREDATOR_STREAM = 11  # the predator's random stream is default_rng([seed, PREDATOR_STREAM, 0])
PRED_TAG = "predator"  # owner tag of the predator's resting orders in the book


@dataclass(frozen=True)
class PredatorParams:
    window_s: float = 240.0  # how far back the detector looks
    min_matches: int = 8  # trades in the regular run needed to trigger
    gap_cv_tol: float = 0.03  # largest coefficient of variation of the gaps between the run's trades; wide enough for
    # the timestamp noise, so the time test is weak and the size test carries the detection (addendum11_pilot.md)
    size_cv_tol: float = 0.15  # largest coefficient of variation of the run's trade sizes
    aggression: float = 0.0  # front-running order = aggression x estimated child size; 0 switches trading off
    lead_s: float = 2.0  # the front-running order is sent between lead_s / 2 and lead_s before the predicted child
    max_hold_s: float = 900.0  # a position is unwound this long after it was opened, whatever the pattern does
    unwind_halflife_s: float = 15.0  # half-life of the buy-back
    inventory_cap: int = 0  # largest absolute position, shares
    ts_noise_s: float = 0.2  # s.d. of the error added to every trade time the predator sees (latency, timestamp noise)
    # fixed, not part of the drawn family
    patience: float = 1.5  # unwinding starts when no matching trade arrives within this many predicted periods
    unwind_step_s: float = 5.0  # spacing of the buy-back orders
    flatten_after_s: float = 45.0  # whatever is left this long after unwinding began is bought back in one order
    min_period_s: float = 5.0  # shortest period the detector considers
    # round 2: the rational variant
    exit: str = "aggressive"  # "aggressive": buy back with market orders (pilot); "passive": with resting limit orders
    gate_mult: float = 0.0  # entry only if estimated gain per share >= gate_mult x estimated cost per share; 0: no gate
    exit_allowance: float = 0.5  # share of the spread added to the cost for having to exit with market orders
    exit_offset_ticks: int = 0  # resting exit orders are posted this many ticks behind the best same-side quote
    reprice_ticks: int = 1  # cancel and re-post when the target price is more than this many ticks from the order
    passive_exit_s: float = 60.0  # what is left this long after the pattern stopped is taken with market orders


@dataclass(frozen=True)
class Detection:
    side: int  # +1 buys, -1 sells
    period: float
    size: float
    last_t: float
    n: int


class TradeLog:
    """The public tape as the predator stores it: (time, side, size), nothing about who traded."""

    FIELDS = ("t", "side", "size")

    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self.rows: list = []

    def add(self, t: float, side: int, size: float) -> int:
        """Store one trade, keeping the rows in time order (noisy time stamps can arrive out of order); returns its position."""
        row = (float(t), int(side), float(size))
        j = len(self.rows)
        while j > 0 and self.rows[j - 1][0] > row[0]:
            j -= 1
        self.rows.insert(j, row)
        while self.rows[0][0] < self.rows[-1][0] - self.window_s:
            self.rows.pop(0)
            j -= 1
        return max(j, 0)


def _cv(x: list) -> float:
    m = sum(x) / len(x)
    if m <= 0:
        return float("inf")
    return (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5 / m


def detect(rows, p: PredatorParams) -> Detection | None:
    """Is the newest trade the end of a regular same-side run?

    rows: sequence of (time, side, size), oldest first. Looks for min_matches same-side trades ending at the newest one
    whose times form an arithmetic progression (other trades may lie in between), with the coefficient of variation of
    the gaps at most gap_cv_tol and that of the sizes at most size_cv_tol. The shortest qualifying period wins.
    """
    m = p.min_matches
    if len(rows) < m:
        return None
    tj, side, zj = rows[-1]
    ts, zs = [], []
    for t, s, z in rows:
        if s == side and t >= tj - p.window_s:
            ts.append(t)
            zs.append(z)
    k = len(ts)
    if k < m:
        return None
    for i in range(k - 2, -1, -1):
        g = tj - ts[i]
        if g < p.min_period_s:
            continue
        if g * (m - 1) > p.window_s:
            break
        tol = 2.0 * p.gap_cv_tol * g
        chain = [k - 1, i]
        while len(chain) < m:
            last = chain[-1]
            target = ts[last] - g
            best = None  # among the trades within the time tolerance: the one closest in size to the newest trade
            for c in range(bisect_left(ts, target - tol, 0, last), last):
                if ts[c] > target + tol:
                    break
                if best is None or (abs(zs[c] - zj), abs(ts[c] - target)) < (abs(zs[best] - zj), abs(ts[best] - target)):
                    best = c
            if best is None:
                break
            chain.append(best)
        if len(chain) < m:
            continue
        gaps = [ts[chain[a]] - ts[chain[a + 1]] for a in range(m - 1)]
        sizes = [zs[c] for c in chain]
        if _cv(gaps) > p.gap_cv_tol or _cv(sizes) > p.size_cv_tol:
            continue
        return Detection(side, sum(gaps) / len(gaps), sum(sizes) / len(sizes), tj, m)
    return None


class Recorder:
    """Collects the predator of every branch that run_twin_set creates.

    run_twin_set deep-copies the model once for the snapshot and once per run ("none", then the policies in order).
    A Recorder is not copied by deepcopy, and every copied Predator registers itself, so after run_twin_set
    branches[0] is the snapshot and branches[1:] are the runs in order. by_run() gives them by name.
    """

    def __init__(self) -> None:
        self.branches: list = []

    def __deepcopy__(self, memo):
        return self

    def by_run(self, policies) -> dict:
        names = ["none", *policies]
        assert len(self.branches) == len(names) + 1, "one snapshot copy and one copy per run expected"
        return dict(zip(names, self.branches[1:]))


class Predator:
    def __init__(self, params: PredatorParams, seed: int, recorder: Recorder | None = None, lot: int = 100) -> None:
        self.p = params
        self.lot = lot
        self.rng = np.random.default_rng([seed, PREDATOR_STREAM, 0])  # its own stream: never shared with the market
        self.recorder = recorder
        self.active = bool(params.aggression > 0 and params.inventory_cap >= lot)
        self.log = TradeLog(params.window_s)
        self.mode = IDLE
        self.inv = 0  # signed shares (negative: short)
        self.cash = 0.0  # in ticks x shares
        self.fills: list = []  # (time, signed shares, notional)
        self.triggers: list = []  # (time, side, period, estimated child size)
        self.tr_side, self.tr_period, self.tr_size, self.tr_last, self.t_open = 0, 0.0, 0.0, 0.0, 0.0
        self.front_t: float | None = None
        self.unwind_start, self.unwind_next, self.unwind_last = 0.0, 0.0, 0.0
        self.held_out = False  # True after a maximum-holding-time exit: no new trigger until flat
        self.history: list | None = None  # set to [] to keep every observed row (time, side, size) for offline replays
        # round 2
        self.passive = bool(self.active and params.exit == "passive")
        self.orders: dict = {}  # resting exit orders: order id -> [is_buy, price, shares left]
        self.quotes: list = []  # public best quotes (time, bid, ask), newest last
        self.legs: list = []  # (kind, signed shares, notional, mid when it traded); kinds: entry, unwind, passive, flatten
        self.run_n, self.run_start = 0, 0.0  # trades seen in the tracked run, and when it began
        self.gate_checks, self.gate_blocks = 0, 0
        self.maint_next = float("inf")

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, v if k == "recorder" else copy.deepcopy(v, memo))
        if self.recorder is not None:
            self.recorder.branches.append(new)
        return new

    # ---- the only way market information reaches the predator
    def observe(self, t: float, side: int, size: float) -> None:
        """One public trade: time, side (+1 buy, -1 sell), shares."""
        if self.history is not None:
            self.history.append((float(t), int(side), float(size)))
        if self.p.ts_noise_s > 0:  # the same error process for every trade, whoever sent it
            t = t + self.p.ts_noise_s * float(np.clip(self.rng.standard_normal(), -3.0, 3.0))
        self._timers(t)
        j = self.log.add(t, side, size)
        if self.mode == TRACK:
            if (side == self.tr_side and abs(t - (self.tr_last + self.tr_period)) <= 2.0 * self.p.gap_cv_tol * self.tr_period
                    and abs(size - self.tr_size) <= 2.0 * self.p.size_cv_tol * self.tr_size):
                self.tr_last = t
                self.tr_size += (size - self.tr_size) / self.p.min_matches
                self.run_n += 1
                self._schedule_front()
            return
        if self.mode == UNWIND and self.held_out:
            return
        hit = detect(self.log.rows[: j + 1], self.p)
        if hit is None or (self.mode == UNWIND and (self.inv > 0) != (hit.side > 0)):
            return
        if self.mode == IDLE:
            self.t_open = t
        self.mode = TRACK
        self.tr_side, self.tr_period, self.tr_size, self.tr_last = hit.side, hit.period, hit.size, t
        self.triggers.append((t, hit.side, hit.period, hit.size))
        self.run_n, self.run_start = hit.n, t - (hit.n - 1) * hit.period
        self._schedule_front()

    # ---- timers and orders
    def _schedule_front(self) -> None:
        if not self.active:
            return
        lead = min(self.p.lead_s, 0.5 * self.tr_period) * (0.5 + 0.5 * self.rng.random())
        self.front_t = self.tr_last + self.tr_period - lead

    def _deadline(self) -> float:
        return min(self.tr_last + self.p.patience * self.tr_period, self.t_open + self.p.max_hold_s)

    def _timers(self, t: float) -> None:
        if self.mode == TRACK and t >= self._deadline():
            dl = self._deadline()
            self.held_out = dl >= self.t_open + self.p.max_hold_s
            self.front_t = None
            if self.inv == 0:
                self.mode, self.held_out = IDLE, False
            else:
                self.mode = UNWIND
                self.unwind_start = self.unwind_next = dl
                self.unwind_last = dl - self.p.unwind_step_s

    def next_action_time(self) -> float:
        if not self.active:
            return float("inf")
        if self.passive:
            if self.mode == TRACK:
                ta = min(self._deadline(), self.front_t) if self.front_t is not None else self._deadline()
                return min(ta, self.maint_next) if (self.inv or self.orders) else ta
            return self.maint_next if self.mode == UNWIND else float("inf")
        if self.mode == TRACK:
            return min(self._deadline(), self.front_t) if self.front_t is not None else self._deadline()
        if self.mode == UNWIND:
            return self.unwind_next
        return float("inf")

    def act(self, now: float, send, ops=None) -> None:
        """Carry out whatever is due at `now`. send(is_buy, qty) -> (shares filled, notional)."""
        p = self.p
        if self.passive:
            return self._act_passive(now, send, ops)
        self._timers(now)
        if self.mode == TRACK and self.front_t is not None and now >= self.front_t:
            self.front_t = None
            room = p.inventory_cap - abs(self.inv)
            qty = min(max(int(round(p.aggression * self.tr_size / self.lot)), 1) * self.lot, room // self.lot * self.lot)
            if qty > 0:
                self._trade(self.tr_side > 0, qty, now, send, "entry")
        if self.mode == UNWIND and now >= self.unwind_next:
            left = abs(self.inv)
            if now >= self.unwind_start + p.flatten_after_s:
                qty = left
            else:
                frac = 1.0 - 0.5 ** ((now - self.unwind_last) / p.unwind_halflife_s)
                qty = min(max(int(round(left * frac / self.lot)), 1) * self.lot, left)
            self._trade(self.inv < 0, qty, now, send, "unwind")
            self.unwind_last, self.unwind_next = now, now + p.unwind_step_s
            if self.inv == 0:
                self.mode, self.held_out = IDLE, False

    def _trade(self, is_buy: bool, qty: int, now: float, send, kind: str = "entry") -> None:
        got, notional = send(is_buy, qty)
        if got:
            self.inv += got if is_buy else -got
            self.cash += -notional if is_buy else notional
            self.fills.append((now, got if is_buy else -got, notional))
            self.legs.append((kind, got if is_buy else -got, notional, self._mid()))

    # ---- round 2: public quotes, the profit gate and the passive exit
    def observe_quote(self, t: float, bid, ask) -> None:
        """The public best quotes at time t (the second and last kind of market information the predator gets)."""
        if bid is None or ask is None:
            return
        self.quotes.append((float(t), float(bid), float(ask)))
        while self.quotes[0][0] < t - 2.0 * self.p.window_s:
            self.quotes.pop(0)

    def _mid(self, t: float | None = None) -> float:
        """Mid of the newest quote (at or before t, if given); nan if no quote was seen."""
        if not self.quotes:
            return float("nan")
        q = self.quotes[-1]
        if t is not None:
            for q in reversed(self.quotes):
                if q[0] <= t:
                    break
        return 0.5 * (q[1] + q[2])

    def gate_estimate(self, now: float) -> tuple[float, float]:
        """(expected gain, cost) per share of one more entry, in ticks, from the trade rows and the quotes only.

        gain   remaining children x mid move per child. Remaining children are not observable; the run is assumed to
               last as long again as it has lasted (run_n more trades), but no longer than the maximum holding time
               allows. The mid move per child is the move of the mid in the run's direction since the run began,
               divided by the number of its trades after the first, and not below zero.
        cost   half the spread paid on entry, plus exit_allowance x spread for the part of the exit that will have to
               cross the spread.
        """
        if not self.quotes:
            return 0.0, float("inf")
        _, bid, ask = self.quotes[-1]
        spread = ask - bid
        left = max(min(float(self.run_n), (self.t_open + self.p.max_hold_s - now) / self.tr_period), 0.0)
        move = self.tr_side * (0.5 * (bid + ask) - self._mid(self.run_start)) / max(self.run_n - 1, 1)
        return left * max(move, 0.0), (0.5 + self.p.exit_allowance) * spread

    def _gate(self, now: float) -> bool:
        if self.p.gate_mult <= 0:
            return True
        self.gate_checks += 1
        gain, cost = self.gate_estimate(now)
        if gain >= self.p.gate_mult * cost:
            return True
        self.gate_blocks += 1
        return False

    def _act_passive(self, now: float, send, ops) -> None:
        """ops: quote() -> (best bid, best ask); place(is_buy, price, qty) -> order id; cancel(order id)."""
        p = self.p
        self._timers(now)
        if self.mode == UNWIND and self.inv == 0:
            self.mode, self.held_out = IDLE, False
        if self.mode == TRACK and self.front_t is not None and now >= self.front_t:
            self.front_t = None
            room = p.inventory_cap - abs(self.inv)
            qty = min(max(int(round(p.aggression * self.tr_size / self.lot)), 1) * self.lot, room // self.lot * self.lot)
            if qty > 0 and self._gate(now):
                self._cancel_all(ops)  # so that its own market order cannot be filled by its own resting order
                self._trade(self.tr_side > 0, qty, now, send, "entry")
        if self.mode == UNWIND and now >= self.unwind_start + p.passive_exit_s:
            self._cancel_all(ops)
            self._trade(self.inv < 0, abs(self.inv), now, send, "flatten")
            if self.inv == 0:
                self.mode, self.held_out = IDLE, False
        if self.inv != 0 and self.mode != IDLE and not (self.mode == UNWIND and now >= self.unwind_start + p.passive_exit_s):
            self._quote_exit(ops)
        elif self.orders:
            self._cancel_all(ops)
        self.maint_next = now + p.unwind_step_s
        if self.mode == UNWIND and now < self.unwind_start + p.passive_exit_s:
            self.maint_next = min(self.maint_next, self.unwind_start + p.passive_exit_s)

    def _cancel_all(self, ops) -> None:
        for oid in list(self.orders):
            ops.cancel(oid)

    def _quote_exit(self, ops) -> None:
        """Keep resting orders for the whole position at the best same-side quote (less exit_offset_ticks)."""
        is_buy = self.inv < 0
        bid, ask = ops.quote()
        same, opp = (bid, ask) if is_buy else (ask, bid)
        step = -1 if is_buy else 1
        if same is not None:
            target = same + step * self.p.exit_offset_ticks
        elif opp is not None:
            target = opp + step
        else:
            return
        if target <= 0:
            return
        for oid, (b, price, _) in list(self.orders.items()):
            if b != is_buy or abs(price - target) > self.p.reprice_ticks:
                ops.cancel(oid)
        need = abs(self.inv) - sum(o[2] for o in self.orders.values())
        if need > 0:
            oid = ops.place(is_buy, int(target), int(need))
            if oid is not None:
                self.orders[oid] = [is_buy, int(target), int(need)]

    def order_filled(self, t: float, oid: int, qty: int, price: int) -> None:
        """A resting order was (partly) filled; reported by the book's order-level record."""
        o = self.orders[oid]
        o[2] -= qty
        if o[2] <= 0:
            del self.orders[oid]
        signed = qty if o[0] else -qty
        self.inv += signed
        self.cash += -price * qty if o[0] else price * qty
        self.fills.append((t, signed, float(price * qty)))
        self.legs.append(("passive", signed, float(price * qty), self._mid()))

    def order_reduced(self, oid: int, qty: int) -> None:
        """Shares of a resting order were cancelled (by the predator itself or by the base world's cancellations)."""
        o = self.orders[oid]
        o[2] -= qty
        if o[2] <= 0:
            del self.orders[oid]

    # ---- reporting
    def summary(self, t0: float = 0.0, mark: float = float("nan")) -> dict:
        """t0: time subtracted from trigger times; mark: price at which any remaining inventory is valued."""
        vol = float(sum(abs(f[1]) for f in self.fills))
        pnl = self.cash + (self.inv * mark if self.inv else 0.0)
        return {"pred_triggers": len(self.triggers), "pred_first_trigger_s": self.triggers[0][0] - t0 if self.triggers else np.nan,
                "pred_period_s": self.triggers[0][2] if self.triggers else np.nan,
                "pred_trigger_side": self.triggers[0][1] if self.triggers else 0,
                "pred_volume": vol, "pred_orders": len(self.fills), "pred_inv_end": int(self.inv), "pred_pnl_ticks": float(pnl),
                "pred_max_inv": float(max((abs(x) for x in np.cumsum([f[1] for f in self.fills])), default=0.0)),
                **self.components(pnl)}

    def components(self, pnl: float) -> dict:
        """P&L split by kind of order, each measured against the mid when it traded (ticks x shares); the rest is the
        move of the mid while the position was held. Volumes by kind; gate counts."""
        out = {"pred_gate_checks": self.gate_checks, "pred_gate_blocks": self.gate_blocks}
        total = 0.0
        for kind in ("entry", "unwind", "passive", "flatten"):
            legs = [x for x in self.legs if x[0] == kind]
            edge = float(sum((x[3] * x[1] - x[2]) if x[1] > 0 else (x[2] + x[3] * x[1]) for x in legs))
            out[f"pred_{kind}_edge_ticks"] = edge
            out[f"pred_{kind}_volume"] = float(sum(abs(x[1]) for x in legs))
            total += edge
        out["pred_hold_pnl_ticks"] = float(pnl - total)
        return out


class ReactiveWorld(FlowModel):
    """A base world (AgentMix or LatentLiquidity, unmodified) with a predator trading in the same book."""

    name = "reactive"

    def __init__(self, base, predator: Predator, ledgers: Recorder | None = None) -> None:
        self.base = base
        self.pred = predator
        self.lot = base.lot
        self.p0 = base.p0
        # reporting only: who filled the predator's resting orders, and who cancelled them. Kept here, never on the
        # Predator, which has no reference to this object.
        self.ledger = {"trader": 0, "background": 0, "bg_cancel": 0, "own_cancel": 0}
        self.ledgers = ledgers
        self._own_tape, self._ti = False, 0

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, v if k == "ledgers" else copy.deepcopy(v, memo))
        if self.ledgers is not None:
            self.ledgers.branches.append(new.ledger)
        return new

    @property
    def log(self):
        return self.base.log

    @log.setter
    def log(self, value) -> None:
        self.base.log = value

    def init_book(self, book: Book) -> None:
        self.base.init_book(book)

    def _send(self, book: Book, tape: list, t: float):
        base = self.base

        def send(is_buy: bool, qty: int):
            # the same steps as a background market order of the base world (apply() of AgentMix / LatentLiquidity)
            base.tick(book, t)
            if base.log is not None:
                base._pre = book_state(book)
            book.now = t
            fills = book.market(is_buy, qty)
            got = sum(f[1] for f in fills)
            if got:
                tape.append((t, got if is_buy else -got))
                note = getattr(base, "_note_trade", None)  # AgentMix: the liquidity providers' signed-flow signal
                if note is not None:
                    note(got if is_buy else -got)
            base.record(t, MB if is_buy else MS, qty, 0)
            return got, float(sum(f[0] * f[1] for f in fills))

        return send

    def _ops(self, book: Book, t: float):
        """Limit-order actions of the passive predator, logged exactly as the base world logs its own."""
        world, base, pred = self, self.base, self.pred

        class Ops:
            @staticmethod
            def quote():
                return book.best_bid(), book.best_ask()

            @staticmethod
            def place(is_buy: bool, price: int, qty: int):
                same = book.best_bid() if is_buy else book.best_ask()
                opp = book.best_ask() if is_buy else book.best_bid()
                if opp is not None and ((is_buy and price >= opp) or (not is_buy and price <= opp)):
                    return None  # never marketable
                base.tick(book, t)
                if base.log is not None:
                    base._pre = book_state(book)
                book.now = t
                _, oid = book.limit(is_buy, price, qty, owner=PRED_TAG)
                level = (same - price if is_buy else price - same) if same is not None else 0
                base.record(t, LB if is_buy else LS, qty, int(level), price)
                return oid

            @staticmethod
            def cancel(oid: int) -> None:
                o = pred.orders.get(oid)
                if o is None:
                    return
                side = "bid" if o[0] else "ask"
                queue = (book.bids if o[0] else book.asks).get(o[1])
                k = next((i for i, x in enumerate(queue) if x[0] == oid), None) if queue else None
                if k is None:
                    return
                base.tick(book, t)
                if base.log is not None:
                    base._pre = book_state(book)
                book.now = t
                prices = book.prices(side, 10)
                level = prices.index(o[1]) if o[1] in prices else len(prices)
                n = book.cancel_order(side, o[1], (k + 0.5) / len(queue))
                base.record(t, CB if o[0] else CS, n, level, o[1])
                world._sync(book, "own_cancel")

        return Ops

    def _sync(self, book: Book, source: str) -> None:
        """Read the book's order-level record since the last call: fills and cancellations of the predator's orders.
        source: "trader" (called from on_external), "background" (after the base world ran) or "own_cancel"."""
        tp, pred = book.tape, self.pred
        for rec in tp[self._ti:]:
            if rec[2] in pred.orders:
                if rec[1] == "E":
                    pred.order_filled(rec[0], rec[2], rec[5], rec[4])
                    self.ledger[source if source != "own_cancel" else "background"] += rec[5]
                elif rec[1] == "X":
                    pred.order_reduced(rec[2], rec[5])
                    self.ledger["own_cancel" if source == "own_cancel" else "bg_cancel"] += rec[5]
        if self._own_tape:
            del tp[:]
        else:
            self._ti = len(tp)

    def _ingest(self, tape: list, n0: int) -> None:
        for t, signed in tape[n0:]:
            self.pred.observe(t, 1 if signed > 0 else -1, abs(signed))

    def advance(self, book: Book, t: float, t_until: float, tape: list) -> None:
        pred = self.pred
        if not pred.active:  # never split the call: bit-identical to the base world
            n0 = len(tape)
            self.base.advance(book, t, t_until, tape)
            self._ingest(tape, n0)
            return
        if pred.passive and book.tape is None:  # the engine leaves the order-level record off and never reads it
            book.tape, self._own_tape = [], True
        cur = t
        while cur < t_until:
            # a long call is cut at the engine's 5 s grid so the predator sees trades as promptly as inside run_twin_set
            stop = min(t_until, (np.floor(cur / GRID_DT + 1e-9) + 1.0) * GRID_DT)
            ta = pred.next_action_time()
            if ta < stop:
                stop = max(ta, cur)
            if stop > cur:
                n0 = len(tape)
                self.base.advance(book, cur, stop, tape)
                if pred.passive:
                    self._sync(book, "background")
                pred.observe_quote(stop, book.best_bid(), book.best_ask())
                self._ingest(tape, n0)
                cur = stop
            if cur < t_until and pred.next_action_time() <= cur:
                pred.act(cur, self._send(book, tape, cur), self._ops(book, cur) if pred.passive else None)
                if pred.passive:
                    self._sync(book, "background")

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        self.base.on_external(is_buy, qty, t, book)
        if self.pred.passive:  # nothing else trades between the trader's fill and this call
            self._sync(book, "trader")
        if self.pred.active:
            self.pred.observe_quote(t, book.best_bid(), book.best_ask())
        self.pred.observe(t, 1 if is_buy else -1, qty)


# ---- construction, strength presets and the pilot markets

# detector variants. "noisy" is the family's detector (the defaults above: noisy time stamps, size-led). "tight" is the
# first pilot version, kept as a comparison arm: exact time stamps and a very tight gap test, so that the engine's exact
# 30 s slice grid carries the detection. "loose" is a negative control: exact time stamps with loose tests.
DETECTORS = {"noisy": {}, "tight": {"ts_noise_s": 0.0, "gap_cv_tol": 0.0005, "size_cv_tol": 0.5, "min_matches": 5},
             "loose": {"ts_noise_s": 0.0, "gap_cv_tol": 0.01, "size_cv_tol": 0.5, "min_matches": 5}}

# strength -> (aggression, inventory cap as a share of the base market's ten-minute volume); detection is the same at
# every strength so that trigger rates are comparable
STRENGTHS = {"off": (0.0, 0.0), "weak": (0.5, 0.01), "medium": (1.0, 0.03), "strong": (2.0, 0.08)}


def predator_params(strength: str, reference_volume: float, detector: str = "noisy", lot: int = 100, **overrides) -> PredatorParams:
    aggression, cap_frac = STRENGTHS[strength]
    cap = int(cap_frac * reference_volume) // lot * lot
    return replace(PredatorParams(aggression=aggression, inventory_cap=cap), **{**DETECTORS[detector], **overrides})


def base_world(family: str, values: dict, seed: int):
    if family == "am":
        return AgentMix(sampled.params("am", values), seed)
    return latent.LatentLiquidity(latent.LatentParams(**values), seed)


def make_reactive(family: str, values: dict, pp: PredatorParams, recorder: Recorder | None = None, ledgers: Recorder | None = None):
    """Factory for run_twin_set: seed -> ReactiveWorld. recorder collects the predators, ledgers the attribution ledgers."""
    return lambda seed: ReactiveWorld(base_world(family, values, seed), Predator(pp, seed, recorder), ledgers)


# round 2 arms -> (strength, parameter changes). The aggressive arms are the pilot's medium and strong.
RATIONAL = {"exit": "passive", "gate_mult": 1.0}
ROUND2_ARMS = {"off": ("off", {}), "aggressive_medium": ("medium", {}), "aggressive_strong": ("strong", {}),
               "rational_medium": ("medium", RATIONAL), "rational_strong": ("strong", RATIONAL),
               "passive_ungated_strong": ("strong", {"exit": "passive", "gate_mult": 0.0})}


def round2_params(arm: str, reference_volume: float) -> PredatorParams:
    strength, changes = ROUND2_ARMS[arm]
    return predator_params(strength, reference_volume, **changes)


def pilot_markets(n_am: int = 3, n_latent: int = 3) -> dict:
    """name -> (family, base parameter values): agent-mix draws first, then latent, from default_rng(PILOT_SEED)."""
    rng = np.random.default_rng(PILOT_SEED)
    out = {}
    for k in range(1, n_am + 1):
        out[f"{PILOT_PREFIX}a{k:02d}"] = ("am", sampled.draw(rng, sampled.AM_RANGES))
    for k in range(1, n_latent + 1):
        out[f"{PILOT_PREFIX}l{k:02d}"] = ("latent", latent.draw(rng))
    return out
