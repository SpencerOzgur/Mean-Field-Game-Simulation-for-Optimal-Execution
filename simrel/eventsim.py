"""
Episode runner for simulators built on simrel.lob.Book.

A flow model generates background events between decision times; the runner
samples the book on a fixed grid, calls the policy at each decision time, and
sends its child order to the same book. All policies for one seed branch from
one deep-copied market state at the task start, so they share an identical
history and an identical unperturbed twin.
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np

from .core import EpisodeRecord, Observation, Policy, Task, child_quantity
from .lob import Book

GRID_DT = 5.0


class FlowModel:
    """Interface for a background order-flow generator."""

    name = "flow"
    lot = 100

    def init_book(self, book: Book) -> None:
        raise NotImplementedError

    def advance(self, book: Book, t: float, t_until: float, tape: list) -> None:
        """Generate and apply all background events in [t, t_until). Trades go to tape as (t, signed_qty)."""
        raise NotImplementedError

    def on_external(self, is_buy: bool, qty: int, t: float, book: Book) -> None:
        """Called after an injected market order, so the flow can react to it."""


def _quote(book: Book, last: tuple[float, float]) -> tuple[float, float]:
    bb, ba = book.best_bid(), book.best_ask()
    return (float(bb) if bb is not None else last[0], float(ba) if ba is not None else last[1])


def run_twin_set(make_model, world: str, seed: int, task: Task, policies: dict[str, Optional[Policy]],
                 warmup_s: float = 1200.0, pre_s: float = 60.0) -> dict[str, EpisodeRecord]:
    """Run the unperturbed twin ("none") and every policy from one shared market state."""
    model = make_model(seed)
    book, tape = Book(), []
    model.init_book(book)
    t0 = float(warmup_s)
    model.advance(book, 0.0, t0 - pre_s, tape)
    pre_grid = []
    q = _quote(book, (np.nan, np.nan))
    t = t0 - pre_s
    while t < t0 - 1e-9:
        q = _quote(book, q)
        pre_grid.append((t - t0, q[0], q[1]))
        model.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    snapshot = copy.deepcopy((model, book, q))

    out = {}
    steps_per_slice = int(round(task.dt / GRID_DT))
    assert abs(steps_per_slice * GRID_DT - task.dt) < 1e-9, "slice length must be a multiple of the 5 s grid"
    for name, policy in {"none": None, **policies}.items():
        w0 = time.time()
        model, book, q = copy.deepcopy(snapshot)
        tape, grid = [], list(pre_grid)
        n = task.n_slices
        rec = EpisodeRecord(world, seed, name, task)
        s_t, s_mid, s_spr, s_req, f_qty, f_not = (np.zeros(n) for _ in range(6))
        remaining, state = task.total_qty, {}
        arrival, prev_mid, own_last, tape_i = None, None, 0.0, 0
        t = t0
        for k in range(n):
            q = _quote(book, q)
            mid = 0.5 * (q[0] + q[1])
            arrival = mid if arrival is None else arrival
            win = tape[tape_i:]
            tape_i = len(tape)
            obs = Observation(
                k=k, n=n, t=t - t0, bid=q[0], ask=q[1],
                bid_depth1=book.depth("bid", 1), ask_depth1=book.depth("ask", 1),
                bid_depth5=book.depth("bid", 5), ask_depth5=book.depth("ask", 5),
                market_volume_last=float(sum(abs(x[1]) for x in win)) + own_last,
                signed_volume_last=float(sum(x[1] for x in win)) + (-own_last if task.side == "sell" else own_last),
                mid_change_last=0.0 if prev_mid is None else mid - prev_mid,
                arrival_mid=arrival, total_qty=task.total_qty, side=task.side,
            )
            qty = child_quantity(policy, obs, remaining, state, model.lot) if task.total_qty and name != "none" else 0
            s_t[k], s_mid[k], s_spr[k], s_req[k] = t - t0, mid, q[1] - q[0], qty
            own_last = 0.0
            if qty > 0:
                book.now = t
                fills = book.market(task.side == "buy", qty)
                got = sum(f[1] for f in fills)
                f_qty[k], f_not[k] = got, sum(f[0] * f[1] for f in fills)
                remaining -= got
                own_last = float(got)
                model.on_external(task.side == "buy", got, t, book)
            prev_mid = mid
            for _ in range(steps_per_slice):
                q = _quote(book, q)
                grid.append((t - t0, q[0], q[1]))
                model.advance(book, t, t + GRID_DT, tape)
                t += GRID_DT
        in_window = float(sum(abs(x[1]) for x in tape))
        while t < t0 + task.horizon_s + task.post_s + 1e-9:
            q = _quote(book, q)
            grid.append((t - t0, q[0], q[1]))
            model.advance(book, t, t + GRID_DT, tape)
            t += GRID_DT
        g = np.array(grid)
        rec.arrival_mid, rec.slice_t, rec.slice_mid, rec.slice_spread, rec.slice_req = arrival, s_t, s_mid, s_spr, s_req
        rec.fill_qty, rec.fill_notional = f_qty, f_not
        rec.grid_t, rec.grid_mid, rec.grid_spread = g[:, 0], 0.5 * (g[:, 1] + g[:, 2]), g[:, 2] - g[:, 1]
        rec.background_volume = in_window
        rec.wall_s = time.time() - w0
        out[name] = rec
    return out
