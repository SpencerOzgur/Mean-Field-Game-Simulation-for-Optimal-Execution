"""
Observable data from a reference world, and the registry of worlds.

`observe()` runs unperturbed episodes and returns only what a market
participant could record: the event log (time, type, size, price level,
price) and a 5-second grid of quotes and depth. Candidate simulators are
fitted to this and never see the world's parameters or rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lob import Book
from . import agent_mix, queue_reactive


def registry() -> dict:
    out = {k: (queue_reactive.QueueReactive, v) for k, v in queue_reactive.regimes().items()}
    out.update({k: (agent_mix.AgentMix, v) for k, v in agent_mix.regimes().items()})
    from . import ablation, sampled  # randomly drawn markets (addendum 4); mechanism ablation (addendum 6)

    for k, (fam, params, _, _) in ablation.markets().items():
        out[k] = (queue_reactive.QueueReactive if fam == "qr" else agent_mix.AgentMix, params)

    for k, (fam, vals) in sampled.accepted().items():
        out[k] = (queue_reactive.QueueReactive if fam == "qr" else agent_mix.AgentMix, sampled.params(fam, vals))
    return out


# well-specified reference markets (addendum 5, experiment C): candidate Hawkes designs, as fitted to a training market
WELL_SPECIFIED = {
    "W1_hk1_R1a": ("R1a_qr_resilient", "h_k1"), "W2_hcfm_R1a": ("R1a_qr_resilient", "h_c_f_m_k1"),
    "W3_hk1_R1b": ("R1b_qr_fragile", "h_k1"), "W4_hcfm_R1b": ("R1b_qr_fragile", "h_c_f_m_k1"),
    "W5_hk1_R2a": ("R2a_am_anchored", "h_k1"), "W6_hcfm_R2a": ("R2a_am_anchored", "h_c_f_m_k1"),
}


def make(world: str):
    if world in WELL_SPECIFIED:
        from ..experiments import goal2_run as G

        src, design = WELL_SPECIFIED[world]
        factory = G.make_factory(src, design)

        def build(seed):
            m = factory(seed)
            m.p0 = m.p.p0  # observe() reads the starting price from the model
            return m

        return build
    cls, params = registry()[world]
    return lambda seed: cls(params, seed)


@dataclass
class Observed:
    seed: int
    duration: float
    events: np.ndarray  # columns: t, type (0 LB, 1 LS, 2 MB, 3 MS, 4 CB, 5 CS), size, level, price, spread, bid depth, ask depth (before the event)
    grid_t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_depth10: np.ndarray
    ask_depth10: np.ndarray
    trades: np.ndarray  # columns: t, signed shares


def observe(world: str, seed: int, duration: float = 1800.0, grid_dt: float = 5.0) -> Observed:
    model = make(world)(seed)
    model.log = []
    book, tape = Book(), []
    model.init_book(book)
    rows, t = [], 0.0
    last = (float(model.p0 - 1), float(model.p0 + 1))
    while t < duration - 1e-9:
        bb, ba = book.best_bid(), book.best_ask()
        last = (float(bb) if bb is not None else last[0], float(ba) if ba is not None else last[1])
        rows.append((t, last[0], last[1], book.depth("bid", 10), book.depth("ask", 10)))
        model.advance(book, t, t + grid_dt, tape)
        t += grid_dt
    g = np.array(rows, dtype=float)
    return Observed(seed, duration, np.array(model.log, dtype=float).reshape(-1, 8), g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4],
                    np.array(tape, dtype=float).reshape(-1, 2))
