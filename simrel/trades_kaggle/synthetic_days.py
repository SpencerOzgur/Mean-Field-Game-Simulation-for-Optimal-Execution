"""
Write a synthetic reference market's observable episodes as LOBSTER days for TRADES (DeepMarket), for design addendum 11.

The episodes are the market's own fitting episodes (seeds SEED_FIT + 0..23, 30 minutes each; the first six are the
study's fitting data, all 24 are the recalibrated arm's data). They are written in blocks of six consecutive episodes
per pseudo-day, placed from 09:30 and stitched end to end; order ids are offset per episode so they stay unique.

    python simrel/trades_kaggle/synthetic_days.py R2a_am_anchored <out dir> [--episodes 24] [--per-day 6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATES = ["2030-01-02", "2030-01-03", "2030-01-04", "2030-01-07", "2030-01-08", "2030-01-09"]  # synthetic weekdays
OPEN_S = 9 * 3600 + 30 * 60
EP_S = 1800


def episode_arrays(world: str, seed: int, k: int):
    """One episode as visible ten-level LOBSTER arrays, with times shifted to its slot in the pseudo-day."""
    from simrel.data import lobster as L
    from simrel.experiments import goal2_run as G
    from simrel.lob import Book
    from simrel.reference_worlds.data import make

    model = make(world)(seed)
    book = Book()
    book.tape = []
    model.init_book(book)
    model.advance(book, 0.0, EP_S, [])
    w = L.tape_to_lobster(book.tape, 1.0, EP_S - 1.0, clock0_s=OPEN_S + k * EP_S, price_unit=100)
    w = G.R.S.visible_window(w)
    msg = w.message.copy()
    msg[:, 2] = msg[:, 2] + (k + 1) * 10_000_000  # unique order ids across stitched episodes
    return msg, w.book


def main() -> None:
    from simrel.configs import reference as C
    from simrel.data.lobster import write_lobster_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--per-day", type=int, default=6)
    ap.add_argument("--symbol", default="SYN")
    a = ap.parse_args()
    n_days = a.episodes // a.per_day
    out = Path(a.out) / a.symbol / f"{a.symbol}_{DATES[0]}_{DATES[n_days - 1]}"
    for d in range(n_days):
        parts = [episode_arrays(a.world, C.SEED_FIT + d * a.per_day + k, k) for k in range(a.per_day)]
        msg, book = np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])
        write_lobster_csv(out, a.symbol, DATES[d], msg, book)
        print(f"{DATES[d]}: {len(msg)} messages, {msg[0, 0] / 1e9 / 3600:.2f} h to {msg[-1, 0] / 1e9 / 3600:.2f} h", flush=True)
    print(out)


if __name__ == "__main__":
    main()
