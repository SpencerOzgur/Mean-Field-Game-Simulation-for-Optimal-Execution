"""
Multi-seed experiment harness for ABIDES execution studies.

For every seed it runs one baseline (the execution agent present with qty=0, so the
agent count, latency model and random draws match) and one run per order size.
Impact is measured as the price difference between each run and its own baseline.

Two phases:
  1. Baselines for all seeds. Gives the market volume in the execution window,
     which turns --pct into share quantities.
  2. Execution runs for every (seed, size), in parallel.

Usage (from the repo root, inside .venv-abides):
  python -m abides_exp.harness --strategy twap --seeds 30 \
      --pct 0.01 0.05 0.10 0.20 --workers 8 --out results/impact_sweep

Outputs in --out:
  meta.json        all settings + calibrated volume, so a run can be reproduced
  baselines.csv    one row per seed (volume, prices)
  runs.csv         one row per (seed, size): fills, VWAP, shortfall, impact summary
  paths.csv        long format (seed, qty, t_rel_s, impact_bps): the impact curves

Sign convention: impact_bps and shortfall_bps are positive when the price moves
AGAINST the trader (down for a seller, up for a buyer).
Size convention: pct = qty / mean baseline market volume in the execution window.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from abides_core import abides
from abides_core.utils import str_to_ns
from abides_markets.configs import rmsc04
from abides_markets.utils import config_add_agents

from abides_exp.agents import STRATEGIES

TICKER = "ABM"


@dataclass
class Schedule:
    end_time: str = "11:00:00"     # simulation end (market opens 09:30)
    start: str = "00:10:00"        # execution starts this long after the open
    duration: str = "00:30:00"
    n_slices: int = 30
    side: str = "sell"
    grid: str = "30s"              # sampling grid for price paths


@dataclass
class Experiment:
    strategy: str = "twap"
    seeds: List[int] = field(default_factory=lambda: list(range(30)))
    pct: Optional[List[float]] = None    # sizes as fraction of window volume
    qty: Optional[List[int]] = None      # or absolute share sizes
    schedule: Schedule = field(default_factory=Schedule)
    market_kwargs: Dict = field(default_factory=dict)  # passed to rmsc04.build_config


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------
def _simulate(seed: int, strategy: str, qty: int, sch: Schedule, market_kwargs: Dict):
    cfg = rmsc04.build_config(
        seed=seed,
        end_time=sch.end_time,
        ticker=TICKER,
        stdout_log_level="WARNING",
        log_orders=False,          # big speed and memory win; we track fills ourselves
        **market_kwargs,
    )
    agent = STRATEGIES[strategy](
        id=len(cfg["agents"]),
        symbol=TICKER,
        total_qty=qty,
        side=sch.side,
        start_delay=str_to_ns(sch.start),
        duration=str_to_ns(sch.duration),
        n_slices=sch.n_slices,
        random_state=np.random.RandomState(seed + 1),
    )
    cfg = config_add_agents(cfg, [agent])  # rebuilds the latency model
    end_state = abides.run(cfg)
    book = end_state["agents"][0].order_books[TICKER]
    mkt_open = end_state["agents"][0].mkt_open
    return book, agent, mkt_open


def _mid_on_grid(book, mkt_open: int, sch: Schedule) -> pd.Series:
    """Mid price (cents) on a regular grid from the open, indexed by seconds since open."""
    l1 = book.get_L1_snapshots()
    t = l1["best_bids"][:, 0].astype("int64")
    bid = pd.to_numeric(pd.Series(l1["best_bids"][:, 1]), errors="coerce").values
    ask = pd.to_numeric(pd.Series(l1["best_asks"][:, 1]), errors="coerce").values
    mid = pd.Series((bid + ask) / 2, index=pd.to_datetime(t)).dropna()
    mid = mid[~mid.index.duplicated(keep="last")]
    grid = pd.date_range(
        pd.to_datetime(mkt_open),
        pd.to_datetime(mkt_open) + pd.Timedelta(str_to_ns(sch.end_time) - str_to_ns("09:30:00")),
        freq=sch.grid,
    )
    on_grid = mid.reindex(mid.index.union(grid)).ffill().reindex(grid)
    on_grid.index = ((on_grid.index - grid[0]).total_seconds()).astype(int)
    return on_grid


def _window_volume(book, mkt_open: int, sch: Schedule) -> int:
    lo = mkt_open + str_to_ns(sch.start)
    hi = lo + str_to_ns(sch.duration)
    tx = book.buy_transactions + book.sell_transactions
    return int(sum(q for t, q in tx if lo <= t < hi))


def run_baseline(args) -> Dict:
    seed, exp = args
    sch = exp.schedule
    book, agent, mkt_open = _simulate(seed, exp.strategy, 0, sch, exp.market_kwargs)
    tx = book.buy_transactions + book.sell_transactions
    return {
        "seed": seed,
        "window_volume": _window_volume(book, mkt_open, sch),
        "total_volume": int(sum(q for _, q in tx)),
        "arrival_mid": agent.arrival_mid,
        "mid": _mid_on_grid(book, mkt_open, sch),
    }


def run_execution(args) -> Dict:
    seed, qty, exp = args
    sch = exp.schedule
    t0 = time.time()
    book, agent, mkt_open = _simulate(seed, exp.strategy, qty, sch, exp.market_kwargs)
    fills = np.array(agent.fills, dtype="float64").reshape(-1, 3)
    filled = int(fills[:, 1].sum()) if len(fills) else 0
    vwap = float((fills[:, 1] * fills[:, 2]).sum() / filled) if filled else np.nan
    return {
        "seed": seed,
        "qty": qty,
        "filled": filled,
        "n_fills": len(fills),
        "arrival_mid": agent.arrival_mid,
        "vwap": vwap,
        "mid": _mid_on_grid(book, mkt_open, sch),
        "wall_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _pool(workers: int):
    # spawn: identical behaviour on macOS and Linux, no fork-inherited state
    return mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=8)


def run_experiment(exp: Experiment, out_dir: str, workers: int = 4) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    sch = exp.schedule
    sign = 1.0 if sch.side == "sell" else -1.0  # +ve = adverse move
    start_s = str_to_ns(sch.start) // 10**9
    end_s = start_s + str_to_ns(sch.duration) // 10**9

    # Phase 1: baselines --------------------------------------------------
    t0 = time.time()
    with _pool(workers) as pool:
        base = pool.map(run_baseline, [(s, exp) for s in exp.seeds])
    base = {b["seed"]: b for b in base}
    mean_vol = float(np.mean([b["window_volume"] for b in base.values()]))
    print(f"[phase 1] {len(base)} baselines in {time.time() - t0:.0f}s; "
          f"mean window volume {mean_vol:,.0f} shares")

    if exp.qty is not None:
        sizes = [int(q) for q in exp.qty]
    elif exp.pct is not None:
        sizes = [int(round(p * mean_vol)) for p in exp.pct]
    else:
        raise ValueError("give pct or qty")
    print(f"[phase 2] sizes (shares): {sizes}")

    # Phase 2: execution runs --------------------------------------------
    tasks = [(s, q, exp) for q in sizes for s in exp.seeds]
    t0 = time.time()
    rows, paths = [], []
    with _pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_execution, tasks), 1):
            b = base[r["seed"]]
            ref = b["mid"].loc[start_s]            # baseline mid at execution start
            impact = sign * (b["mid"] - r["mid"]) / ref * 1e4  # sell: price drop = +ve
            arr = r["arrival_mid"]
            shortfall = sign * (arr - r["vwap"]) / arr * 1e4 if arr and r["filled"] else np.nan

            rows.append({
                "strategy": exp.strategy,
                "seed": r["seed"],
                "qty": r["qty"],
                "pct": r["qty"] / mean_vol,
                "filled": r["filled"],
                "n_fills": r["n_fills"],
                "arrival_mid": arr / 100 if arr else np.nan,
                "vwap": r["vwap"] / 100,
                "shortfall_bps": shortfall,
                "impact_at_end_bps": impact.loc[end_s],
                "impact_peak_bps": impact.loc[start_s:end_s].max(),
                "impact_end_plus_30m_bps": impact.get(end_s + 1800, np.nan),
                # Sanity check: must be 0 if the baseline is a true counterfactual
                "pre_start_max_abs_bps": impact.loc[: start_s - 1].abs().max(),
                "window_volume": b["window_volume"],
                "wall_s": r["wall_s"],
            })
            paths.append(pd.DataFrame({
                "seed": r["seed"], "qty": r["qty"],
                "t_rel_s": impact.index - start_s, "impact_bps": impact.values,
            }))
            if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} runs ({time.time() - t0:.0f}s)")

    runs = pd.DataFrame(rows).sort_values(["qty", "seed"]).reset_index(drop=True)
    paths = pd.concat(paths, ignore_index=True).sort_values(["qty", "seed", "t_rel_s"])
    baselines = pd.DataFrame([{k: v for k, v in b.items() if k != "mid"} for b in base.values()])

    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    paths.to_csv(os.path.join(out_dir, "paths.csv"), index=False)
    baselines.to_csv(os.path.join(out_dir, "baselines.csv"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({**asdict(exp), "sizes": sizes, "mean_window_volume": mean_vol}, f, indent=2)

    bad = runs[runs.pre_start_max_abs_bps > 1e-9]
    if len(bad):
        print(f"WARNING: {len(bad)} runs diverge from baseline before execution starts; "
              "the counterfactual is broken for those seeds.")
    return {"runs": runs, "paths": paths, "baselines": baselines}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI half-width across seeds, per size."""
    cols = ["shortfall_bps", "impact_peak_bps", "impact_at_end_bps", "impact_end_plus_30m_bps"]
    g = runs.groupby(["qty", "pct"])[cols]
    mean, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
    out = mean.round(2).astype(str) + " ± " + (1.96 * se).round(2).astype(str)
    out["fill_rate"] = (runs.groupby(["qty", "pct"]).apply(lambda d: d.filled.sum() / d.qty.sum())).round(3)
    out["n_seeds"] = g.count().iloc[:, 0]
    return out.reset_index()


def parse_market(items: List[str]) -> Dict:
    """['num_value_agents=25', 'mm_pov=0.01'] -> {'num_value_agents': 25, 'mm_pov': 0.01}"""
    import inspect
    valid = set(inspect.signature(rmsc04.build_config).parameters)
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        if key not in valid:
            raise SystemExit(f"--market: unknown key '{key}'. Valid: {sorted(valid)}")
        for cast in (int, float):
            try:
                out[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="twap", choices=sorted(STRATEGIES))
    p.add_argument("--seeds", type=int, default=30, help="number of seeds (0..N-1)")
    p.add_argument("--seed-offset", type=int, default=0)
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--pct", type=float, nargs="+", help="sizes as fraction of window volume")
    size.add_argument("--qty", type=int, nargs="+", help="sizes in shares")
    p.add_argument("--side", default="sell", choices=["sell", "buy"])
    p.add_argument("--end-time", default="11:00:00")
    p.add_argument("--start", default="00:10:00", help="delay after the open")
    p.add_argument("--duration", default="00:30:00")
    p.add_argument("--slices", type=int, default=30)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="results/sweep")
    p.add_argument("--market", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override rmsc04.build_config args, e.g. num_value_agents=25 mm_pov=0.01")
    a = p.parse_args()

    logging.getLogger("abides").setLevel(logging.WARNING)
    exp = Experiment(
        market_kwargs=parse_market(a.market),
        strategy=a.strategy,
        seeds=list(range(a.seed_offset, a.seed_offset + a.seeds)),
        pct=a.pct,
        qty=a.qty,
        schedule=Schedule(end_time=a.end_time, start=a.start, duration=a.duration,
                          n_slices=a.slices, side=a.side),
    )
    t0 = time.time()
    res = run_experiment(exp, a.out, workers=a.workers)
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {a.out}/\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summarize(res["runs"]).to_string(index=False))


if __name__ == "__main__":
    main()
