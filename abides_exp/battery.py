"""
Scenario battery: run strategies and market events against same-seed baselines and
report the same market-quality metrics for all of them.

Usage (repo root, inside .venv-abides):
  python -m abides_exp.battery --seeds 30 --workers 8                      # all scenarios
  python -m abides_exp.battery --only spoofing flash_crash --seeds 50
  python -m abides_exp.battery --seeds 30 --market num_value_agents=50     # other market
  python -m abides_exp.battery --list

Outputs in --out (default results/battery/):
  <scenario>/runs.csv    one row per seed: all scalar metrics (see METRICS below)
  <scenario>/paths.csv   seed, t_s, dmid_bps, spread_t, spread_b, mispricing_t_bps, mispricing_b_bps
  summary.csv            mean and 95% CI per scenario and metric
  battery.png            overview figure
  meta.json              settings

METRICS (treated minus baseline unless stated; bps relative to baseline mid at window start)
  dmid_*            mid-price difference: mean in window, min, max, at window end, mean 30 min after
  spread_ratio      mean spread treated / baseline, inside the window
  vol_ratio         std of 10s mid returns treated / baseline, inside the window
  mispricing_*      mean |mid - true fundamental| in bps, inside the window, each run
  pnl_<AgentType>   change in PnL ($) of all agents of that type, marked at the run's final
                    true fundamental. These sum to ~0: who pays whom.
  subject_*         the strategy or adversary itself (shortfall, fill rate, PnL)
  victim_*          the TWAP seller in the predatory scenario (treated and baseline runs)
  pre_start_max_abs_bps  must be 0: runs identical before the window starts
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from abides_core import abides

from abides_exp.agents.execution import ExecutionAgent
from abides_exp.agents.scripted import ScriptedTrader
from abides_exp.events import fundamental_path
from abides_exp.scenarios import END_TIME, SCENARIOS, TICKER

NS = 10**9
GRID_S = 10


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def _book_grid(book, mkt_open: int, grid_ns: np.ndarray):
    l1 = book.get_L1_snapshots()
    t = l1["best_bids"][:, 0].astype("int64")
    bid = pd.to_numeric(pd.Series(l1["best_bids"][:, 1]), errors="coerce").values
    ask = pd.to_numeric(pd.Series(l1["best_asks"][:, 1]), errors="coerce").values
    order = np.argsort(t, kind="stable")
    t, bid, ask = t[order], bid[order], ask[order]
    # Sample the last book snapshot at or before each grid time. An empty side stays NaN:
    # a one-sided book has no mid, and hiding that would understate crashes.
    idx = np.searchsorted(t, grid_ns, side="right") - 1
    ok = idx >= 0
    b = np.where(ok, bid[np.clip(idx, 0, None)], np.nan)
    a = np.where(ok, ask[np.clip(idx, 0, None)], np.nan)
    return (a + b) / 2, a - b


def _pnl_by_type(agents, f_end: float) -> Dict[str, float]:
    out = defaultdict(float)
    for ag in agents[1:]:  # skip exchange
        h = getattr(ag, "holdings", None)
        if not h:
            continue
        pnl = h.get("CASH", 0) + h.get(TICKER, 0) * f_end - ag.starting_cash
        out[ag.type] += pnl / 100.0  # dollars
    return dict(out)


def _exec_stats(agent: ExecutionAgent) -> Dict:
    if agent is None or not isinstance(agent, ExecutionAgent):
        return {}
    f = np.array(agent.fills, dtype=float).reshape(-1, 3)
    filled = f[:, 1].sum() if len(f) else 0
    vwap = (f[:, 1] * f[:, 2]).sum() / filled if filled else np.nan
    arr = agent.arrival_mid
    sign = 1 if agent.side == "sell" else -1
    sf = sign * (arr - vwap) / arr * 1e4 if (arr and filled) else np.nan
    return {"filled": filled, "fill_rate": filled / agent.total_qty if agent.total_qty else np.nan,
            "vwap": vwap / 100 if filled else np.nan, "arrival_mid": arr / 100 if arr else np.nan,
            "shortfall_bps": sf}


def _run(scenario: str, seed: int, treated: bool, market: Dict, params: Dict):
    cfg, h = SCENARIOS[scenario](seed, treated, market, params or None)
    end_state = abides.run(cfg)
    ex = end_state["agents"][0]
    mkt_open = ex.mkt_open
    end_s = (cfg["stop_time"] - mkt_open) // NS
    grid_s = np.arange(0, end_s + 1, GRID_S)
    grid_ns = mkt_open + grid_s * NS
    mid, spread = _book_grid(ex.order_books[TICKER], mkt_open, grid_ns)
    fund = fundamental_path(h["oracle"], TICKER, grid_ns)
    f_end = fund[~np.isnan(fund)][-1]
    out = {"grid_s": grid_s, "mid": mid, "spread": spread, "fund": fund,
           "pnl": _pnl_by_type(end_state["agents"], f_end), "window": h["window"],
           "kind": h["kind"], "meta": h["meta"], "f_end": f_end, "mkt_open": mkt_open}
    subj, vic = h["subject"], h["victim"]
    out["subject"] = _exec_stats(subj)
    if isinstance(subj, ScriptedTrader):
        traded = sum(q for _, q, _, _ in subj.fills)
        out["subject"] = {"pnl_usd": subj.pnl_cents(f_end) / 100, "shares_traded": traded,
                          "market_fill_rate": subj.market_fill_rate(),
                          "end_position": subj.position(), "n_fills": len(subj.fills)}
        out["script_log"] = subj.log_rows
    out["victim"] = _exec_stats(vic)
    return out


# ---------------------------------------------------------------------------
# Paired metrics
# ---------------------------------------------------------------------------
def _at(grid_s, series, t):
    i = int(np.searchsorted(grid_s, t, side="right") - 1)
    return series[max(i, 0)]


def run_pair(args):
    scenario, seed, market, params = args
    t0 = time.time()
    T = _run(scenario, seed, True, market, params)
    B = _run(scenario, seed, False, market, params)
    g = T["grid_s"]
    s0, s1 = T["window"]
    ref = _at(g, B["mid"], s0)
    dmid = (T["mid"] - B["mid"]) / ref * 1e4
    mis_t = np.abs(T["mid"] - T["fund"]) / T["fund"] * 1e4
    mis_b = np.abs(B["mid"] - B["fund"]) / B["fund"] * 1e4
    win = (g >= s0) & (g <= s1)
    post = (g > s1) & (g <= s1 + 1800)
    pre = g < s0

    def ret_std(m):
        r = np.diff(np.log(m[win]))
        r = r[np.isfinite(r)]
        return r.std() if len(r) > 2 else np.nan

    row = {
        "scenario": scenario, "seed": seed,
        "pre_start_max_abs_bps": np.nanmax(np.abs(dmid[pre])) if pre.any() else 0.0,
        "dmid_mean_bps": np.nanmean(dmid[win]),
        "dmid_min_bps": np.nanmin(dmid[win]),
        "dmid_max_bps": np.nanmax(dmid[win]),
        "dmid_end_bps": _at(g, dmid, s1),
        "dmid_post30_bps": np.nanmean(dmid[post]) if post.any() else np.nan,
        "spread_ratio": np.nanmean(T["spread"][win]) / np.nanmean(B["spread"][win]),
        "spread_t_cents": np.nanmean(T["spread"][win]),
        "vol_ratio": ret_std(T["mid"]) / ret_std(B["mid"]),
        "mispricing_t_bps": np.nanmean(mis_t[win]),
        "mispricing_b_bps": np.nanmean(mis_b[win]),
        # share of the window with an empty bid or ask side (no valid mid)
        "one_sided_t": float(np.mean(np.isnan(T["mid"][win]))),
        "one_sided_b": float(np.mean(np.isnan(B["mid"][win]))),
    }
    types = set(T["pnl"]) | set(B["pnl"])
    for ty in sorted(types):
        row[f"pnl_{ty}"] = T["pnl"].get(ty, 0.0) - B["pnl"].get(ty, 0.0)
    row["pnl_sum_check"] = sum(row[f"pnl_{ty}"] for ty in types)
    for k, v in T["subject"].items():
        row[f"subject_{k}"] = v
    if T["victim"]:
        for k, v in T["victim"].items():
            row[f"victim_t_{k}"] = v
        for k, v in B["victim"].items():
            row[f"victim_b_{k}"] = v
        row["victim_extra_cost_bps"] = row["victim_t_shortfall_bps"] - row["victim_b_shortfall_bps"]

    # Scenario-specific
    meta = T["meta"]
    if scenario == "fundamental_jump":
        j = meta["jump_bps"]
        after = g >= meta["at_s"]
        for frac in (0.5, 0.9):
            hit = np.where(after & (np.sign(j) * dmid >= abs(j) * frac))[0]
            row[f"time_to_{int(frac * 100)}pct_s"] = (g[hit[0]] - meta["at_s"]) if len(hit) else np.nan
        # Where the price settles relative to the jump, 10-30 minutes after it
        settle = (g >= meta["at_s"] + 600) & (g <= meta["at_s"] + 1800)
        row["settled_gap_bps"] = np.nanmean(dmid[settle]) - j
    if scenario == "flash_crash":
        after = g >= s0
        i_min = np.nanargmin(np.where(after, dmid, np.nan))
        trough = dmid[i_min]
        rec = np.where((g > g[i_min]) & (dmid >= trough / 2))[0]
        row["trough_bps"] = trough
        row["trough_time_s"] = g[i_min] - s0
        row["half_recovery_s"] = (g[rec[0]] - g[i_min]) if len(rec) else np.nan
    if scenario == "spoofing" and T.get("script_log"):
        # Did the fake order move the mid in its direction before the real trade?
        pushes = []
        for r in T["script_log"]:
            if r["type"] != "limit":
                continue
            t_on = (r["t"] - T["mkt_open"]) / NS
            sgn = 1 if r["side"] == "buy" else -1
            before = _at(g, dmid, t_on)
            during = _at(g, dmid, t_on + meta["cycle_s"] / 2 - 1)
            pushes.append(sgn * (during - before))
        row["spoof_push_bps"] = float(np.nanmean(pushes)) if pushes else np.nan

    row["wall_s"] = time.time() - t0
    paths = pd.DataFrame({"seed": seed, "t_s": g, "dmid_bps": dmid, "spread_t": T["spread"],
                          "spread_b": B["spread"], "mispricing_t_bps": mis_t, "mispricing_b_bps": mis_b})
    return row, paths, {"window": T["window"], "kind": T["kind"], "meta": meta}


# ---------------------------------------------------------------------------
# Orchestration and summary
# ---------------------------------------------------------------------------
def ci95(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


KEY_METRICS = ["dmid_mean_bps", "dmid_min_bps", "dmid_end_bps", "dmid_post30_bps",
               "spread_ratio", "vol_ratio", "mispricing_t_bps", "mispricing_b_bps"]


def summarize(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skip = {"seed", "wall_s", "scenario"}
    for scen, d in all_runs.groupby("scenario", sort=False):
        for col in d.columns:
            if col in skip or not np.issubdtype(d[col].dtype, np.number):
                continue
            x = d[col].dropna()
            if x.empty:
                continue
            rows.append({"scenario": scen, "metric": col, "mean": x.mean(), "ci95": ci95(x),
                         "n": len(x)})
    return pd.DataFrame(rows)


def plot(out_dir, paths_by, info_by, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK_2, GRID, SURF = "#1f1f1e", "#5f5e5a", "#e6e5e1", "#fcfcfb"
    BLUE = "#2a78d6"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    names = list(paths_by)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.9 * nrow), squeeze=False)
    for ax, name in zip(axes.flat, names):
        p = paths_by[name]
        s0, s1 = info_by[name]["window"]
        d = p.groupby("t_s").dmid_bps
        m, c = d.mean(), d.apply(ci95)
        t = (m.index - s0) / 60
        ax.axvspan(0, (s1 - s0) / 60, color=GRID, alpha=0.7, lw=0)
        ax.axhline(0, color=INK_2, lw=0.7)
        ax.fill_between(t, m - c, m + c, color=BLUE, alpha=0.2, lw=0)
        ax.plot(t, m, color=BLUE, lw=1.6)
        ax.set_title(name.replace("_", " "), loc="left", fontsize=10)
        ax.set_xlim(max(t.min(), -10), min(t.max(), (s1 - s0) / 60 + 30))
        ax.grid(axis="y", color=GRID, lw=0.7)
        n = p.seed.nunique()
        ax.text(0.99, 0.03, f"n={n}", transform=ax.transAxes, ha="right", va="bottom",
                color=INK_2, fontsize=8)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ mid vs baseline (bps)")
    for ax in axes[-1, :]:
        ax.set_xlabel("minutes from window start")
    fig.suptitle("Mean mid-price effect vs same-seed baseline (shaded: active window; band: 95% CI)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "battery.png"), dpi=170, facecolor=SURF)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None, help="subset of scenarios")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--market", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default="results/battery")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    if a.list:
        for k in SCENARIOS:
            print(k)
        return
    from abides_exp.harness import parse_market
    market = parse_market(a.market)
    names = a.only or list(SCENARIOS)
    bad = [n for n in names if n not in SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenarios {bad}; use --list")
    logging.getLogger("abides").setLevel(logging.WARNING)
    os.makedirs(a.out, exist_ok=True)

    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    tasks = [(n, s, market, {}) for n in names for s in seeds]
    print(f"{len(names)} scenarios x {len(seeds)} seeds = {len(tasks)} pairs ({2 * len(tasks)} simulations)")
    rows, paths_by, info_by = [], defaultdict(list), {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
        for i, (row, paths, info) in enumerate(pool.imap_unordered(run_pair, tasks), 1):
            rows.append(row)
            paths_by[row["scenario"]].append(paths)
            info_by[row["scenario"]] = info
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} pairs ({time.time() - t0:.0f}s)")

    all_runs = pd.DataFrame(rows)
    for name in names:
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        r = all_runs[all_runs.scenario == name].dropna(axis=1, how="all").sort_values("seed")
        r.to_csv(os.path.join(d, "runs.csv"), index=False)
        paths_by[name] = pd.concat(paths_by[name], ignore_index=True).sort_values(["seed", "t_s"])
        paths_by[name].to_csv(os.path.join(d, "paths.csv"), index=False)
    summ = summarize(all_runs)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"seeds": seeds, "market": market, "grid_s": GRID_S, "end_time": END_TIME,
                   "scenarios": {n: {k: v for k, v in info_by[n].items()} for n in names}},
                  f, indent=2, default=str)
    plot(a.out, {n: paths_by[n] for n in names}, info_by, summ)

    bad = all_runs[all_runs.pre_start_max_abs_bps > 1e-9]
    if len(bad):
        print(f"\nWARNING: {len(bad)} pairs diverge before their window starts:")
        print(bad.groupby("scenario").size().to_string())
    leak = all_runs[all_runs.pnl_sum_check.abs() > 1.0]
    if len(leak):
        print(f"\nWARNING: PnL changes do not sum to zero in {len(leak)} pairs (check accounting)")

    show = summ[summ.metric.isin(KEY_METRICS + ["subject_shortfall_bps", "subject_pnl_usd",
                                                 "victim_extra_cost_bps", "spoof_push_bps",
                                                 "time_to_50pct_s", "time_to_90pct_s",
                                                 "trough_bps", "half_recovery_s", "settled_gap_bps",
                                                 "subject_fill_rate", "subject_market_fill_rate",
                                                 "one_sided_t", "one_sided_b"])]
    show = show.assign(v=show["mean"].round(2).astype(str) + " ± " + show["ci95"].round(2).astype(str))
    table = show.pivot(index="metric", columns="scenario", values="v").reindex(columns=names)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 16):
        print("\n" + table.fillna("").to_string())
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {a.out}/")


if __name__ == "__main__":
    main()
