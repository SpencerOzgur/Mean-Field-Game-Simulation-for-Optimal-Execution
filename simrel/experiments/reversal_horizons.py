"""
Horizon sweep for the reversal check (simrel/reports/reversal_horizons_plan.md). Exploratory: the 30 s result of the
committed check was known when the sweep was defined.

The same 24 real files and the same 24 simulated markets and passive episodes as the committed check; S2, S3 and S4
at h = 5, 10, 30, 60 and 120 s, with the signal's two components and two lot sizes. New files only:
results/simrel/goal2/reversal_check/horizons.csv, horizons_reproduction.csv, fig_reversal_horizons.png.

    python simrel/experiments/reversal_horizons.py timing [procs]
    python simrel/experiments/reversal_horizons.py run [procs]      segments (cached), table, reproduction check, figure
"""

from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.eventsim import GRID_DT  # noqa: E402
from simrel.experiments import reversal_check as R  # noqa: E402
from simrel.lob import Book  # noqa: E402
from simrel.metrics import reversal as RV  # noqa: E402

OUT, RAW = R.OUT, R.RAW
HORIZONS = [5, 10, 30, 60, 120]
REGRESSORS = {"both": "s3_x", "flow": "s3_imb", "ret": "s3_ret"}
VARIANTS = [(lot, comp) for lot in (1, 100) for comp in ("both", "flow", "ret")]
N_BOOT = 2000


def compact(seg: dict, qty: int) -> dict:
    """What the table and the bootstraps need from one segment at one horizon."""
    out = {k: seg[k] for k in ("s2_num", "s2_den", "s3_x", "s3_imb", "s3_ret", "s3_y", "zero_signal_n", "zero_volume_n", "zero_move_n", "n_bins",
                               "mean_mid", "tick")}
    y = seg["s3_y"].astype(float)
    for name, key in REGRESSORS.items():
        x = seg[key].astype(float)
        out["sums_" + name] = (len(x), x.sum(), y.sum(), (x * x).sum(), (x * y).sum())
    for lot, comp in VARIANTS:
        g = [RV.backtest(w["mid"], w["vol"], w["signed"], qty, "sell", False, w["bid"], w["ask"], lot=lot, component=comp) for w in seg["windows"]]
        out[f"gap_{lot}_{comp}"] = np.array([x["p6"] - x["twap"] for x in g])
    return out


def sim_job(args):
    """The passive episode of the committed check (same market, same seed), summarised at every horizon."""
    market, seed, qty = args
    warnings.filterwarnings("ignore")
    model = R.make(market)(seed)
    book, tape, t = Book(), [], 0.0
    model.init_book(book)
    bid, ask = [], []
    end = R.WARMUP_S + R.EPISODE_S
    while t < end + 1e-9:
        if t >= R.WARMUP_S - 1e-9:
            bb, ba = book.best_bid(), book.best_ask()
            bid.append(float(bb) if bb is not None else np.nan)
            ask.append(float(ba) if ba is not None else np.nan)
        if t >= end - 1e-9:
            break
        model.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    tr = np.array([(x[0] - R.WARMUP_S, x[1]) for x in tape if x[0] >= R.WARMUP_S], dtype=float).reshape(-1, 2)
    bid, ask = np.array(bid), np.array(ask)
    return {"unit": market, "segment": seed, **{h: compact(RV.segment_summary(bid, ask, tr, 1.0, bin_s=float(h)), qty) for h in HORIZONS}}


def real_job(args):
    """One day of one stock (the committed check's segment), summarised at every horizon."""
    symbol, day, qty = args
    from simrel.configs import slices as CS
    from simrel.data import lobster as L

    msg, book = L.load_day(CS.DATA_DIR, symbol, day)
    t = msg[:, 0]
    a, b = np.searchsorted(t, R.DAY_START_S * L.NS, side="left"), np.searchsorted(t, (R.DAY_START_S + R.DAY_LEN_S) * L.NS, side="left")
    w = L.Window(symbol, day, R.DAY_START_S, R.DAY_LEN_S, msg[a:b], book[a:b], book[a - 1])
    _, bid, ask = L.grid_quotes(w)
    bad = (ask >= L.EMPTY) | (bid <= -L.EMPTY) | (ask <= 0) | (bid <= 0)
    bid, ask = np.where(bad, np.nan, bid), np.where(bad, np.nan, ask)
    ex = w.message[(w.message[:, 1] == 4) | (w.message[:, 1] == 5)]
    tr = np.c_[(ex[:, 0] - R.DAY_START_S * L.NS) / L.NS, -ex[:, 5] * ex[:, 3]].astype(float)
    return {"unit": symbol, "segment": day, **{h: compact(RV.segment_summary(bid, ask, tr, float(L.TICK), bin_s=float(h)), qty) for h in HORIZONS}}


def quantities() -> tuple[dict, dict]:
    """The parent sizes of the committed check: per simulated market and per stock."""
    s, r = pd.read_csv(OUT / "sim_table.csv"), pd.read_csv(OUT / "real_table.csv")
    return ({x["unit"]: int(x["qty"]) for _, x in s[s["unit"] != "pooled"].iterrows()},
            {x["family"]: int(x["qty"]) for _, x in r[r["unit"] == "8 days"].iterrows()})


def segments(procs: int) -> tuple[list, list]:
    from simrel.experiments.run_reference import pool_map

    qs, qr = quantities()
    sim = R.cached("horizons_sim", lambda: pool_map(sim_job, [(m, R.SEED_PASSIVE + k, qs[m]) for m in R.markets() for k in range(R.N_PASSIVE)], procs, "sim"))
    assert not R.missing_files(), R.missing_files()
    real = R.cached("horizons_real", lambda: pool_map(real_job, [(s, d, qr[s]) for s in R.SYMBOLS for d in R.DAYS], procs, "real"))
    key = lambda x: (x["unit"], x["segment"])  # noqa: E731
    return sorted(sim, key=key), sorted(real, key=key)


# ---- statistics of a set of segments at one horizon
def _slope(sums: np.ndarray) -> float:
    n, sx, sy, sxx, sxy = sums
    den = sxx - sx * sx / n
    return (sxy - sx * sy / n) / den if den > 0 else np.nan


def point(segs: list, h: int) -> dict:
    """Point estimates with the functions of the committed check."""
    ss = [s[h] for s in segs]
    out = {"S2": RV.s2(ss)}
    for name, key in REGRESSORS.items():
        b, t = RV.s3(ss, key)
        out[f"S3_{name}"], out[f"S3_{name}_t"] = b, t
    for lot, comp in VARIANTS:
        g = np.concatenate([s[f"gap_{lot}_{comp}"] for s in ss])
        out[f"S4_lot{lot}_{comp}"] = float(g.mean())
        if comp == "both":
            out[f"S4_lot{lot}_share_p6_cheaper"], out[f"S4_lot{lot}_share_tie"] = float((g < 0).mean()), float((g == 0).mean())
    n = sum(len(s["s3_x"]) for s in ss)
    out.update({"n_segments": len(ss), "n_windows": int(sum(len(s["gap_1_both"]) for s in ss)), "n_bins": int(n),
                "share_zero_signal": sum(s["zero_signal_n"] for s in ss) / n, "share_zero_volume": sum(s["zero_volume_n"] for s in ss) / n,
                "share_zero_move": sum(s["zero_move_n"] for s in ss) / n,
                "tick_bp": float(np.mean([s["tick"] / s["mean_mid"] for s in ss]) * 1e4)})
    return out


def boot(segs: list, h: int, groups: list | None, rng) -> dict:
    """Percentile 95% intervals. Without groups the segments are resampled (days); with groups (the market of each
    segment) markets are resampled and segments within each drawn market. Slopes come from per-segment sums."""
    ss = [s[h] for s in segs]
    n = len(ss)
    s2 = np.array([(s["s2_num"], s["s2_den"]) for s in ss])
    sums = {name: np.array([s["sums_" + name] for s in ss], dtype=float) for name in REGRESSORS}
    gap = {v: (np.array([s[f"gap_{v[0]}_{v[1]}"].sum() for s in ss]), np.array([len(s[f"gap_{v[0]}_{v[1]}"]) for s in ss], dtype=float)) for v in VARIANTS}
    idx = np.arange(n)
    by = None if groups is None else {m: idx[np.array(groups) == m] for m in sorted(set(groups))}
    res = {k: [] for k in ["S2"] + [f"S3_{c}" for c in REGRESSORS] + [f"S4_lot{lot}_{c}" for lot, c in VARIANTS]}
    for _ in range(N_BOOT):
        if by is None:
            pick = rng.choice(idx, n)
        else:
            names = list(by)
            pick = np.concatenate([rng.choice(by[m], len(by[m])) for m in rng.choice(names, len(names))])
        a = s2[pick].sum(0)
        res["S2"].append(a[0] / a[1] if a[1] > 0 else np.nan)
        for name in REGRESSORS:
            res[f"S3_{name}"].append(_slope(sums[name][pick].sum(0)))
        for v in VARIANTS:
            res[f"S4_lot{v[0]}_{v[1]}"].append(gap[v][0][pick].sum() / gap[v][1][pick].sum())
    out = {}
    for k, v in res.items():
        out[k + "_lo"], out[k + "_hi"] = float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
    return out


def table(sim: list, real: list) -> pd.DataFrame:
    M = R.markets()
    rng = np.random.default_rng(20261114)
    units = [("real", sym, [s for s in real if s["unit"] == sym], None) for sym in R.SYMBOLS]
    for fam, label in R.FAMILY.items():
        segs = [s for s in sim if M[s["unit"]]["family"] == fam]
        units.append(("simulated", label, segs, [s["unit"] for s in segs]))
    rows = []
    for side, name, segs, groups in units:
        for h in HORIZONS:
            row = {"side": side, "unit": name, "horizon_s": h, "decisions_per_window": int(600 / h), **point(segs, h), **boot(segs, h, groups, rng)}
            for k in [c for c in row if c.startswith(("S3_", "S4_")) and not c.endswith(("_t", "_share_p6_cheaper", "_share_tie"))]:
                row[k + "_ticks"] = row[k] / row["tick_bp"]  # point estimates and interval ends in ticks
            rows.append(row)
        print(f"  {side} {name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "horizons.csv", index=False)
    return out


def reproduction(t: pd.DataFrame) -> pd.DataFrame:
    """h = 30, lot 100, combined signal against the committed tables (point estimates)."""
    s, r = pd.read_csv(OUT / "sim_table.csv"), pd.read_csv(OUT / "real_table.csv")
    old = pd.concat([r[r["unit"] == "8 days"].assign(unit=lambda d: d["family"]), s[s["unit"] == "pooled"].assign(unit=lambda d: d["family"])])
    new = t[t["horizon_s"] == 30].set_index("unit")
    pairs = {"S2": "S2", "S3_slope": "S3_both", "S3_t": "S3_both_t", "S4": "S4_lot100_both", "S4_share_p6_cheaper": "S4_lot100_share_p6_cheaper",
             "S4_share_tie": "S4_lot100_share_tie", "n_windows": "n_windows", "n_segments": "n_segments"}
    rows = []
    for _, o in old.iterrows():
        for a, b in pairs.items():
            rows.append({"unit": o["unit"], "statistic": a, "committed": o[a], "sweep_h30": new.loc[o["unit"], b], "difference": new.loc[o["unit"], b] - o[a],
                         "identical": bool(new.loc[o["unit"], b] == o[a])})
        for a, b in (("S2", "S2"), ("S3_slope", "S3_both"), ("S4", "S4_lot100_both")):
            for e in ("_lo", "_hi"):
                rows.append({"unit": o["unit"], "statistic": a + e + " (interval, redrawn)", "committed": o[a + e], "sweep_h30": new.loc[o["unit"], b + e],
                             "difference": new.loc[o["unit"], b + e] - o[a + e], "identical": False})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "horizons_reproduction.csv", index=False)
    return out


def figure(t: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2933", "#52606d", "#d9dee4"
    color = {"INTC": "#c2571a", "CSCO": "#8a5a00", "MSFT": "#b0307a", "queue-reactive": "#7aa7e0", "agent-mix": "#12427f", "latent": "#2a9d8f"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, k, title in ((axes[0], "S3_both", "S3: next-h return on P6's signal, ticks per unit of signal (positive = continuation)"),
                         (axes[1], "S4_lot1_both", "S4: impact-free P6 minus TWAP, ticks, lot 1 (negative = P6 cheaper)")):
        for u, d in t.groupby("unit", sort=False):
            d = d.sort_values("horizon_s")
            real = d["side"].iloc[0] == "real"
            ax.fill_between(d["horizon_s"], d[k + "_lo_ticks"], d[k + "_hi_ticks"], color=color[u], alpha=0.13, linewidth=0)
            ax.plot(d["horizon_s"], d[k + "_ticks"], color=color[u], lw=2, ls="--" if real else "-", marker="o", ms=5, markeredgecolor="white",
                    label=f"{u} ({'real' if real else 'simulated'})")
        ax.axhline(0, color=muted, lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.minorticks_off()
        ax.set_xlabel("horizon h (s)", color=muted, fontsize=9)
        ax.set_title(title, fontsize=9.5, color=ink, loc="left")
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2)
    fig.suptitle("Horizon sweep (exploratory): bands are 95% bootstrap intervals; dashed lines are real stocks. S4 at h other than 30 s is P6 "
                 "re-parameterised, not the study's P6", fontsize=9, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_reversal_horizons.png", dpi=160)
    plt.close(fig)


def timing(procs: int) -> None:
    qs, qr = quantities()
    t0 = time.time()
    sim_job(("R2a_am_anchored", R.SEED_PASSIVE, qs["R2a_am_anchored"]))
    ts = time.time() - t0
    t0 = time.time()
    real_job(("MSFT", R.DAYS[0], qr["MSFT"]))
    tr = time.time() - t0
    n_s, n_r = len(R.markets()) * R.N_PASSIVE, len(R.SYMBOLS) * len(R.DAYS)
    print(f"  simulated {n_s} episodes x {ts:.1f} s, real {n_r} days x {tr:.1f} s: {(n_s * ts + n_r * tr) / 60:.1f} core-minutes, "
          f"about {(n_s * ts + n_r * tr) / 60 / procs * 1.25:.1f} minutes on {procs} processes, plus the bootstraps (a few minutes)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pd.set_option("display.width", 250)
    if cmd == "timing":
        timing(procs)
    else:
        sim, real = segments(procs)
        t = table(sim, real)
        rep = reproduction(t)
        pts = rep[~rep["statistic"].str.contains("interval")]
        print("reproduction at h = 30: point estimates identical:", bool(pts["identical"].all()), " largest difference:", float(pts["difference"].abs().max()))
        figure(t)
