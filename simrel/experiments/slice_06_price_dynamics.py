"""
Price dynamics that the default LOB-Bench statistics miss: (1) LOB-Bench-scored return and volatility statistics,
(2) the price response to aggressive orders, for every simulator version and slice, and for TRADES in its window.

    python simrel/experiments/slice_06_price_dynamics.py [--procs N]
"""

from __future__ import annotations

import glob
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402
from simrel.metrics import impact_response as IR  # noqa: E402

OUT = R.OUT
RAW = R.RAW / "price"
SOURCES = ["real_fit_days", "real_test", "abides", "abides_quoting", "abides_quoting_mom"] + [v for v in C.HAWKES_VERSIONS if v != "hawkes_s"]


def windows(source: str, name: str) -> list:
    if source in ("real_fit_days", "real_test"):
        return R.real_windows(name, "fit" if source == "real_fit_days" else "test")
    return [R.two_sided(w)[0] for w in R.generated(source, name)]


def subset(ws: list) -> list:
    keep = sorted(np.random.default_rng(1).permutation(len(ws))[: C.MAX_SCORE_WINDOWS])
    return [ws[i] for i in keep]


def job(args) -> str:
    from simrel.metrics import lobbench

    source, name = args
    path = RAW / f"{source}_{name}.pkl"
    if path.exists():
        return "cached"
    R._memory_guard(path)
    ws = windows(source, name)
    resp = pd.concat([IR.responses(w.message, w.book, w.book_before) for w in ws], ignore_index=True)
    out = {"responses": resp}
    if source != "real_test":
        real = subset(R.real_windows(name, "test"))
        gen = subset(ws) if source == "real_fit_days" else ws
        real = real[: len(gen)]  # as in the default scoring: LOB-Bench needs at least as many generated windows as real ones
        t = lobbench.score([(w.message, w.book) for w in real], [(w.message, w.book) for w in gen], config=lobbench.price_dynamics_config())
        out["table"] = t
    R.save(path, out)
    return f"{source} {name}"


def trades_job(_=None) -> str:
    from simrel.experiments import slice_03_trades_score as T
    from simrel.metrics import lobbench

    path = RAW / "trades_window.pkl"
    if path.exists():
        return "cached"
    R._memory_guard(path)
    paths = sorted(glob.glob(str(Path("~/simdata/trades_runs").expanduser() / "gen_*.pkl")))
    days = sorted({Path(p).name.split("_")[1] for p in paths})
    real = [T.real_window(d) for d in days]
    sets = {"trades": [R.two_sided(T.trades_window(p))[0] for p in paths], "real_fit_days": [T.real_window(d) for d in C.FIT_DAYS],
            "hawkes": [R.two_sided(w)[0] for w in R.generated("hawkes", "volH_voluH_trendH")],
            "abides": [R.two_sided(w)[0] for w in R.generated("abides", "volH_voluH_trendH")]}
    out = {"real_responses": pd.concat([IR.responses(w.message, w.book, w.book_before) for w in real], ignore_index=True)}
    for src, ws in sets.items():
        out[src] = {"table": lobbench.score([(w.message, w.book) for w in real], [(w.message, w.book) for w in ws], config=lobbench.price_dynamics_config()),
                    "responses": pd.concat([IR.responses(w.message, w.book, w.book_before) for w in ws], ignore_index=True)}
    R.save(path, out)
    return "trades window"


def report() -> None:
    pd.set_option("display.width", 250)
    tabs, resp = [], []
    for p in sorted(RAW.glob("*_vol*.pkl")):
        stem = p.stem
        name = next(n for n in R.S.slice_names() if stem.endswith("_" + n))
        source = stem[: -len(name) - 1]
        d = R.load(p)
        if "failed" in d:
            print("FAILED", stem, d["failed"])
            continue
        r = d["responses"].assign(source=source, slice=name)
        resp.append(r)
        if "table" in d:
            tabs.append(d["table"].assign(source=source, slice=name))
    t = pd.concat(tabs)
    t.to_csv(OUT / "price_dynamics_by_statistic.csv", index=False)
    wide = t.pivot_table(index="source", columns="statistic", values="l1")
    wide["price_l1_mean"] = wide.mean(axis=1)
    default = pd.read_csv(OUT / "realism.csv").groupby("source")["l1_mean"].mean().rename("lobbench_default_l1")
    wide = wide.join(default)
    r = pd.concat(resp, ignore_index=True)
    r.to_csv(OUT / "trade_responses.csv", index=False)
    r["size_bin"] = pd.cut(r["size"], IR.SIZE_BINS, labels=["<=100", "101-500", "501-2000", ">2000"])
    big = r[r["size"] > 500].groupby("source")[["resp_1s", "resp_10s", "resp_60s"]].mean().add_prefix("big_")
    allr = r.groupby("source")[["resp_10s", "resp_60s"]].mean().add_prefix("all_")
    summary = wide.join(big).join(allr).sort_values("price_l1_mean")
    summary.to_csv(OUT / "price_dynamics_summary.csv")
    print(summary.round(3).to_string())
    print(r.groupby(["source", "size_bin"], observed=False)["resp_10s"].mean().unstack().round(2).to_string())
    print(r.groupby(["source", "size_bin"], observed=False).size().unstack().to_string())
    tw = R.load(RAW / "trades_window.pkl") if (RAW / "trades_window.pkl").exists() else None
    if tw:
        rows = []
        for src in ("real_fit_days", "hawkes", "abides", "trades"):
            x = tw[src]
            rr = x["responses"]
            rows.append({"source": src, "price_l1_mean": x["table"]["l1"].mean(), **x["table"].set_index("statistic")["l1"].to_dict(),
                         "big_resp_10s": rr[rr["size"] > 500]["resp_10s"].mean(), "big_resp_60s": rr[rr["size"] > 500]["resp_60s"].mean()})
        rr = tw["real_responses"]
        rows.append({"source": "real_held_out", "big_resp_10s": rr[rr["size"] > 500]["resp_10s"].mean(), "big_resp_60s": rr[rr["size"] > 500]["resp_60s"].mean()})
        tt = pd.DataFrame(rows)
        tt.to_csv(OUT / "price_dynamics_trades_window.csv", index=False)
        print("== 09:46 window (TRADES comparison)")
        print(tt.round(3).to_string(index=False))


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 3
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(s, n) for n in R.usable_slices() for s in SOURCES if not (RAW / f"{s}_{n}.pkl").exists()
            and (s.startswith("real") or len(R.generated(s, n)) >= C.N_GEN)]
    if "--report" not in sys.argv:
        R.pool_map(job, jobs, procs, "price")
        print(trades_job())
    report()
