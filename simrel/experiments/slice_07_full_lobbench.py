"""
The rest of LOB-Bench's protocol for every simulator and slice: its conditional statistics and its impact-response
evaluation, scored against the same held-out real windows as the default statistics.

    python simrel/experiments/slice_07_full_lobbench.py [--procs N] [--report]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402
from simrel.experiments.slice_06_price_dynamics import subset, windows  # noqa: E402

OUT = R.OUT
RAW = R.RAW / "full"


def sources() -> list:
    out = ["real_fit_days", "abides"]
    for v in ["abides_quoting", "abides_quoting_mom"] + list(C.HAWKES_VERSIONS):
        if v != "hawkes_s" and all(len(R.generated(v, n)) >= C.N_GEN for n in R.usable_slices()):
            out.append(v)
    return out


def job(args) -> str:
    from simrel.metrics import lobbench

    source, name = args
    path = RAW / f"{source}_{name}.pkl"
    if path.exists():
        return "cached"
    R._memory_guard(path)
    real = subset(R.real_windows(name, "test"))
    gen = subset(windows(source, name)) if source == "real_fit_days" else windows(source, name)
    real = real[: len(gen)]
    rp, gp = [(w.message, w.book) for w in real], [(w.message, w.book) for w in gen]
    cond = lobbench.score_conditional(rp, gp)
    imp = lobbench.impact_response(rp, gp, tempfile.mkdtemp())
    R.save(path, {"conditional": cond, "impact": imp})
    return f"{source} {name}: impact {imp['mean']:.2f}"


def trades_job() -> str:
    import glob

    from simrel.experiments import slice_03_trades_score as T
    from simrel.metrics import lobbench

    path = RAW / "trades_window.pkl"
    if path.exists():
        return "cached"
    paths = sorted(glob.glob(str(Path("~/simdata/trades_runs").expanduser() / "gen_*.pkl")))
    if not paths or not Path(C.DATA_DIR).expanduser().exists():
        return "trades window skipped (TRADES episodes or day files not on this machine)"
    days = sorted({Path(p).name.split("_")[1] for p in paths})
    real = [(w.message, w.book) for w in (T.real_window(d) for d in days)]
    sets = {"trades": [R.two_sided(T.trades_window(p))[0] for p in paths], "real_fit_days": [T.real_window(d) for d in C.FIT_DAYS],
            "hawkes": [R.two_sided(w)[0] for w in R.generated("hawkes", "volH_voluH_trendH")],
            "abides": [R.two_sided(w)[0] for w in R.generated("abides", "volH_voluH_trendH")]}
    out = {}
    for src, ws in sets.items():
        gp = [(w.message, w.book) for w in ws]
        out[src] = {"conditional": lobbench.score_conditional(real, gp), "impact": lobbench.impact_response(real, gp, tempfile.mkdtemp())}
    R.save(path, out)
    return "trades window"


def report() -> None:
    pd.set_option("display.width", 250)
    rows = []
    for p in sorted(RAW.glob("*_vol*.pkl")):
        name = next(n for n in R.S.slice_names() if p.stem.endswith("_" + n))
        source = p.stem[: -len(name) - 1]
        d = R.load(p)
        if "failed" in d:
            print("FAILED", p.stem, d["failed"])
            continue
        c = d["conditional"].set_index("statistic")
        keep = [s for s in c.index if s not in C.EXCLUDED_CONDITIONAL]
        rows.append({"source": source, "slice": name, "conditional": c.loc[keep, "l1"].mean(), **{f"cond: {s}": c.loc[s, "l1"] for s in c.index},
                     "impact": d["impact"]["mean"], **{f"impact {k}": v for k, v in d["impact"]["by_event"].items()}})
    full = pd.DataFrame(rows)
    order_book = pd.read_csv(OUT / "realism.csv")[["slice", "source", "l1_mean"]].rename(columns={"l1_mean": "order_book"})
    price = pd.read_csv(OUT / "price_dynamics_by_statistic.csv").groupby(["source", "slice"])["wasserstein"].mean().rename("price").reset_index()
    prof = full.merge(order_book, on=["source", "slice"], how="left").merge(price, on=["source", "slice"], how="left")
    prof.to_csv(OUT / "realism_profile_by_slice.csv", index=False)
    cols = ["order_book", "conditional", "price", "impact"] + [c for c in prof.columns if c.startswith("impact ") or c.startswith("cond: ")]
    summary = prof.groupby("source")[cols].mean().sort_values("impact")
    summary.to_csv(OUT / "realism_profile.csv")
    print(summary.round(3).to_string())
    if (RAW / "trades_window.pkl").exists():
        tw = R.load(RAW / "trades_window.pkl")
        rows = []
        for src, d in tw.items():
            c = d["conditional"].set_index("statistic")
            rows.append({"source": src, "conditional": c.drop(index=C.EXCLUDED_CONDITIONAL, errors="ignore")["l1"].mean(), "impact": d["impact"]["mean"],
                         **{f"impact {k}": v for k, v in d["impact"]["by_event"].items()}})
        t = pd.DataFrame(rows)
        t.to_csv(OUT / "realism_profile_trades_window.csv", index=False)
        print("== 09:46 window (TRADES comparison)")
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 2
    RAW.mkdir(parents=True, exist_ok=True)
    if "--report" not in sys.argv:
        jobs = [(s, n) for n in R.usable_slices() for s in sources() if not (RAW / f"{s}_{n}.pkl").exists()]
        R.pool_map(job, jobs, procs, "full")
        print(trades_job(), flush=True)
    report()
