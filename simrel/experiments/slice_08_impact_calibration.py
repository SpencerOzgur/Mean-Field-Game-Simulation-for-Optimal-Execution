"""
Calibrate impact on the FIT days: choose, per slice, the scale of the excitation caused by market orders (Hawkes) or
the number of momentum traders (ABIDES-calibrated) that brings the simulated LOB-Bench impact-response curves closest
to those of the slice's fit-day windows. Held-out (test) windows are never used here.

    python simrel/experiments/slice_08_impact_calibration.py [--procs N]

Writes results/simrel/slices/impact_calibration.json ({version: {slice: value}}) and impact_calibration.csv.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402

RAW = R.RAW / "impact_calib_v2"  # v1 (abandoned) let factors above 1 make the process explosive
SEED = 50_000


def sim_windows(version: str, name: str, value) -> list:
    out = []
    for k in range(C.N_CALIB_WINDOWS):
        if version.startswith("abides"):
            w = R.abides_window(R.abides_family_spec("abides_quoting", name, momentum=value), SEED + k)
        else:
            from simrel.lob import Book

            model, book = R.hawkes_model(name, version, impact_scale=value, fit_day_starts=True)(SEED + k), Book()
            book.tape = []
            model.init_book(book)
            model.advance(book, 0.0, C.HAWKES_WARMUP_S + C.WINDOW_S, [])
            w = R.S.visible_window(L.tape_to_lobster(book.tape, C.HAWKES_WARMUP_S, C.WINDOW_S))
        out.append(R.two_sided(w)[0])
    return out


def job(args) -> str:
    from simrel.metrics import lobbench

    version, name, value = args
    path = RAW / f"{version}_{name}_{value}.pkl"
    if path.exists():
        return "cached"
    R._memory_guard(path, 6.0)
    if not version.startswith("abides"):  # a scale that makes the process explosive is not a candidate
        params = R.hawkes_model(name, version, impact_scale=value, fit_day_starts=True)(SEED).p
        branching = float(np.max(np.abs(np.linalg.eigvals(np.asarray(params.alpha)))))
        if branching >= C.MAX_CALIB_BRANCHING:
            R.save(path, {"failed": f"unstable: branching ratio {branching:.2f}", "branching": branching})
            return f"{version} {name} {value}: unstable (branching {branching:.2f})"
    try:
        gen = sim_windows(version, name, value)
        fit = R.real_windows(name, "fit")
        keep = sorted(np.random.default_rng(2).permutation(len(fit))[: C.N_CALIB_WINDOWS])
        real = [fit[i] for i in keep][: len(gen)]
        imp = lobbench.impact_response([(w.message, w.book) for w in real], [(w.message, w.book) for w in gen], tempfile.mkdtemp())
        feats = [L.features(w) for w in gen]
        rec = {"impact": imp["mean"], "by_event": imp["by_event"], "events": float(np.mean([f["events"] for f in feats])),
               "volatility_bps": float(np.mean([f["volatility_bps"] for f in feats])), "real_events": float(np.mean([L.features(w)["events"] for w in real]))}
    except Exception as err:
        rec = {"failed": repr(err)[:300]}
    R.save(path, rec)
    return f"{version} {name} {value}: " + (f"impact {rec['impact']:.2f}" if "impact" in rec else rec["failed"][:80])


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    RAW.mkdir(parents=True, exist_ok=True)
    grid = {v: C.IMPACT_SCALES for v in C.IMPACT_CALIBRATE}
    grid["abides_quoting"] = C.ABIDES_MOMENTUM_GRID
    jobs = [(v, n, x) for v, xs in grid.items() for n in R.usable_slices() for x in xs]
    R.pool_map(job, [j for j in jobs if not (RAW / f"{j[0]}_{j[1]}_{j[2]}.pkl").exists()], procs, "impact calibration")
    rows = []
    for v, n, x in jobs:
        d = R.load(RAW / f"{v}_{n}_{x}.pkl")
        rows.append({"version": v, "slice": n, "value": x, **{k: d.get(k) for k in ("impact", "events", "volatility_bps", "real_events", "branching", "failed")}})
    t = pd.DataFrame(rows)
    t.to_csv(R.OUT / "impact_calibration.csv", index=False)
    # a candidate whose activity runs away (more than three times the real event count) is not eligible
    ok = t[t["impact"].notna() & (t["events"] < 3 * t["real_events"])]
    chosen = {v: {n: (float(g.loc[g["impact"].idxmin(), "value"]) if not v.startswith("abides") else int(g.loc[g["impact"].idxmin(), "value"]))
                  for n, g in gv.groupby("slice")} for v, gv in ok.groupby("version")}
    json.dump(chosen, open(R.OUT / "impact_calibration.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["version", "slice"], columns="value", values="impact").round(2).to_string())
    print(json.dumps(chosen, indent=1))
