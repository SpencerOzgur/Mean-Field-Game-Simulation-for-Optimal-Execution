"""
Realism of the TRADES simulator, scored like the others.

TRADES replays the first fifteen minutes of a real day and then generates, so every episode covers the same clock
window (09:46 to 09:56). Its unperturbed episodes on held-out days are converted to LOBSTER format and scored with
LOB-Bench against the real 09:46-09:56 windows of those days. For comparison, the same real windows are used to score
(a) real windows at the same time on the fit days, (b) the Hawkes and ABIDES windows of the slice those windows fall in.

    python simrel/experiments/slice_03_trades_score.py <dir with gen_<date>_s<seed>.pkl> [output tag]
"""

from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel import slices as S  # noqa: E402
from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402
from simrel.metrics import lobbench  # noqa: E402

START_S, LENGTH_S = 9 * 3600 + 46 * 60, 600
KEYS = ["volatility_bps", "spread_bps", "volume_visible", "events", "depth10", "trend_bps"]


def real_window(date: str) -> L.Window:
    m, b = L.load_day(C.DATA_DIR, C.SYMBOL, date)
    a, z = np.searchsorted(m[:, 0], [START_S * L.NS, (START_S + LENGTH_S) * L.NS])
    return S.visible_window(L.Window(C.SYMBOL, date, START_S, LENGTH_S, m[a:z], b[a:z], b[a - 1]))


def trades_window(path: str) -> L.Window:
    rec = pickle.load(open(path, "rb"))
    w = L.tape_to_lobster(rec["tape"], rec["task_start_s"], LENGTH_S, clock0_s=0, price_unit=1, symbol="TRADES")
    return S.visible_window(w)


if __name__ == "__main__":
    out = S.out_dir()
    tag = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
    R._memory_guard(out / "raw" / f"trades_score_guard{tag}.pkl", 6.0)
    paths = sorted(glob.glob(str(Path(sys.argv[1]).expanduser() / "gen_*.pkl")))
    held_out = sorted({Path(p).name.split("_")[1] for p in paths})
    real = [real_window(d) for d in held_out]
    f = pd.read_csv(out / "windows.csv")
    labels = [f[(f["date"] == d) & (f["start_s"] == 9 * 3600 + 45 * 60)]["slice"].iloc[0] for d in held_out]
    print("held-out days", held_out, "slices of their 09:45 windows", labels, flush=True)
    sets = {"trades": [], "real_fit_days": [real_window(d) for d in C.FIT_DAYS]}
    empty = {}
    for p in paths:
        w, share = R.two_sided(trades_window(p))
        empty[p] = share
        if len(w.message) > 50:  # an episode whose book has one side empty throughout cannot be scored
            sets["trades"].append(w)
    print(f"TRADES episodes scored: {len(sets['trades'])} of {len(paths)}", flush=True)
    name = max(set(labels), key=labels.count)
    for sim in ("hawkes", "abides"):
        sets[sim] = [R.two_sided(w)[0] for w in R.generated(sim, name)]
    rows, tables = [], []
    for src, ws in sets.items():
        table = lobbench.score([(w.message, w.book) for w in real], [(w.message, w.book) for w in ws])
        table.insert(0, "source", src)
        tables.append(table)
        feats = [L.features(w) for w in ws]
        rows.append({"source": src, "n_windows": len(ws), **lobbench.aggregate(table), **{k: float(np.mean([x[k] for x in feats])) for k in KEYS}})
        print(src, "L1", round(rows[-1]["l1_mean"], 3), flush=True)
    rf = [L.features(w) for w in real]
    rows.append({"source": "real_held_out", "n_windows": len(real), **{k: float(np.mean([x[k] for x in rf])) for k in KEYS}})
    res = pd.DataFrame(rows)
    res.to_csv(out / f"trades_realism{tag}.csv", index=False)
    pd.concat(tables).to_csv(out / f"trades_realism_by_statistic{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(pd.concat(tables).pivot_table(index="statistic", columns="source", values="l1").round(2).to_string())
    print("share of TRADES events with one side of the book empty: mean %.4f max %.4f" % (np.mean(list(empty.values())), np.max(list(empty.values()))))
    print("Hawkes and ABIDES windows are those of slice", name)
