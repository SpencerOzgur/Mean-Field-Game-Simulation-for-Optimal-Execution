"""
Rebuild summary.csv and battery.png from the per-scenario folders of a battery run,
without re-simulating. Useful when a later --only run overwrote the combined files.

Usage:  python -m abides_exp.rebuild_summary results/battery
"""
import os
import sys

import pandas as pd

from abides_exp.battery import plot, summarize
from abides_exp.scenarios import SCENARIOS


def main(folder: str):
    names = [n for n in SCENARIOS if os.path.exists(os.path.join(folder, n, "runs.csv"))]
    if not names:
        raise SystemExit(f"no scenario folders with runs.csv in {folder}")
    runs, paths, info = [], {}, {}
    for n in names:
        r = pd.read_csv(os.path.join(folder, n, "runs.csv"))
        r["scenario"] = n
        runs.append(r)
        paths[n] = pd.read_csv(os.path.join(folder, n, "paths.csv"))
        _, h = SCENARIOS[n](0, True, {}, None)  # builds the config only, to recover the window
        info[n] = {"window": h["window"], "kind": h["kind"], "meta": h["meta"]}
        print(f"{n:20s} {r.seed.nunique():4d} seeds")
    summ = summarize(pd.concat(runs, ignore_index=True))
    summ.to_csv(os.path.join(folder, "summary.csv"), index=False)
    plot(folder, paths, info, summ)
    print(f"\nwrote {folder}/summary.csv and {folder}/battery.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/battery")
