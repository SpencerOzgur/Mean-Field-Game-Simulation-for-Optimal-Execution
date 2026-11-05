"""
Re-export the cross-check rollouts so that every rollout has its own day (the reference tool reads one child-order
schedule per day and side; our rollouts on the same day have different schedules). Each rollout gets a placeholder
date, and the daily statistics file gives that date the volume and price range of the rollout's real day.

    python simrel/experiments/impact_05_unique_days.py
"""

import datetime as dt
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "simrel" / "lob_impact" / "INTC"
SRC, DST = BASE / "csv", BASE / "csv_unique"
DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    daily = pd.read_csv(BASE / "daily_stats.csv").set_index("day")
    rows, k = [], 0
    for exp in sorted(p for p in SRC.rglob("exp_*") if p.is_dir()):
        gen = exp / "data_gen"
        msgs = sorted(gen.glob("*message*gen*.csv"))
        if not msgs:
            continue
        real_day = DATE.search(msgs[0].name).group(1)
        fake = (dt.date(2001, 1, 1) + dt.timedelta(days=k)).isoformat()
        k += 1
        out = DST / exp.relative_to(SRC)
        (out / "data_gen").mkdir(parents=True, exist_ok=True)
        for f in gen.glob("*.csv"):
            shutil.copy2(f, out / "data_gen" / f.name.replace(real_day, fake))
        for f in exp.glob("aggressive_indices_*.csv"):
            shutil.copy2(f, out / f.name.replace(real_day, fake))
        rows.append({"ticker": "INTC", "day": fake, **daily.loc[real_day].drop("ticker").to_dict(), "real_day": real_day})
    pd.DataFrame(rows).to_csv(BASE / "daily_stats_unique.csv", index=False)
    print(f"{k} rollouts re-exported to {DST}")


if __name__ == "__main__":
    main()
