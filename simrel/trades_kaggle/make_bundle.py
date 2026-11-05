"""
Build trades_bundle.zip: LOBSTER-format CSVs for one symbol, the DeepMarket patch and the training script.

    python simrel/trades_kaggle/make_bundle.py <itch npz dir> <out dir> [symbol]
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from simrel.data.lobster import visible, write_lobster_csv

if __name__ == "__main__":
    src, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    symbol = sys.argv[3] if len(sys.argv) > 3 else "INTC"
    days = sorted(p.stem.split("_")[1] for p in src.glob(f"{symbol}_*.npz"))
    stage = out / "trades_bundle"
    shutil.rmtree(stage, ignore_errors=True)
    data = stage / f"{symbol}_{days[0]}_{days[-1]}"
    rows = {}
    for d in days:
        z = np.load(src / f"{symbol}_{d}.npz")
        m, b = visible(z["message"], z["book"], z["book"][0])  # as in a ten-level LOBSTER file: only events that touch the visible book
        write_lobster_csv(data, symbol, d, m, b)
        rows[d] = int(len(m))
        print(d, rows[d], "events", flush=True)
    shutil.copy(HERE.parent.parent / "simbench" / "trades" / "deepmarket_simbench.patch", stage / "deepmarket_simbench.patch")
    shutil.copy(HERE / "train_trades_multiday.py", stage / "train_trades_multiday.py")
    (stage / "MANIFEST.json").write_text(json.dumps({"symbol": symbol, "days": rows, "source": "Nasdaq TotalView-ITCH 5.0 public sample files, https://emi.nasdaq.com/ITCH/",
                                                     "format": "LOBSTER message/orderbook, 10 levels, 09:30-16:00", "deepmarket_commit": "8f1f89b7285c79a73f528b88b60f74ce58faadc4"}, indent=2))
    shutil.make_archive(str(out / "trades_bundle"), "zip", root_dir=out, base_dir="trades_bundle")
    print("wrote", out / "trades_bundle.zip", round((out / "trades_bundle.zip").stat().st_size / 1e6), "MB")
