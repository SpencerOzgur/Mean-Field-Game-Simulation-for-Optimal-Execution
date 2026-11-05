"""
Write INTC sequences in LOB-Bench's file layout, so LOB-Bench's own baseline model (Cont-Stoikov-Talreja, in
lob_bench/cst_model) can be fitted on the fit days and generate continuations for the same held-out sequences used in
slice_11_lobbench_native.py.

    python simrel/experiments/slice_12_cst_baseline_data.py <out dir>

Creates <out>/fit_cond (parameter estimation), <out>/test_cond (500 context messages) and <out>/test_real (the real
next 500 messages), with names <SYM>_<date>_{message,orderbook}_real_id_<k>.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_11_lobbench_native as N  # noqa: E402


def write(d: Path, date: str, k: int, m: np.ndarray, b: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    msg = pd.DataFrame(m, columns=["time", "type", "id", "size", "price", "direction"])
    msg["time"] = (msg["time"] // L.NS).astype(str) + "." + (msg["time"] % L.NS).astype(str).str.zfill(9)
    msg.to_csv(d / f"{C.SYMBOL}_{date}_message_real_id_{k}.csv", header=False, index=False)
    pd.DataFrame(b).to_csv(d / f"{C.SYMBOL}_{date}_orderbook_real_id_{k}.csv", header=False, index=False)


if __name__ == "__main__":
    out = Path(sys.argv[1]).expanduser()
    test, fit = N.day_windows(C.TEST_DAYS), N.day_windows(C.FIT_DAYS)
    for k, (d, i, _) in enumerate(N.sample_points(test, 128, 1)):  # the same points as the native benchmark
        w = test[d]
        write(out / "test_cond", d, k, w.message[i - N.N_CTX: i], w.book[i - N.N_CTX: i])
        write(out / "test_real", d, k, w.message[i: i + N.N_GEN], w.book[i: i + N.N_GEN])
    for k, (d, i, _) in enumerate(N.sample_points(fit, 128, 3)):
        w = fit[d]
        write(out / "fit_cond", d, k, w.message[i - N.N_CTX: i], w.book[i - N.N_CTX: i])
    print("written to", out)
