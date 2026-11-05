"""
Check that our LOB-Bench wrapper (simrel.metrics.lobbench) gives the same scores as LOB-Bench's own loader and
benchmark on LOB-Bench's public sample (GOOG 2012-06-21, 128 sequences of 500 real and 500 generated messages).

    python simrel/experiments/slice_10_lobbench_reproduce.py <sample dir containing data_real, data_gen, data_cond>
"""

from __future__ import annotations

import contextlib
import glob
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.metrics import lobbench  # noqa: E402


def native(d: str) -> pd.DataFrame:
    data_loading, run_bench, scoring = lobbench._modules()
    loader = data_loading.Simple_Loader(d + "/data_real", d + "/data_gen", d + "/data_cond")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        scores, _, _ = scoring.run_benchmark(loader, run_bench.DEFAULT_SCORING_CONFIG, run_bench.DEFAULT_METRICS)
    return pd.DataFrame([{"statistic": s, **{m: float(v[0]) for m, v in d_.items()}} for s, d_ in scores.items()])


def arrays(path: str, date: str) -> np.ndarray:
    m = pd.read_csv(path, header=None).to_numpy()
    sec = np.array([str(x).split(".") for x in m[:, 0]], dtype=object)
    ns = np.array([int(a) * 10**9 + int((b + "000000000")[:9]) for a, b in sec], dtype=np.int64)
    return np.column_stack([ns, m[:, 1:].astype(np.int64)])


def ours(d: str) -> pd.DataFrame:
    real, gen = [], []
    for mp in sorted(glob.glob(d + "/data_real/*message*.csv")):
        rid = mp.split("real_id_")[1].split(".")[0]
        bp = mp.replace("message", "orderbook")
        real.append((arrays(mp, ""), pd.read_csv(bp, header=None).to_numpy()))
        for gp in sorted(glob.glob(d + f"/data_gen/*message*real_id_{rid}_gen_id_*.csv")):
            gen.append((arrays(gp, ""), pd.read_csv(gp.replace("message", "orderbook"), header=None).to_numpy()))
    with contextlib.redirect_stderr(io.StringIO()):
        return lobbench.score(real, gen)[["statistic", "l1", "wasserstein"]]


if __name__ == "__main__":
    d = sys.argv[1]
    a = native(d).set_index("statistic")
    b = ours(d).set_index("statistic")
    t = a.join(b, lsuffix="_lobbench", rsuffix="_ours")
    t["l1_diff"] = (t["l1_lobbench"] - t["l1_ours"]).abs()
    t["w_diff"] = (t["wasserstein_lobbench"] - t["wasserstein_ours"]).abs()
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())
    print("mean L1: LOB-Bench %.4f, ours %.4f | max abs difference L1 %.2e, Wasserstein %.2e" % (
        t["l1_lobbench"].mean(), t["l1_ours"].mean(), t["l1_diff"].max(), t["w_diff"].max()))
    t.to_csv(ROOT / "results" / "simrel" / "slices" / "lobbench_reproduction.csv")
