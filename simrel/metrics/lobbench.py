"""
Realism scoring with LOB-Bench (github.com/peernagy/lob_bench).

Nothing is re-implemented here: every statistic and both distances (L1 over
histogram bins, Wasserstein on standardised values) are LOB-Bench's own,
called through its `scoring.run_benchmark` with its default configuration.
This module only builds the objects LOB-Bench expects from in-memory LOBSTER
arrays, so that no CSV round trip is needed, and reduces its output to a table.

A "set" is a list of (message, book) pairs, one per ten-minute window:

    message  N x 6   time (ns after midnight), type, order id, size, price, direction
    book     N x 40  ask price 1, ask size 1, bid price 1, bid size 1, ...

LOB-Bench pools each statistic over all real windows and over all generated
windows and compares the two pooled distributions, so the two sets need not
have the same number of windows.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

LOB_BENCH_DIR = os.path.expanduser(os.environ.get("LOB_BENCH_DIR", "~/simdata/lob_bench"))
COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


def _modules():
    if LOB_BENCH_DIR not in sys.path:
        sys.path.insert(0, LOB_BENCH_DIR)  # LOB-Bench modules import each other as top-level names
    import data_loading
    import run_bench
    import scoring

    return data_loading, run_bench, scoring


def frames(message: np.ndarray, book: np.ndarray, date: str = "2019-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.DataFrame(message, columns=COLS)
    m["time"] = pd.Timestamp(date) + pd.to_timedelta(message[:, 0], unit="ns")
    m["order_id"] = m["order_id"] % (2**31 - 1)
    return m, pd.DataFrame(book)


def price_dynamics_config() -> dict:
    """Price statistics that LOB-Bench's default configuration leaves out, built from LOB-Bench's own functions
    (eval.mid_returns, eval.volatility) and scored through the same machinery. Returns are in basis points."""
    _modules()
    import eval as lb_eval

    return {
        "mid_return_5s": {"fn": lambda m, b: lb_eval.mid_returns(m, b, "5s").dropna().values * 1e4},
        "mid_return_1min": {"fn": lambda m, b: lb_eval.mid_returns(m, b, "1min").dropna().values * 1e4},
        "abs_mid_return_5s": {"fn": lambda m, b: np.abs(lb_eval.mid_returns(m, b, "5s").dropna().values) * 1e4},
        "volatility_5s": {"fn": lambda m, b: np.array([lb_eval.volatility(m, b, "5s") * 1e4])},
        "window_return": {"fn": lambda m, b: np.array([float(np.log(lb_eval.mid_price(m, b).iloc[-1] / lb_eval.mid_price(m, b).iloc[0])) * 1e4])},
    }


def score(real: list, generated: list, quiet: bool = True, config: dict | None = None) -> pd.DataFrame:
    """One row per LOB-Bench statistic: L1 and Wasserstein distance with LOB-Bench's bootstrap intervals.

    config defaults to LOB-Bench's DEFAULT_SCORING_CONFIG (21 statistics)."""
    data_loading, run_bench, scoring = _modules()
    real_f = [frames(m, b) for m, b in real]
    gen_f = [frames(m, b) for m, b in generated]
    seqs = []
    for i, (m, b) in enumerate(real_f):
        mine = gen_f[i::len(real_f)] if i < len(gen_f) else []
        seqs.append(data_loading.Lobster_Sequence("2019-01-01", i, m, b, num_gen_series=(len(mine),),
                                                  m_gen=tuple(x[0] for x in mine) or None, b_gen=tuple(x[1] for x in mine) or None))
    if len(gen_f) < len(real_f):  # every LOB-Bench sequence carries one real series and at least one generated series
        raise ValueError("need at least as many generated windows as real windows")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout):
        warnings.simplefilter("ignore")
        scores, _, _ = scoring.run_benchmark(seqs, config or run_bench.DEFAULT_SCORING_CONFIG, run_bench.DEFAULT_METRICS)
    rows = []
    for stat, by_metric in scores.items():
        row = {"statistic": stat}
        for name, (value, ci, _) in by_metric.items():
            row[name], row[name + "_lo"], row[name + "_hi"] = float(value), float(ci[0]), float(ci[1])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(table: pd.DataFrame) -> dict:
    """LOB-Bench's summary: mean, median and interquartile mean across statistics, for each distance."""
    out = {}
    for name in ("l1", "wasserstein"):
        v = np.sort(table[name].dropna().to_numpy())
        n = len(v)
        out[name + "_mean"], out[name + "_median"] = float(v.mean()), float(np.median(v))
        out[name + "_iqm"] = float(v[n // 4: n - n // 4].mean())
    out["n_statistics"] = int(len(table))
    return out


def _sequences(real: list, generated: list) -> list:
    data_loading, _, _ = _modules()
    real_f = [frames(m, b) for m, b in real]
    gen_f = [frames(m, b) for m, b in generated]
    if len(gen_f) < len(real_f):
        raise ValueError("need at least as many generated windows as real windows")
    seqs = []
    for i, (m, b) in enumerate(real_f):
        mine = gen_f[i::len(real_f)]
        seqs.append(data_loading.Lobster_Sequence("2019-01-01", i, m, b, num_gen_series=(len(mine),),
                                                  m_gen=tuple(x[0] for x in mine), b_gen=tuple(x[1] for x in mine)))
    return seqs


def score_conditional(real: list, generated: list, quiet: bool = True) -> pd.DataFrame:
    """LOB-Bench's conditional statistics (DEFAULT_SCORING_CONFIG_COND), with its default metrics."""
    _, run_bench, _ = _modules()
    return score(real, generated, quiet=quiet, config=run_bench.DEFAULT_SCORING_CONFIG_COND)


def impact_response(real: list, generated: list, out_dir: str, quiet: bool = True) -> dict:
    """LOB-Bench's impact-response evaluation (impact.impact_compare): price-response curves after six event types
    (market order, limit order, cancel; each with and without a mid-price change), at lags of 1 to about 200 events,
    for real and generated data. Its score per event type is the sum over lags of the absolute difference between
    the real and generated mean response, in ticks; "mean" is LOB-Bench's Mean_Diffs."""
    _modules()
    import impact

    seqs = _sequences(real, generated)
    # impact_compare re-filters and re-classifies every sequence for each of its 6 x 21 (event, lag) pairs; both steps
    # are deterministic functions of the sequence, so they are computed once per sequence (identical results, ~100x faster)
    orig_filter, orig_classify = impact._filter_touch_events, impact._classify_event_types
    cache_f, cache_c = {}, {}

    def filt(m, b, *a, **k):
        if id(m) not in cache_f:
            cache_f[id(m)] = (m, orig_filter(m, b, *a, **k))
        return cache_f[id(m)][1]

    def classify(df):
        if id(df) not in cache_c:
            cache_c[id(df)] = (df, orig_classify(df))
        return cache_c[id(df)][1]

    for sq in seqs:  # materialise the lazily loaded frames so their identities are stable
        sq.m_real, sq.b_real = sq.m_real, sq.b_real
        sq.m_gen, sq.b_gen = tuple(sq.m_gen), tuple(sq.b_gen)
    impact._filter_touch_events, impact._classify_event_types = filt, classify
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout), \
                contextlib.redirect_stderr(io.StringIO() if quiet else sys.stderr):
            warnings.simplefilter("ignore")
            res = impact.impact_compare(seqs, ticker="INTC", model="sim", ticksize=100, save_dir=out_dir)
    finally:
        impact._filter_touch_events, impact._classify_event_types = orig_filter, orig_classify
    lags, ys_real, ys_gen, _, _, diffs = res["Saved_Tuple"]
    events = ["MO_0", "MO_1", "LO_0", "LO_1", "CA_0", "CA_1"]
    return {"mean": float(res["Mean_Diffs"]), "by_event": dict(zip(events, [float(d) for d in diffs])),
            "lags": np.asarray(lags), "real": [np.asarray(y, dtype=float) for y in ys_real], "gen": [np.asarray(y, dtype=float) for y in ys_gen]}
