"""
LOB-Bench's own protocol on our INTC held-out days, so our simulators can be placed next to its published leaderboard.

LOB-Bench samples sequences from test days: 500 real messages of context, then the model generates 500 messages,
which are compared with the real next 500 (its public sample has exactly this shape). Here:

    real_other_days  the next 500 real messages after random points on the FIT days (a real-data reference)
    hawkes, ...      the Hawkes versions, started from the real book at the sequence start, with their excitation
                     state set from the 500 context messages (the model's own conditioning on history)
    abides, ...      ABIDES cannot be conditioned on a history: 500-message stretches of its own unperturbed windows
    trades           TRADES conditions on real history by design: its first 500 messages after 09:45 on each
                     held-out day (it replays the day until then), against that day's real messages after 09:45

    python simrel/experiments/slice_11_lobbench_native.py [--n 128]
"""

from __future__ import annotations

import dataclasses
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.adapters.hawkes import MB, MS  # noqa: E402
from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402
from simrel.lob import Book  # noqa: E402

N_CTX, N_GEN = 500, 500
HAWKES = ["hawkes", "hawkes_c_f_m"]
C.HAWKES_VERSIONS.setdefault("hawkes_c_f_m", {"close": True, "fill": True, "multi": True})  # same-day start is implied here
ABIDES = ["abides", "abides_quoting", "abides_quoting_mom"]


def day_windows(days):
    """Per day: the visible message and book arrays for the whole session."""
    out = {}
    for d in days:
        m, b = L.load_day(C.DATA_DIR, C.SYMBOL, d)
        w = L.cut_windows(C.SYMBOL, d, m, b, length_s=6 * 3600, first_s=9 * 3600 + 45 * 60, last_s=9 * 3600 + 46 * 60)[0]
        out[d] = R.S.visible_window(w)
    return out


def sample_points(days: dict, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    f = pd.read_csv(R.OUT / "windows.csv")
    pts = []
    keys = list(days)
    while len(pts) < n:
        d = keys[rng.integers(len(keys))]
        w = days[d]
        i = int(rng.integers(N_CTX + 1, len(w.message) - N_GEN - 1))
        s = int(w.message[i, 0] // L.NS)
        row = f[(f["date"] == d) & (f["start_s"] <= s) & (f["start_s"] + C.WINDOW_S > s)]
        if row.empty or row["slice"].iloc[0] not in R.usable_slices():
            continue
        pts.append((d, i, row["slice"].iloc[0]))
    return pts


def real_seq(w, i):
    return w.message[i: i + N_GEN], w.book[i: i + N_GEN]


def hawkes_continuation(version: str, name: str, w, i: int, seed: int):
    model = R.hawkes_model(name, version)(seed)
    ctx = L.Window(w.symbol, w.date, int(w.message[i - N_CTX, 0] // L.NS), C.WINDOW_S, w.message[i - N_CTX: i], w.book[i - N_CTX: i], w.book[i - N_CTX - 1])
    obs = L.to_observed(ctx)
    p = model.p
    orders, p0 = R.S.book_orders(w.book[i - 1])
    p = dataclasses.replace(p, init_orders=orders, p0=p0, ref_depth=max(0.5 * sum(q for _, _, q in orders), 1000.0))
    model = type(model)(p, seed)
    # excitation carried over from the context: kernel mass of each component and source at the end of the context
    betas, alphas = model._betas, model._alphas
    t, k, sz = obs.events[:, 0], obs.events[:, 1].astype(int), obs.events[:, 2]
    wgt = np.where((k == MB) | (k == MS), np.minimum((sz / p.mean_market_size()) ** p.weight_exponent, p.weight_cap), 1.0)
    end = (w.message[i - 1, 0] - ctx.start_s * L.NS) / L.NS
    a = np.zeros((len(betas), 6))
    for tn, kn, wn in zip(t, k, wgt):
        a[:, kn] += betas * wn * np.exp(-betas * (end - tn))
    model.e = np.einsum("mij,mj->mi", alphas, a)
    book = Book()
    book.tape = []
    model.init_book(book)
    tt = 0.0
    while True:  # simulate until 500 visible messages exist after the start
        model.advance(book, tt, tt + 5.0, [])
        tt += 5.0
        # the starting book is entered at time 0; the continuation is everything after it
        sim = L.tape_to_lobster(book.tape, 1e-6, tt, clock0_s=int(w.message[i - 1, 0] // L.NS))
        vm, vb = L.visible(sim.message, sim.book, sim.book_before)
        if len(vm) >= N_GEN or tt > 600:
            return vm[:N_GEN], vb[:N_GEN]


def abides_chunks(version: str, name: str, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    ws = [w for w in R.generated(version, name) if len(w.message) > N_GEN + 2]
    for _ in range(n):
        w = ws[rng.integers(len(ws))]
        j = int(rng.integers(0, len(w.message) - N_GEN))
        out.append((w.message[j: j + N_GEN], w.book[j: j + N_GEN]))
    return out


def score(real, gen, tag):
    from simrel.metrics import lobbench

    gen = [(m, b) for m, b in gen if len(m) == N_GEN and (b[:, 0] < L.EMPTY).all() and (b[:, 2] > -L.EMPTY).all()]
    real = real[: len(gen)]
    t = lobbench.score(real, gen)
    agg = lobbench.aggregate(t)
    print(f"{tag:22s} sequences {len(gen):4d}  L1 mean {agg['l1_mean']:.3f}  Wasserstein mean {agg['wasserstein_mean']:.3f}", flush=True)
    return {"source": tag, "n": len(gen), **agg}, t.assign(source=tag)


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 128
    test, fit = day_windows(C.TEST_DAYS), day_windows(C.FIT_DAYS)
    pts = sample_points(test, n, 1)
    real = [real_seq(test[d], i) for d, i, _ in pts]
    rows, tabs = [], []

    ref_pts = sample_points(fit, n, 2)
    r, t = score(real, [real_seq(fit[d], i) for d, i, _ in ref_pts], "real_other_days")
    rows.append(r), tabs.append(t)
    for v in HAWKES:
        gen = [hawkes_continuation(v, s, test[d], i, 7000 + k) for k, (d, i, s) in enumerate(pts)]
        r, t = score(real, gen, v)
        rows.append(r), tabs.append(t)
    for v in ABIDES:
        gen = []
        for s in sorted({p[2] for p in pts}):
            gen += abides_chunks(v, s, sum(p[2] == s for p in pts), 11)
        r, t = score(real, gen, v)
        rows.append(r), tabs.append(t)
    # TRADES: its first 500 messages after 09:45 on each held-out day, against that day's real next 500
    for tag, d_ in [("trades_1step", "~/simdata/trades_runs"), ("trades_10step", "~/simdata/trades_runs/kaggle_ddim10_v2/runs")]:
        from simrel.experiments import slice_03_trades_score as T

        gen, rl = [], []
        for p in sorted(glob.glob(str(Path(d_).expanduser() / "gen_*.pkl"))):
            day = Path(p).name.split("_")[1]
            rec = pd.read_pickle(p)
            w = L.tape_to_lobster(rec["tape"], 9 * 3600 + 45 * 60, 120, clock0_s=0, price_unit=1)
            vm, vb = L.visible(w.message, w.book, w.book_before)
            if len(vm) >= N_GEN:
                gen.append((vm[:N_GEN], vb[:N_GEN]))
                tw = test[day]
                j = int(np.searchsorted(tw.message[:, 0], (9 * 3600 + 45 * 60) * L.NS))
                rl.append(real_seq(tw, j))
        if gen:
            r, t = score(rl, gen, tag)
            rows.append(r), tabs.append(t)
    res = pd.DataFrame(rows)
    res.to_csv(R.OUT / "lobbench_native.csv", index=False)
    pd.concat(tabs).to_csv(R.OUT / "lobbench_native_by_statistic.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
