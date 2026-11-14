"""
Reversal check (simrel/reports/reversal_check_plan.md): is the advantage of P6_signal_reversal in the reference
markets present in real data? Descriptive.

    python simrel/experiments/reversal_check.py markets          write the list of simulated markets (pilot parameters
                                                                 are read from the pilot branches with git show)
    python simrel/experiments/reversal_check.py sim [procs]      simulated side: passive episodes (S1 to S4) and truth
                                                                 runs (S5); writes sim_table.csv
    python simrel/experiments/reversal_check.py files            size and message count of every real data file
    python simrel/experiments/reversal_check.py real [procs]     real side (S1 to S4); writes real_table.csv
    python simrel/experiments/reversal_check.py report           the combined tables

No Test B market is read or simulated: reference_worlds.data.registry() (which builds the Test B draws) is not
called; worlds are built directly from their parameter classes.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import reference as C  # noqa: E402
from simrel.configs import slices as CS  # noqa: E402
from simrel.core import Task  # noqa: E402
from simrel.eventsim import GRID_DT, run_twin_set  # noqa: E402
from simrel.lob import Book  # noqa: E402
from simrel.metrics import reversal as RV  # noqa: E402
from simrel.metrics.cost import episode_metrics  # noqa: E402
from simrel.policies.library import candidate_set  # noqa: E402
from simrel.reference_worlds import agent_mix, latent, queue_reactive  # noqa: E402

OUT = ROOT / "results" / "simrel" / "goal2" / "reversal_check"
RAW = OUT / "raw"
SEED_PASSIVE, N_PASSIVE, N_TRUTH, N_BOOT = 70_000, 20, 100, 2000
WARMUP_S, EPISODE_S = 600.0, 3600.0
SYMBOLS = ["INTC", "CSCO", "MSFT"]
DAYS = CS.FIT_DAYS + CS.TEST_DAYS
DAY_START_S, DAY_LEN_S = 9 * 3600 + 45 * 60, 6 * 3600  # 09:45 to 15:45
FAMILY = {"qr": "queue-reactive", "am": "agent-mix", "latent": "latent"}


# ---- markets
def write_markets() -> None:
    def show(ref: str, path: str) -> dict:
        return json.loads(subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout)["markets"]

    out = {}
    for k, p in queue_reactive.regimes().items():
        out[k] = {"family": "qr", "source": "queue_reactive.regimes(): hand-built development market", "params": p.__dict__}
    for k, p in agent_mix.regimes().items():
        out[k] = {"family": "am", "source": "agent_mix.regimes(): hand-built development market", "params": p.__dict__}
    for k, v in show("origin/reactive-family", "results/simrel/goal2/addendum11_pilot/pilot_markets.json").items():
        out[k] = {"family": v["family"], "source": "addendum 11 development pilot (stream 20261120), predator off; origin/reactive-family", "params": v["params"]}
    for k, v in show("origin/lambda-family", "results/simrel/goal2/lambda_pilot/pilot_markets.json").items():
        out[k] = {"family": v["family"], "source": "lambda development pilot (stream 20261130), lambda = 0; origin/lambda-family", "params": v["params"]}
    assert not any("testb" in k.lower() or k.startswith(("B", "TB")) for k in out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "markets.json").write_text(json.dumps(out, indent=2))
    print(len(out), "markets:", {f: sum(1 for v in out.values() if v["family"] == f) for f in FAMILY})


def markets() -> dict:
    return json.loads((OUT / "markets.json").read_text())


def make(market: str):
    m = markets()[market]
    if m["family"] == "qr":
        p = queue_reactive.QRParams(**m["params"])
        return lambda seed: queue_reactive.QueueReactive(p, seed)
    if m["family"] == "am":
        p = agent_mix.AMParams(**m["params"])
        return lambda seed: agent_mix.AgentMix(p, seed)
    p = latent.LatentParams(**m["params"])
    return lambda seed: latent.LatentLiquidity(p, seed)


# ---- simulated side
def passive_job(args):
    """One passive episode: 600 s warm-up, then a 3,600 s segment on the 5 s quote grid, and its summary."""
    market, seed = args
    warnings.filterwarnings("ignore")
    model = make(market)(seed)
    book, tape, t = Book(), [], 0.0
    model.init_book(book)
    bid, ask = [], []
    end = WARMUP_S + EPISODE_S
    while t < end + 1e-9:
        if t >= WARMUP_S - 1e-9:
            bb, ba = book.best_bid(), book.best_ask()
            bid.append(float(bb) if bb is not None else np.nan)
            ask.append(float(ba) if ba is not None else np.nan)
        if t >= end - 1e-9:
            break
        model.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    tr = np.array([(x[0] - WARMUP_S, x[1]) for x in tape if x[0] >= WARMUP_S], dtype=float).reshape(-1, 2)
    s = RV.segment_summary(np.array(bid), np.array(ask), tr, 1.0)
    s.update({"market": market, "seed": seed})
    return s


class Spy:
    """Requests nothing and records each observation. The harness sends the whole order at the last decision, after
    the record is complete, so the path it sees is the unperturbed twin's."""

    def __init__(self) -> None:
        self.obs = []

    def __call__(self, obs, remaining, time_remaining, state) -> float:
        self.obs.append(obs)
        return 0.0


def truth_job(args):
    """Twin, TWAP, P6 and the spy on one seed: the true cost gap and the impact-free backtests on the twin's path."""
    market, qty, seed = args
    warnings.filterwarnings("ignore")
    t = C.TASK
    pol = candidate_set(RV.SIZE_FRAC)
    spy = Spy()
    row = {"market": market, "seed": seed, "qty": qty}
    try:
        recs = run_twin_set(make(market), market, seed, Task(t.side, qty, t.horizon_s, t.n_slices, t.post_s),
                            {"P0_twap": pol["P0_twap"], "P6_signal_reversal": pol["P6_signal_reversal"], "spy": spy})
        tw, p6 = episode_metrics(recs["P0_twap"], recs["none"]), episode_metrics(recs["P6_signal_reversal"], recs["none"])
        row.update({"twap_is": tw["is_bps"], "p6_is": p6["is_bps"], "gap_true": p6["is_bps"] - tw["is_bps"],
                    "twap_impact": tw.get("impact_bps", np.nan), "p6_impact": p6.get("impact_bps", np.nan),
                    "p6_first_half_share": float(recs["P6_signal_reversal"].fill_qty[:10].sum() / qty),
                    "twap_first_half_share": float(recs["P0_twap"].fill_qty[:10].sum() / qty)})
        o = spy.obs  # 19 observations: the harness does not call the policy at the last decision
        mid = np.array([x.mid for x in o] + [recs["none"].slice_mid[-1]])
        vol = np.array([x.market_volume_last for x in o] + [0.0])
        sgn = np.array([x.signed_volume_last for x in o] + [0.0])
        assert len(o) == t.n_slices - 1 and np.allclose(mid[:-1], recs["none"].slice_mid[:-1])
        a, b = RV.backtest(mid, vol, sgn, qty, "sell", False), RV.backtest(mid, vol, sgn, qty, "sell", True)
        row.update({"s4_twin": a["p6"] - a["twap"], "s4b_twin": b["p6"] - b["twap"],
                    "p6_first_half_share_impact_free": float(sum(a["p6_sched"][:10]) / qty),
                    "p6_first_half_share_own_flow": float(sum(b["p6_sched"][:10]) / qty)})
    except Exception as err:
        row["failed"] = repr(err)[:200]
    return row


def cached(name: str, make_fn):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    val = make_fn()
    tmp = path.with_suffix(".tmp")
    pickle.dump(val, open(tmp, "wb"))
    tmp.replace(path)
    return val


def _ci(x) -> tuple[float, float]:
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def stats_of(segs: list, gaps: list) -> dict:
    """S1 to S4 of a set of segments; gaps[i] are the window gaps of segs[i]."""
    g = {k: np.concatenate([x[k] for x in gaps]) for k in ("s4", "s4b", "s4_buy")}
    return {"S1": RV.s1(segs), "S2": RV.s2(segs), "S3_slope": RV.s3_slope(segs), "S4": float(g["s4"].mean()), "S4b": float(g["s4b"].mean()),
            "S4_buy": float(g["s4_buy"].mean())}


def describe(segs: list, gaps: list, groups: list | None, rng) -> dict:
    """Point estimates with bootstrap 95% intervals. Without groups the segments are resampled (days or seeds); with
    groups (the market of each segment) markets are resampled and segments within each drawn market."""
    out = stats_of(segs, gaps)
    out["S3_t"] = RV.s3(segs)[1]
    g = np.concatenate([x["s4"] for x in gaps])
    out.update({"S4_se_windows": float(g.std(ddof=1) / np.sqrt(len(g))), "S4_share_p6_cheaper": float((g < 0).mean()), "S4_share_tie": float((g == 0).mean()),
                "n_segments": len(segs), "n_windows": int(len(g))})
    idx = np.arange(len(segs))
    boots = []
    for _ in range(N_BOOT):
        if groups is None:
            pick = rng.choice(idx, len(idx))
        else:
            names = sorted(set(groups))
            pick = np.concatenate([rng.choice(idx[np.array(groups) == m], int((np.array(groups) == m).sum())) for m in rng.choice(names, len(names))])
        boots.append(stats_of([segs[i] for i in pick], [gaps[i] for i in pick]))
    b = pd.DataFrame(boots)
    for k in ("S1", "S2", "S3_slope", "S4", "S4b", "S4_buy"):
        out[k + "_lo"], out[k + "_hi"] = _ci(b[k])
    return out


def sim(procs: int) -> None:
    from simrel.experiments.run_reference import pool_map

    M = markets()
    segs = cached("sim_passive", lambda: pool_map(passive_job, [(m, SEED_PASSIVE + s) for m in M for s in range(N_PASSIVE)], procs, "passive"))
    by = {m: sorted([s for s in segs if s["market"] == m], key=lambda s: s["seed"]) for m in M}
    qty = {m: RV.parent_qty(by[m]) for m in M}
    truth = pd.DataFrame(cached("sim_truth", lambda: pool_map(truth_job, [(m, qty[m], s) for m in M for s in range(N_TRUTH)], procs, "truth")))
    if "failed" in truth:
        print("failed truth runs:", int(truth["failed"].notna().sum()))
        truth = truth[truth["failed"].isna()]
    gaps = {m: [RV.window_gaps(s, qty[m]) for s in by[m]] for m in M}
    rng = np.random.default_rng(0)
    rows = []

    def s5(t: pd.DataFrame, groups: bool) -> dict:
        def point(d):
            return {"G_true_gap": d["gap_true"].mean(), "S4_twin": d["s4_twin"].mean(), "S4b_twin": d["s4b_twin"].mean(),
                    "S5": (d["gap_true"] - d["s4_twin"]).mean(), "S5_own_flow_timing": (d["s4b_twin"] - d["s4_twin"]).mean(),
                    "S5_price_response": (d["gap_true"] - d["s4b_twin"]).mean()}

        out = point(t)
        boots = []
        for _ in range(N_BOOT):
            if groups:
                ms = rng.choice(sorted(t["market"].unique()), t["market"].nunique())
                d = pd.concat([t[t["market"] == m].sample(frac=1.0, replace=True, random_state=int(rng.integers(1 << 31))) for m in ms])
            else:
                d = t.sample(frac=1.0, replace=True, random_state=int(rng.integers(1 << 31)))
            boots.append(point(d))
        b = pd.DataFrame(boots)
        for k in list(out):
            out[k + "_lo"], out[k + "_hi"] = _ci(b[k])
        out.update({"twap_cost": t["twap_is"].mean(), "p6_cost": t["p6_is"].mean(), "n_truth": len(t), "p6_first_half_share": t["p6_first_half_share"].mean(),
                    "p6_first_half_share_impact_free": t["p6_first_half_share_impact_free"].mean(),
                    "p6_first_half_share_own_flow": t["p6_first_half_share_own_flow"].mean()})
        return out

    for fam in FAMILY:
        ms = [m for m in M if M[m]["family"] == fam]
        allsegs = [s for m in ms for s in by[m]]
        row = {"side": "simulated", "family": FAMILY[fam], "unit": "pooled", "n_markets": len(ms)}
        row.update(describe(allsegs, [g for m in ms for g in gaps[m]], [s["market"] for s in allsegs], rng))
        row.update(RV.context(allsegs))
        row.update(s5(truth[truth["market"].isin(ms)], True))
        rows.append(row)
        for m in ms:
            row = {"side": "simulated", "family": FAMILY[fam], "unit": m, "n_markets": 1, "qty": qty[m]}
            row.update(describe(by[m], gaps[m], None, rng))
            row.update(RV.context(by[m]))
            row.update(s5(truth[truth["market"] == m], False))
            rows.append(row)
        print(fam, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "sim_table.csv", index=False)


# ---- real side
def day_segment(symbol: str, day: str) -> dict:
    from simrel.data import lobster as L

    msg, book = L.load_day(CS.DATA_DIR, symbol, day)
    t = msg[:, 0]
    a, b = np.searchsorted(t, DAY_START_S * L.NS, side="left"), np.searchsorted(t, (DAY_START_S + DAY_LEN_S) * L.NS, side="left")
    w = L.Window(symbol, day, DAY_START_S, DAY_LEN_S, msg[a:b], book[a:b], book[a - 1])
    _, bid, ask = L.grid_quotes(w)
    bad = (ask >= L.EMPTY) | (bid <= -L.EMPTY) | (ask <= 0) | (bid <= 0)
    bid, ask = np.where(bad, np.nan, bid), np.where(bad, np.nan, ask)
    ex = w.message[(w.message[:, 1] == 4) | (w.message[:, 1] == 5)]
    tr = np.c_[(ex[:, 0] - DAY_START_S * L.NS) / L.NS, -ex[:, 5] * ex[:, 3]].astype(float)
    s = RV.segment_summary(bid, ask, tr, float(L.TICK))
    s.update({"symbol": symbol, "day": day, "n_messages_segment": int(b - a), "n_executions_segment": int(len(ex))})
    return s


def files() -> pd.DataFrame:
    d = Path(CS.DATA_DIR).expanduser()
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    rows = []
    for s in SYMBOLS:
        for day in DAYS:
            p = d / f"{s}_{day}.npz"
            row = {"file": p.name, "exists": p.exists()}
            if p.exists():
                z = np.load(p)
                m = z["message"]
                row.update({"bytes": p.stat().st_size, "messages": int(len(m)), "book_rows": int(len(z["book"])), "executions": int((m[:, 1] == 4).sum()),
                            "hidden_executions": int((m[:, 1] == 5).sum()), "first_ns": int(m[0, 0]), "last_ns": int(m[-1, 0]),
                            "manifest_events": man.get(f"{s}_{day}", {}).get("events")})
            rows.append(row)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "real_files.csv", index=False)
    return out


def _day_job(args):
    return day_segment(*args)


def real(procs: int) -> None:
    from simrel.experiments.run_reference import pool_map

    segs = cached("real_segments", lambda: pool_map(_day_job, [(s, d) for s in SYMBOLS for d in DAYS], procs, "real days"))
    rng = np.random.default_rng(1)
    rows = []
    for sym in SYMBOLS:
        ss = sorted([s for s in segs if s["symbol"] == sym], key=lambda s: s["day"])
        qty = RV.parent_qty(ss)
        gaps = [RV.window_gaps(s, qty) for s in ss]
        row = {"side": "real", "family": sym, "unit": "8 days", "qty": qty, "grid_points_missing": sum(s["n_missing"] for s in ss)}
        row.update(describe(ss, gaps, None, rng))
        row.update(RV.context(ss))
        rows.append(row)
        for s, g in zip(ss, gaps):
            rows.append({"side": "real", "family": sym, "unit": s["day"], "qty": qty, "grid_points_missing": s["n_missing"], **stats_of([s], [g]),
                         "S4_share_p6_cheaper": float((g["s4"] < 0).mean()), "n_windows": len(g["s4"]), **RV.context([s])})
    pd.DataFrame(rows).to_csv(OUT / "real_table.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pd.set_option("display.width", 250)
    if cmd == "markets":
        write_markets()
    elif cmd == "sim":
        sim(procs)
    elif cmd == "files":
        print(files().to_string(index=False))
    elif cmd == "real":
        real(procs)
    else:
        for name in ("sim_table.csv", "real_table.csv"):
            if (OUT / name).exists():
                print(pd.read_csv(OUT / name).round(3).to_string(index=False))
