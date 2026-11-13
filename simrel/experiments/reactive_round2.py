"""
Design addendum 11, round 2 (development only): the rational predator on the six pilot markets.

The stopping rule is simrel/reports/addendum11_round2_rule.md, committed and pushed before this file existed. Arms
(simrel/reference_worlds/reactive.py, ROUND2_ARMS): off; aggressive_medium and aggressive_strong (the pilot's medium
and strong, rerun so that the P&L components exist; the report checks they reproduce the pilot's costs exactly);
rational_medium and rational_strong (passive exit and profit gate); passive_ungated_strong (passive exit, no gate).

Same markets, reference volumes, seeds and methods as reactive_pilot.py. No candidate simulator is fitted, no trust
rule is applied, no market is drawn.

    python simrel/experiments/reactive_round2.py timing [procs]
    python simrel/experiments/reactive_round2.py run [procs]
    python simrel/experiments/reactive_round2.py report
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import reference as C  # noqa: E402
from simrel.eventsim import GRID_DT, run_twin_set  # noqa: E402
from simrel.experiments import reactive_pilot as P  # noqa: E402
from simrel.lob import Book  # noqa: E402
from simrel.metrics.cost import episode_metrics  # noqa: E402
from simrel.metrics.realism import episode_stats  # noqa: E402
from simrel.policies.reactive import reactive_candidate_set  # noqa: E402
from simrel.reference_worlds import reactive as RX  # noqa: E402

OUT = ROOT / "results" / "simrel" / "goal2" / "addendum11_round2"
RAW = OUT / "raw"
MARKETS = P.MARKETS
ARMS = list(RX.ROUND2_ARMS)
RATIONAL_ARMS = ["rational_medium", "rational_strong"]  # the arms the rule judges
NEW_ARMS = RATIONAL_ARMS + ["passive_ungated_strong"]
N_TRUTH, N_VIAB, N_PASSIVE_EXTRA, N_E3 = P.N_TRUTH, P.N_VIAB, P.N_PASSIVE_EXTRA, P.N_E3
WARMUP_S = P.WARMUP_S
LEDGER = ["trader", "background", "bg_cancel", "own_cancel"]


def reference_volume() -> dict:
    """The pilot's reference volumes (strength off), so that orders and caps are the same shares as in the pilot."""
    return json.loads((P.OUT / "reference_volume.json").read_text())


def _ledger(led: dict) -> dict:
    return {f"fill_by_{k}" if k in ("trader", "background") else k: led[k] for k in LEDGER}


# ---- jobs
def twin_job(args):
    market, arm, volume, seed = args
    warnings.filterwarnings("ignore")
    fam, vals = MARKETS[market]
    row = {"market": market, "arm": arm, "seed": seed, "failed": None}
    try:
        rec = RX.Recorder()
        r = run_twin_set(RX.make_reactive(fam, vals, RX.round2_params(arm, volume), rec), market, seed, P._task(0), {})["none"]
        row.update(episode_stats(r))
        row.update(rec.by_run({})["none"].summary(WARMUP_S, P._mark(r)))
    except Exception as err:
        row["failed"] = repr(err)[:200]
    return row


def truth_job(args):
    market, arm, volume, seed = args
    warnings.filterwarnings("ignore")
    fam, vals = MARKETS[market]
    pp = RX.round2_params(arm, volume)
    rows, t_start = [], time.time()
    for f in C.SIZES:
        q = max(int(f * volume) // 100 * 100, 100)
        pol = reactive_candidate_set(f, market, seed)
        try:
            rec, led = RX.Recorder(), RX.Recorder()
            recs = run_twin_set(RX.make_reactive(fam, vals, pp, rec, led), market, seed, P._task(q), pol)
        except Exception as err:
            rows.append({"market": market, "arm": arm, "seed": seed, "size_frac": f, "failed": repr(err)[:200]})
            continue
        preds, ledgers = rec.by_run(pol), led.by_run(pol)
        twin = recs["none"]
        for name, r in recs.items():
            row = episode_metrics(r, twin) if name != "none" else {"policy": "none", "seed": seed, "failed": twin.failed}
            row.update({"market": market, "arm": arm, "size_frac": f, "qty": q, "arrival_mid": twin.arrival_mid})
            pr = preds[name]
            row.update(pr.summary(WARMUP_S, P._mark(r)))
            row.update(_ledger(ledgers[name]))
            in_task = [x for x in pr.triggers if WARMUP_S <= x[0] <= WARMUP_S + C.TASK.horizon_s and x[1] == -1]
            row["trig_task"] = bool(in_task)
            row["trig_task_first_s"] = in_task[0][0] - WARMUP_S if in_task else np.nan
            rows.append(row)
    for r in rows:
        r["job_wall_s"] = time.time() - t_start
    return rows


def passive_job(args):
    market, arm, volume, seed = args
    warnings.filterwarnings("ignore")
    fam, vals = MARKETS[market]
    model = RX.make_reactive(fam, vals, RX.round2_params(arm, volume))(seed)
    book, tape, t = Book(), [], 0.0
    model.init_book(book)
    while t < P.PASSIVE_S - 1e-9:
        model.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    row = {"market": market, "arm": arm, "seed": seed, "market_volume": float(sum(abs(x[1]) for x in tape)), "n_trades": len(tape)}
    row.update(model.pred.summary(0.0, book.last_price or np.nan))
    row.update(_ledger(model.ledger))
    return row


def e3_job(args):
    market, arm, volume, seed, tag = args
    warnings.filterwarnings("ignore")
    from simrel.data import lobster as L
    from simrel.experiments import goal2_reliability as GR

    fam, vals = MARKETS[market]
    row = {"market": market, "arm": arm, "seed": seed, "tag": tag}
    t_start = time.time()
    try:
        model = RX.make_reactive(fam, vals, RX.round2_params(arm, volume))(seed)
        book, tape, t = Book(), [], 0.0
        book.tape = []
        model.init_book(book)
        end = GR.WARMUP_S + GR.SAMPLE_S
        while t < end - 1e-9:
            model.advance(book, t, t + GRID_DT, tape)
            t += GRID_DT
        w = L.tape_to_lobster(book.tape, GR.WARMUP_S, GR.SAMPLE_S, price_unit=100)
        row.update(GR.features(GR.G.R.two_sided(GR.G.R.S.visible_window(w))[0]))
        row.update(model.pred.summary(0.0, book.last_price or np.nan))
        row.update(_ledger(model.ledger))
        row["market_volume"] = float(sum(abs(x[1]) for x in tape))
    except Exception as err:
        row["failed"] = repr(err)[:300]
    row["wall_s"] = time.time() - t_start
    return row


def cached(name: str, make):
    path = RAW / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    val = make()
    tmp = path.with_suffix(".tmp")
    pickle.dump(val, open(tmp, "wb"))
    tmp.replace(path)
    return val


def run(procs: int) -> None:
    from simrel.experiments.run_reference import pool_map

    RAW.mkdir(parents=True, exist_ok=True)
    V = reference_volume()
    (OUT / "round2_arms.json").write_text(json.dumps({
        "rule": "simrel/reports/addendum11_round2_rule.md", "markets": list(MARKETS), "reference_volume": V,
        "arms": {a: {"strength": s, "changes": ch} for a, (s, ch) in RX.ROUND2_ARMS.items()},
        "params": {a: RX.round2_params(a, V["XRP_pilot_a01"]).__dict__ for a in ARMS},
        "seeds": "as in reactive_pilot.py"}, indent=2))
    cached("twins_off", lambda: pool_map(twin_job, [(m, "off", V[m], C.SEED_HELDOUT + s) for m in MARKETS for s in range(N_VIAB)], procs, "twins off"))
    cached("twins_on", lambda: pool_map(twin_job, [(m, a, V[m], C.SEED_HELDOUT + s) for m in MARKETS for a in ARMS[1:] for s in range(N_VIAB)], procs, "twins on"))
    seeds = [C.SEED_FIT + s for s in range(6)] + [C.SEED_FIT + 100 + s for s in range(N_PASSIVE_EXTRA)]
    cached("passive", lambda: pool_map(passive_job, [(m, a, V[m], s) for m in MARKETS for a in ARMS for s in seeds], procs, "passive"))
    cached("truth", lambda: pool_map(truth_job, [(m, a, V[m], s) for m in MARKETS for a in ARMS for s in range(N_TRUTH)], procs, "truth"))
    jobs = [(m, "off", V[m], P.SEED_E3B + k, "B") for m in MARKETS for k in range(N_E3)]
    jobs += [(m, a, V[m], P.SEED_E3A + k, "A") for m in MARKETS for a in ["off"] + NEW_ARMS for k in range(N_E3)]
    cached("e3", lambda: pool_map(e3_job, jobs, procs, "e3"))


def timing(procs: int) -> None:
    V = reference_volume()
    ms = ["XRP_pilot_a01", "XRP_pilot_l01"]
    per = {}
    for name, job, arm, extra in [("twins", twin_job, "rational_strong", (C.SEED_HELDOUT,)), ("passive", passive_job, "passive_ungated_strong", (C.SEED_FIT,)),
                                  ("truth", truth_job, "passive_ungated_strong", (0,)), ("e3", e3_job, "passive_ungated_strong", (P.SEED_E3A, "A"))]:
        t0 = time.time()
        out = [job((m, arm, V[m]) + extra) for m in ms]
        per[name] = (time.time() - t0) / len(ms)
        if name == "e3":
            print("  E3 timing samples failed:", [r.get("failed") for r in out])
    nm = len(MARKETS)
    n = {"twins": nm * len(ARMS) * N_VIAB, "passive": nm * len(ARMS) * (6 + N_PASSIVE_EXTRA), "truth": nm * len(ARMS) * N_TRUTH,
         "e3": nm * (len(NEW_ARMS) + 2) * N_E3}
    total = sum(n[k] * per[k] for k in n)
    for k in n:
        print(f"  {k:8s} {n[k]:5d} jobs x {per[k]:6.1f} s = {n[k] * per[k] / 3600:5.2f} core-hours")
    print(f"projected: {total / 3600:.2f} core-hours, about {total / 3600 / procs * 1.25:.2f} hours on {procs} processes (25% overhead allowed)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if cmd == "timing":
        timing(procs)
    elif cmd == "run":
        run(procs)
    else:
        from simrel.experiments import reactive_round2_report

        reactive_round2_report.main()
