"""
Design addendum 11, development stage: pilot of the reactive (predator) reference family.

Pilot markets only (XRP_pilot_*, parameter stream default_rng(20261120), simrel/reference_worlds/reactive.py). No
candidate simulator is fitted, no trust rule is applied and no held-out market is drawn here.

    E1  detection: trigger rate and time to first trigger for each of the 13 strategies, and the false-trigger rate
        on passive (no metaorder) episodes
    E2  true costs of the 13 strategies for each market, strength and size (100 seeds)
    E3  observability: the mechanism-diagnostic features of goal2_reliability.py on passive samples with the predator
        on and off; cross-validated AUC of a logistic classifier (the method of addendum 10, section 5)
    E4  the viability rule of run_reference.py at each strength

    python simrel/experiments/reactive_pilot.py timing [procs]     tiny timing run and projected runtime
    python simrel/experiments/reactive_pilot.py run [procs]        all stages (resumable)
    python simrel/experiments/reactive_pilot.py report             CSVs and figures from the stored stages

Seeds (episode seeds; the markets themselves come from the pilot parameter stream):
    viability twins   SEED_HELDOUT + 0..39     (configs/reference.py)
    truth             0..99
    passive episodes  SEED_FIT + 0..5 (the six "fitting-like" episodes) and SEED_FIT + 100..139 (for a tighter rate)
    E3 samples        SEED_E3A + k and SEED_E3B + k, k < N_E3
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.configs import reference as C  # noqa: E402
from simrel.core import Task  # noqa: E402
from simrel.eventsim import GRID_DT, run_twin_set  # noqa: E402
from simrel.lob import Book  # noqa: E402
from simrel.metrics.cost import episode_metrics  # noqa: E402
from simrel.metrics.realism import episode_stats  # noqa: E402
from simrel.policies.reactive import reactive_candidate_set  # noqa: E402
from simrel.reference_worlds import reactive as RX  # noqa: E402

OUT = ROOT / "results" / "simrel" / "goal2" / "addendum11_pilot"
RAW = OUT / "raw"
MARKETS = RX.pilot_markets()
STRENGTHS = ["off", "weak", "medium", "strong"]  # the family's detector (noisy time stamps, size-led)
TIGHT = ["weak_tight", "medium_tight", "strong_tight"]  # comparison arms: the tight-timing detector on exact time stamps
LOOSE = "strong_loose"  # negative control: the strong predator with a loose detector on exact time stamps
TRUTH_ARMS = STRENGTHS + TIGHT
ARMS = TRUTH_ARMS + [LOOSE]
N_TRUTH, N_VIAB, N_PASSIVE_EXTRA, N_E3, N_SWEEP = 100, 40, 40, 20, 30
SEED_E3A, SEED_E3B = 66_000, 67_000
PASSIVE_S = 1800.0
WARMUP_S = 1200.0  # run_twin_set's warm-up: the task starts here
# detector settings replayed in shadow on the strength-off truth runs (E1): the default and one-at-a-time changes
SWEEP = {"default": {}, "noise_0": {"ts_noise_s": 0.0}, "noise_0.3": {"ts_noise_s": 0.3}, "gap_0.02": {"gap_cv_tol": 0.02},
         "size_0.1": {"size_cv_tol": 0.1}, "size_0.25": {"size_cv_tol": 0.25}, "size_0.5": {"size_cv_tol": 0.5},
         "m_6": {"min_matches": 6}, "m_7": {"min_matches": 7}, "tight": RX.DETECTORS["tight"], "loose": RX.DETECTORS["loose"]}


def arm_params(arm: str, volume: float) -> RX.PredatorParams:
    strength, _, detector = arm.partition("_")
    return RX.predator_params(strength, volume, detector or "noisy")


def _task(q: int) -> Task:
    t = C.TASK
    return Task(t.side, q, t.horizon_s, t.n_slices, t.post_s)


def _mark(rec) -> float:
    return float(rec.grid_mid[-1])


# ---- jobs
def twin_job(args):
    """Unperturbed twin of the standard task: stylised facts for the viability rule, and what the predator did."""
    market, arm, volume, seed = args
    warnings.filterwarnings("ignore")
    fam, vals = MARKETS[market]
    row = {"market": market, "arm": arm, "seed": seed, "failed": None}
    try:
        rec = RX.Recorder()
        r = run_twin_set(RX.make_reactive(fam, vals, arm_params(arm, volume), rec), market, seed, _task(0), {})["none"]
        row.update(episode_stats(r))
        row.update(rec.by_run({})["none"].summary(WARMUP_S, _mark(r)))
    except Exception as err:
        row["failed"] = repr(err)[:200]
    return row


def shadow_replay(rows: list, pp: RX.PredatorParams, seed: int = 0) -> list:
    p = RX.Predator(replace(pp, aggression=0.0), seed)
    for t, side, size in rows:
        p.observe(t, side, size)
    return p.triggers


def truth_job(args):
    """Twin and the 13 strategies at both sizes; costs, the predator's account, and (strength off) detector replays."""
    market, arm, volume, seed = args
    warnings.filterwarnings("ignore")
    fam, vals = MARKETS[market]
    pp = arm_params(arm, volume)
    rows, t_start = [], time.time()
    for f in C.SIZES:
        q = max(int(f * volume) // 100 * 100, 100)
        pol = reactive_candidate_set(f, market, seed)
        try:
            rec = RX.Recorder()
            factory = RX.make_reactive(fam, vals, pp, rec)
            sweep = arm == "off" and seed < N_SWEEP

            def make(s, factory=factory, sweep=sweep):
                m = factory(s)
                if sweep:
                    m.pred.history = []
                return m

            recs = run_twin_set(make, market, seed, _task(q), pol)
        except Exception as err:
            rows.append({"market": market, "arm": arm, "seed": seed, "size_frac": f, "failed": repr(err)[:200]})
            continue
        preds = rec.by_run(pol)
        twin = recs["none"]
        for name, r in recs.items():
            row = episode_metrics(r, twin) if name != "none" else {"policy": "none", "seed": seed, "failed": twin.failed}
            row.update({"market": market, "arm": arm, "size_frac": f, "qty": q, "arrival_mid": twin.arrival_mid})
            pr = preds[name]
            row.update(pr.summary(WARMUP_S, _mark(r)))
            in_task = [x for x in pr.triggers if WARMUP_S <= x[0] <= WARMUP_S + C.TASK.horizon_s and x[1] == -1]
            row["trig_task"] = bool(in_task)
            row["trig_task_first_s"] = in_task[0][0] - WARMUP_S if in_task else np.nan
            row["trig_task_period_s"] = in_task[0][2] if in_task else np.nan
            if name != "none":
                row["n_children"] = int((r.fill_qty > 0).sum())
                fq = r.fill_qty[r.fill_qty > 0]
                row["child_size_cv"] = float(fq.std() / fq.mean()) if len(fq) else np.nan
                row["first_slices"] = ",".join(str(int(x)) for x in r.fill_qty)
            if sweep:
                hist = [x for x in pr.history if x[0] >= WARMUP_S - pp.window_s]
                for tag, ch in SWEEP.items():
                    tr = [x for x in shadow_replay(hist, replace(pp, **ch), seed) if WARMUP_S <= x[0] <= WARMUP_S + C.TASK.horizon_s and x[1] == -1]
                    row[f"sw_{tag}"] = bool(tr)
                    row[f"sw_{tag}_first_s"] = tr[0][0] - WARMUP_S if tr else np.nan
            rows.append(row)
    for r in rows:
        r["job_wall_s"] = time.time() - t_start
    return rows


def passive_job(args):
    """One 30-minute passive episode (no metaorder), as observe() runs it: what the predator did."""
    market, arm, volume, seed = args
    warnings.filterwarnings("ignore")
    fam, vals = MARKETS[market]
    pp = arm_params(arm, volume)
    model = RX.make_reactive(fam, vals, pp)(seed)
    if arm == "off":
        model.pred.history = []
    book, tape, t = Book(), [], 0.0
    model.init_book(book)
    while t < PASSIVE_S - 1e-9:
        model.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    row = {"market": market, "arm": arm, "seed": seed, "market_volume": float(sum(abs(x[1]) for x in tape)), "n_trades": len(tape)}
    row.update(model.pred.summary(0.0, book.last_price or np.nan))
    if arm == "off":  # shadow replays of other detector settings on the same tape
        for tag, ch in SWEEP.items():
            row[f"sw_{tag}"] = len(shadow_replay(model.pred.history, replace(pp, **ch), seed))
    return row


def e3_job(args):
    """One observation sample (10 minutes warm-up, three hours), its diagnostic features, and the predator's footprint."""
    market, arm, volume, seed, tag = args
    warnings.filterwarnings("ignore")
    from simrel.data import lobster as L
    from simrel.experiments import goal2_reliability as GR

    fam, vals = MARKETS[market]
    row = {"market": market, "arm": arm, "seed": seed, "tag": tag}
    t_start = time.time()
    try:
        model = RX.make_reactive(fam, vals, arm_params(arm, volume))(seed)
        book, tape, t = Book(), [], 0.0
        book.tape = []
        model.init_book(book)
        end = GR.WARMUP_S + GR.SAMPLE_S
        while t < end - 1e-9:  # the 5 s grid for every arm, so paired on / off runs differ only by the predator's orders
            model.advance(book, t, t + GRID_DT, tape)
            t += GRID_DT
        w = L.tape_to_lobster(book.tape, GR.WARMUP_S, GR.SAMPLE_S, price_unit=100)
        row.update(GR.features(GR.G.R.two_sided(GR.G.R.S.visible_window(w))[0]))
        row.update(model.pred.summary(0.0, book.last_price or np.nan))
        row["market_volume"] = float(sum(abs(x[1]) for x in tape))
    except Exception as err:
        row["failed"] = repr(err)[:300]
    row["wall_s"] = time.time() - t_start
    return row


# ---- stages
def cached(name: str, make):
    path = RAW / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    val = make()
    tmp = path.with_suffix(".tmp")
    pickle.dump(val, open(tmp, "wb"))
    tmp.replace(path)
    return val


def volumes(twins_off: list) -> dict:
    out = {}
    for m in MARKETS:
        v = [r["volume"] for r in twins_off if r["market"] == m and not r.get("failed")]
        out[m] = float(np.mean(v)) if v else float("nan")
    return out


def run(procs: int) -> None:
    from simrel.experiments.run_reference import pool_map

    RAW.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_markets.json").write_text(json.dumps({
        "parameter_stream": f"numpy default_rng({RX.PILOT_SEED}); agent-mix draws first (sampled.AM_RANGES), then latent (latent.RANGES)",
        "note": "pilot stream, development only; never reuse for held-out markets",
        "predator_stream": f"default_rng([seed, {RX.PREDATOR_STREAM}, 0])",
        "markets": {k: {"family": f, "params": v} for k, (f, v) in MARKETS.items()},
        "strengths": {k: {"aggression": a, "inventory_cap_share_of_volume": c} for k, (a, c) in RX.STRENGTHS.items()},
        "detector_defaults": asdict(RX.PredatorParams()), "detector_variants": RX.DETECTORS,
        "seeds": {"viability": [C.SEED_HELDOUT, C.SEED_HELDOUT + N_VIAB - 1], "truth": [0, N_TRUTH - 1],
                  "passive": [C.SEED_FIT, C.SEED_FIT + 5], "passive_extra": [C.SEED_FIT + 100, C.SEED_FIT + 100 + N_PASSIVE_EXTRA - 1],
                  "e3_a": [SEED_E3A, SEED_E3A + N_E3 - 1], "e3_b": [SEED_E3B, SEED_E3B + N_E3 - 1]}}, indent=2))
    twins_off = cached("twins_off", lambda: pool_map(twin_job, [(m, "off", 0.0, C.SEED_HELDOUT + s) for m in MARKETS for s in range(N_VIAB)], procs, "twins off"))
    V = volumes(twins_off)
    (OUT / "reference_volume.json").write_text(json.dumps(V, indent=2))
    print("reference volume (strength off):", {k: round(v) for k, v in V.items()}, flush=True)
    cached("twins_on", lambda: pool_map(twin_job, [(m, a, V[m], C.SEED_HELDOUT + s) for m in MARKETS for a in ARMS[1:] for s in range(N_VIAB)], procs, "twins on"))
    seeds = [C.SEED_FIT + s for s in range(6)] + [C.SEED_FIT + 100 + s for s in range(N_PASSIVE_EXTRA)]
    cached("passive", lambda: pool_map(passive_job, [(m, a, V[m], s) for m in MARKETS for a in ARMS for s in seeds], procs, "passive"))
    cached("truth", lambda: pool_map(truth_job, [(m, a, V[m], s) for m in MARKETS for a in TRUTH_ARMS for s in range(N_TRUTH)], procs, "truth"))
    jobs = [(m, "off", V[m], SEED_E3B + k, "B") for m in MARKETS for k in range(N_E3)]
    jobs += [(m, a, V[m], SEED_E3A + k, "A") for m in MARKETS for a in ARMS for k in range(N_E3)]
    cached("e3", lambda: pool_map(e3_job, jobs, procs, "e3"))


def timing(procs: int) -> None:
    ms = ["XRP_pilot_a01", "XRP_pilot_l01"]
    t0 = time.time()
    tw = [twin_job((m, "off", 0.0, C.SEED_HELDOUT)) for m in ms]
    t_twin = (time.time() - t0) / len(ms)
    V = {m: r["volume"] for m, r in zip(ms, tw)}
    t0 = time.time()
    for m in ms:
        passive_job((m, "strong", V[m], C.SEED_FIT))
    t_pass = (time.time() - t0) / len(ms)
    t0 = time.time()
    for m in ms:
        truth_job((m, "strong", V[m], 0))
    t_truth = (time.time() - t0) / len(ms)
    t0 = time.time()
    for m in ms:
        truth_job((m, "off", V[m], 0))
    t_truth_sw = (time.time() - t0) / len(ms)
    t0 = time.time()
    e3 = [e3_job((m, "strong", V[m], SEED_E3A, "A")) for m in ms]
    t_e3 = (time.time() - t0) / len(ms)
    nm = len(MARKETS)
    n = {"twins": nm * len(ARMS) * N_VIAB, "passive": nm * len(ARMS) * (6 + N_PASSIVE_EXTRA),
         "truth": nm * (len(TRUTH_ARMS) * N_TRUTH - N_SWEEP), "truth_sweep": nm * N_SWEEP, "e3": nm * (len(ARMS) + 1) * N_E3}
    per = {"twins": t_twin, "passive": t_pass, "truth": t_truth, "truth_sweep": t_truth_sw, "e3": t_e3}
    total = sum(n[k] * per[k] for k in n)
    for k in n:
        print(f"  {k:12s} {n[k]:5d} jobs x {per[k]:6.1f} s = {n[k] * per[k] / 3600:5.2f} core-hours")
    print(f"  E3 timing samples failed: {[r.get('failed') for r in e3]}")
    print(f"projected: {total / 3600:.2f} core-hours, about {total / 3600 / procs * 1.25:.2f} hours on {procs} processes (25% overhead allowed)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if cmd == "timing":
        timing(procs)
    elif cmd == "run":
        run(procs)
    else:
        from simrel.experiments import reactive_pilot_report

        reactive_pilot_report.main()
