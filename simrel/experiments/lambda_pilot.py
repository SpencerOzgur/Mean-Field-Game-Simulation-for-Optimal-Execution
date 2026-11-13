"""
Pilot of the permanent-impact (lambda) agent-mix family (development only).

Markets XLP_pilot_* from the parameter stream default_rng(20261130) (simrel/reference_worlds/lambda_world.py). No
candidate simulator is fitted, no trust rule is applied and no held-out market is drawn here. The stopping rule is
simrel/reports/lambda_pilot_rule.md, committed before the main run.

    python simrel/experiments/lambda_pilot.py setup [procs]    viability of the draws at lambda = 0
    python simrel/experiments/lambda_pilot.py grid [procs]     first pass: volatility matching and persistence on a
                                                               grid of lambda values, first six draws
    python simrel/experiments/lambda_pilot.py final [procs]    final matching at the three levels, market selection,
                                                               arms, persistence of the final markets
    python simrel/experiments/lambda_pilot.py timing [procs]   timing check and projected runtime of the main run
    python simrel/experiments/lambda_pilot.py run [procs]      main run (E1 to E4), only after the rule is pushed
    python simrel/experiments/lambda_pilot.py report           CSVs, figures and the verdict

Seeds (episode seeds; the markets come from the pilot parameter stream):
    viability twins            SEED_HELDOUT + 0..39
    truth (evaluation)         0..99
    volatility calibration     SEED_CALIB + 0..47 (first pass), SEED_CALIB + 0..191 (final matching); passive episodes
    volatility check           SEED_VOLCHECK + 0..47               passive episodes, not used in the search
    persistence calibration    SEED_PERSIST + 0..N_PERSIST-1        twin and TWAP
    E2 samples                 SEED_E2A + k and SEED_E2B + k, k < N_E2
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
from simrel.core import Task  # noqa: E402
from simrel.eventsim import GRID_DT, run_twin_set  # noqa: E402
from simrel.lob import Book  # noqa: E402
from simrel.metrics.cost import episode_metrics  # noqa: E402
from simrel.metrics.realism import episode_stats  # noqa: E402
from simrel.policies.library import candidate_set  # noqa: E402
from simrel.reference_worlds import lambda_world as LW  # noqa: E402

OUT = ROOT / "results" / "simrel" / "goal2" / "lambda_pilot"
RAW = OUT / "raw"
N_AM, N_LATENT = 6, 2
N_TRUTH, N_VIAB, N_E2 = 100, 40, 20
N_VOL_CAL, N_VOL_CHECK, N_PERSIST = 48, 48, 300
VOL_EPISODE_S = 3600.0  # passive sample after a 600 s warm-up
SEED_VOLCHECK, SEED_PERSIST = 40_000, 50_000
SEED_E2A, SEED_E2B = 66_000, 67_000
GRID_BP = [0.25, 0.5, 0.75, 1.0, 1.5]  # lambda values (bp per 1% of reference volume) examined in calibration
# final levels (team decision after the calibration): dose (unjudged), band_low (120 s retention of 2/3 on the fitted
# calibration curve), max_feasible (the highest level at which every market can be volatility-matched)
LEVELS = {"dose": 0.25, "band_low": 0.6, "max_feasible": 1.0}
N_VOL_CAL_FINAL, VOL_TOL_FINAL = 192, 0.02  # final matching: SEED_CALIB + 0..191, 2% tolerance on the calibration seeds


def _task(q: int) -> Task:
    t = C.TASK
    return Task(t.side, q, t.horizon_s, t.n_slices, t.post_s)


def _order(frac: float, volume: float) -> int:
    return max(int(frac * volume) // 100 * 100, 100)


def load_json(name: str):
    return json.loads((OUT / name).read_text())


def markets() -> dict:
    """name -> (family, values) of the pilot markets (pilot_markets.json, written by write_markets)."""
    return {k: (v["family"], v["params"]) for k, v in load_json("pilot_markets.json")["markets"].items()}


def arms() -> dict:
    """arm -> {market -> world keyword arguments}; written by lambda_pilot_calib.finalise."""
    return load_json("arms.json")["arms"]


def factory(market: str, kw: dict, registry=None):
    fam, vals = ALL[market] if market in ALL else markets()[market]
    if fam == "am":
        return LW.make_lambda(vals, registry=registry, **kw)
    return LW.make_world(fam, vals)


ALL = LW.pilot_draws(n_extra=8)  # the block, then a13 to a20 (the search for matchable markets is capped at a20)
BLOCK = LW.pilot_draws()
N_EXTRA_TWINNED = 6  # further draws whose viability has been run (a13 to a18); the search ended at a15


# ---- jobs
def twin_job(args):
    """Unperturbed twin of the standard task: stylised facts for the viability rule."""
    market, arm, kw, seed = args
    warnings.filterwarnings("ignore")
    row = {"market": market, "arm": arm, "seed": seed, "failed": None}
    try:
        r = run_twin_set(factory(market, kw), market, seed, _task(0), {})["none"]
        row.update(episode_stats(r))
    except Exception as err:
        row["failed"] = repr(err)[:200]
    return row


def vol_job(args):
    """One passive episode: its mid on the 5 s grid (for the 600 s return variance) and descriptive statistics."""
    market, tag, kw, seed = args
    warnings.filterwarnings("ignore")
    r = LW.passive_mid(factory(market, kw), seed, VOL_EPISODE_S)
    mid, tape = r["mid"], r["tape"]
    d5 = np.diff(mid)
    book = r["book"]
    bb, ba = book.best_bid(), book.best_ask()

    def ac(x, lag=1):
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if len(x) > lag + 2 and np.std(x) > 0 else np.nan

    sign = np.sign([x[1] for x in tape])
    d60 = mid[12:] - mid[:-12]
    return {"market": market, "tag": tag, "seed": seed, "mid": mid.astype(np.float32),
            "volume_10min": float(sum(abs(x[1]) for x in tape)) / (VOL_EPISODE_S / 600.0), "n_trades": len(tape),
            "vol5_ticks": float(d5.std()), "ret5_ac1": ac(d5), "ret60_ac1": ac(d60[::12]), "abs5_ac1": ac(np.abs(d5)), "sign_ac1": ac(sign),
            "end_spread": float(ba - bb) if bb is not None and ba is not None else np.nan,
            "end_depth": 0.5 * (book.depth("bid", 10) + book.depth("ask", 10)),
            "F_minus_mid_end": float(r["model"].F - mid[-1]) if hasattr(r["model"], "F") else np.nan}


def persist_job(args):
    """Twin and TWAP only, at both sizes: displacement at the deadline and 120 s later (persistence calibration)."""
    market, tag, kw, volume, seed = args
    warnings.filterwarnings("ignore")
    rows = []
    for f in C.SIZES:
        q = _order(f, volume)
        try:
            recs = run_twin_set(factory(market, kw), market, seed, _task(q), {"P0_twap": candidate_set(f)["P0_twap"]})
        except Exception as err:
            rows.append({"market": market, "tag": tag, "seed": seed, "size_frac": f, "failed": repr(err)[:200]})
            continue
        row = episode_metrics(recs["P0_twap"], recs["none"])
        row.update({"market": market, "tag": tag, "size_frac": f, "qty": q})
        rows.append(row)
    return rows


def truth_job(args):
    """Twin and the 13 strategies at both sizes."""
    market, arm, kw, volume, seed = args
    warnings.filterwarnings("ignore")
    rows, t_start = [], time.time()
    for f in C.SIZES:
        q = _order(f, volume)
        pol = candidate_set(f)
        try:
            recs = run_twin_set(factory(market, kw), market, seed, _task(q), pol)
        except Exception as err:
            rows.append({"market": market, "arm": arm, "seed": seed, "size_frac": f, "failed": repr(err)[:200]})
            continue
        twin = recs["none"]
        for name, r in recs.items():
            if name == "none":
                continue
            row = episode_metrics(r, twin)
            row.update({"market": market, "arm": arm, "size_frac": f, "qty": q, "arrival_mid": twin.arrival_mid})
            rows.append(row)
    for r in rows:
        r["job_wall_s"] = time.time() - t_start
    return rows


def e2_job(args):
    """One observation sample (10 minutes warm-up, three hours) and its addendum 10 diagnostic features."""
    market, arm, kw, seed, tag = args
    warnings.filterwarnings("ignore")
    from simrel.data import lobster as L
    from simrel.experiments import goal2_reliability as GR

    row = {"market": market, "arm": arm, "seed": seed, "tag": tag}
    t_start = time.time()
    try:
        model = factory(market, kw)(seed)
        book, tape, t = Book(), [], 0.0
        book.tape = []
        model.init_book(book)
        end = GR.WARMUP_S + GR.SAMPLE_S
        while t < end - 1e-9:
            model.advance(book, t, t + GRID_DT, tape)
            t += GRID_DT
        w = L.tape_to_lobster(book.tape, GR.WARMUP_S, GR.SAMPLE_S, price_unit=100)
        row.update(GR.features(GR.G.R.two_sided(GR.G.R.S.visible_window(w))[0]))
        row["market_volume"] = float(sum(abs(x[1]) for x in tape))
        row["tape_hash"] = hash(tuple(tape))
    except Exception as err:
        row["failed"] = repr(err)[:300]
    row["wall_s"] = time.time() - t_start
    return row


# ---- stages
def cached(name: str, make):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    val = make()
    tmp = path.with_suffix(".tmp")
    pickle.dump(val, open(tmp, "wb"))
    tmp.replace(path)
    return val


def viability(rows: list) -> dict:
    out = {}
    for m in sorted({r["market"] for r in rows}):
        d = [r for r in rows if r["market"] == m]
        ok = [r for r in d if not r.get("failed")]
        fail = 1.0 - len(ok) / len(d)
        wild = float(np.mean([r["move_bps_abs"] > C.WILD_MOVE_BPS for r in ok])) if ok else 1.0
        out[m] = {"n": len(d), "fail_share": fail, "wild_share": wild, "viable": bool(fail <= C.MAX_FAIL_SHARE and wild <= C.MAX_WILD_SHARE),
                  "volume": float(np.mean([r["volume"] for r in ok])) if ok else float("nan"),
                  "spread_bps": float(np.mean([r["spread_bps"] for r in ok])) if ok else float("nan"),
                  "move_bps_abs": float(np.mean([r["move_bps_abs"] for r in ok])) if ok else float("nan")}
    return out


def draw_viability(procs: int) -> dict:
    """Viability rule at lambda = 0 (40 twins) for the block and for the further draws examined so far."""
    from simrel.experiments.run_reference import pool_map

    extra = [m for m in ALL if m not in BLOCK][:N_EXTRA_TWINNED]
    tw = cached("twins_draws", lambda: pool_map(twin_job, [(m, "lam0", {}, C.SEED_HELDOUT + s) for m in BLOCK for s in range(N_VIAB)], procs, "twins draws"))
    tw = tw + cached("twins_extra", lambda: pool_map(twin_job, [(m, "lam0", {}, C.SEED_HELDOUT + s) for m in extra for s in range(N_VIAB)], procs, "twins extra"))
    return viability(tw)


def write_markets(kept: list, procs: int = 6) -> None:
    """pilot_markets.json: viability of every examined draw, the agent-mix markets kept by the selection rule
    (lambda_pilot_calib.kept_markets; empty before the final matching) and the first two viable latent draws."""
    OUT.mkdir(parents=True, exist_ok=True)
    v = draw_viability(procs)
    la = [m for m in ALL if ALL[m][0] == "latent" and v[m]["viable"]][:N_LATENT]
    chosen = list(kept) + la
    (OUT / "pilot_markets.json").write_text(json.dumps({
        "parameter_stream": f"numpy default_rng({LW.PILOT_SEED}); a block of 12 agent-mix draws (sampled.AM_RANGES), 4 latent draws (latent.RANGES), "
                            "then further agent-mix draws (a13 onward, at most through a20)",
        "note": "pilot stream, development only; never reuse",
        "selection": "agent-mix: in draw order, the first six draws that pass the viability rule at lambda = 0 (40 twins) and are volatility-matched "
                     "at every level (calib_final.csv); latent: the first two viable draws. See simrel/reports/lambda_pilot_rule.md",
        "draw_block_viability": v,
        "markets": {k: {"family": ALL[k][0], "params": ALL[k][1]} for k in chosen},
        "reference_volume": {k: v[k]["volume"] for k in chosen}}, indent=2))
    for m in v:
        print(f"  {m}  viable={v[m]['viable']}  fail={v[m]['fail_share']:.2f} wild={v[m]['wild_share']:.2f} volume={v[m]['volume']:.0f} "
              f"{'PILOT' if m in chosen else ''}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if cmd == "setup":
        write_markets([], procs)
    elif cmd in ("grid", "final"):
        from simrel.experiments import lambda_pilot_calib

        lambda_pilot_calib.main(cmd, procs)
    elif cmd in ("timing", "run"):
        from simrel.experiments import lambda_pilot_run

        getattr(lambda_pilot_run, cmd)(procs)
    else:
        from simrel.experiments import lambda_pilot_report

        lambda_pilot_report.main()
