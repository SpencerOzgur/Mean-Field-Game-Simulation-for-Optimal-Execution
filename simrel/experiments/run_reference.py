"""
Phase 4: controlled decision validity against reference worlds.

For each reference world (a market whose rules we wrote):

  1. record unperturbed data from it (event logs and quotes);
  2. fit each candidate simulator to that data only:
       hawkes  maximum-likelihood fit of the marked Hawkes simulator
       abides  simulated-moments search over an ABIDES configuration grid
       mars    the pretrained MarS model started from replayed reference order flow
               (it cannot be refitted: no training code is released)
  3. let each fitted simulator choose its best policy from the candidate set;
  4. evaluate every candidate policy in the reference world itself, on many seeds.

Stage outputs are written under results/simrel/ and the run is resumable.

    python simrel/experiments/run_reference.py <MarS checkout> <MarS data dir> [procs] [worlds,comma,separated] [sims]
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simrel.configs import reference as C
from simrel.core import Task
from simrel.metrics.cost import episode_metrics
from simrel.metrics.realism import episode_stats, realism_error
from simrel.policies.library import candidate_set

OUT = REPO_ROOT / "results" / "simrel"
RAW = OUT / "raw"
CTX: dict = {}


def _task(q: int) -> Task:
    return Task(C.TASK.side, q, C.TASK.horizon_s, C.TASK.n_slices, C.TASK.post_s)


def _run(sim: str, world: str, seed: int, jobs: dict[float, tuple[int, dict]], spec):
    """Twin plus policies in one simulator. jobs: size -> (qty, policies). spec: fitted parameters for that simulator."""
    if sim == "reference" or sim.startswith("hawkes"):
        from simrel.eventsim import run_twin_set

        if sim == "reference":
            from simrel.reference_worlds.data import make

            factory = make(world)
        else:
            from simrel.adapters.hawkes import HawkesLOB

            factory = lambda s: HawkesLOB(spec, s)
        return {f: run_twin_set(factory, f"{world}:{sim}", seed, _task(q), pol) for f, (q, pol) in jobs.items()}
    if sim == "abides":
        from simrel.adapters.abides import run_twin_set
        from simrel.adapters.abides_fit import register

        name = register(spec)
        out = {f: run_twin_set(name, seed, _task(q), pol) for f, (q, pol) in jobs.items()}
        for recs in out.values():
            for r in recs.values():
                r.world = f"{world}:{sim}"
        return out
    from simrel.adapters.mars import run_twin_set
    from simrel.adapters.mars_local import setup_mars
    from simrel.reference_worlds.data import observe

    setup_mars(CTX["mars_root"], CTX["mars_data"])
    policies, sizes = {}, {}
    for f, (q, pol) in jobs.items():
        for name, p in pol.items():
            policies[f"{f}|{name}"], sizes[f"{f}|{name}"] = p, q
    recs = run_twin_set(seed, C.TASK, policies, world=f"{world}:{sim}", sizes=sizes, burn_in_s=300,
                        init_observed=observe(world, C.SEED_MARS_INIT + seed))
    out = {f: {"none": recs["none"]} for f in jobs}
    for k, r in recs.items():
        if k != "none":
            f, name = k.split("|", 1)
            r.policy = name
            out[float(f)][name] = r
    return out


def twin_job(args):
    """Unperturbed twin only: returns stylised facts and volume."""
    sim, world, seed, spec, ctx = args
    warnings.filterwarnings("ignore")
    CTX.update(ctx)
    try:
        r = _run(sim, world, seed, {0.0: (0, {})}, spec)[0.0]["none"]
        if r.failed:
            return {"sim": sim, "world": world, "seed": seed, "failed": r.failed}
        return {"sim": sim, "world": world, "seed": seed, "failed": None, "tag": getattr(spec, "name", None), **episode_stats(r)}
    except Exception as err:
        return {"sim": sim, "world": world, "seed": seed, "failed": repr(err)[:200]}


def policy_job(args):
    sim, world, seed, volume, spec, ctx = args
    warnings.filterwarnings("ignore")
    CTX.update(ctx)
    lot = 1 if sim == "abides" else 100
    jobs = {f: (max(int(f * volume) // lot * lot, lot), candidate_set(f)) for f in C.SIZES}
    t0 = time.time()
    try:
        sets = _run(sim, world, seed, jobs, spec)
    except Exception as err:
        return [{"sim": sim, "world": world, "seed": seed, "failed": repr(err)[:200]}]
    rows = []
    for f, recs in sets.items():
        twin = recs["none"]
        for name, r in recs.items():
            if name == "none":
                continue
            row = episode_metrics(r, twin)
            row.update({"sim": sim, "world": world, "size_frac": f, "job_wall_s": time.time() - t0})
            rows.append(row)
    return rows


def pool_map(fn, jobs, procs, label):
    out, t0 = [], time.time()
    if not jobs:
        return out
    with Pool(procs) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
            out.extend(res if isinstance(res, list) else [res])
            if i % 50 == 0 or i == len(jobs):
                print(f"  [{label}] {i}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    return out


def cached(path: Path, make):
    if path.exists():
        return pickle.load(open(path, "rb"))
    val = make()
    tmp = path.with_suffix(".tmp")
    pickle.dump(val, open(tmp, "wb"))
    tmp.replace(path)
    return val


def mean_volume(rows):
    v = [r["volume"] for r in rows if not r.get("failed") and np.isfinite(r.get("volume", np.nan))]
    return float(np.mean(v)) if v else float("nan")


if __name__ == "__main__":
    ctx = {"mars_root": sys.argv[1], "mars_data": sys.argv[2]}
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    from simrel.reference_worlds.data import observe, registry

    worlds = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "all" else list(registry())
    sims = sys.argv[5].split(",") if len(sys.argv) > 5 else ["hawkes", "abides", "mars"]
    RAW.mkdir(parents=True, exist_ok=True)

    for W in worlds:
        print(f"=== {W}", flush=True)
        D = RAW / f"ref_{W}"
        D.mkdir(exist_ok=True)

        # 1. observable reference data: fitting episodes and held-out twins
        fit_data = cached(D / "fit_data.pkl", lambda: [observe(W, C.SEED_FIT + s) for s in range(C.N_FIT_EPISODES)])
        heldout = cached(D / "heldout.pkl", lambda: pool_map(twin_job, [("reference", W, C.SEED_HELDOUT + s, None, ctx) for s in range(C.N_HELDOUT)], procs, "reference held-out"))
        ref_volume = mean_volume(heldout)
        from simrel.reference_worlds import ablation

        if W in ablation.markets():  # ablation markets keep the base market's order size in shares (addendum 6)
            ref_volume = ablation.BASE_VOLUME[ablation.markets()[W][0]]
        mean_price = float(np.mean([0.5 * (o.bid + o.ask).mean() for o in fit_data]))
        info = {"reference_volume": ref_volume, "mean_price_ticks": mean_price}
        # viability rule (addendum 3): the stability gate applied to every simulator, applied to the market itself
        ok = [r for r in heldout if not r.get("failed")]
        fail_share = 1.0 - len(ok) / max(len(heldout), 1)
        wild_share = float(np.mean([r["move_bps_abs"] > C.WILD_MOVE_BPS for r in ok])) if ok else 1.0
        info.update(viable=bool(fail_share <= C.MAX_FAIL_SHARE and wild_share <= C.MAX_WILD_SHARE), heldout_fail_share=fail_share, heldout_wild_share=wild_share)
        if not info["viable"]:
            ipath = D / "info.json"
            ipath.write_text(json.dumps({**(json.loads(ipath.read_text()) if ipath.exists() else {}), **info}, indent=2, default=str))
            print(f"  {W}: not viable ({fail_share:.0%} of held-out twins failed, {wild_share:.0%} moved more than {C.WILD_MOVE_BPS:.0f} bps); excluded", flush=True)
            continue

        # 2. the reference truth: every candidate policy on many seeds
        n_ref = int(os.environ.get("SIMREL_N_REFERENCE", C.N_REFERENCE))
        cached(D / "rows_reference.pkl", lambda: pool_map(policy_job, [("reference", W, s, ref_volume, None, ctx) for s in range(n_ref)], procs, "reference policies"))

        # 3. fit each candidate simulator on the observable data only
        specs = {}
        for hv in [x for x in sims if x.startswith("hawkes")]:
            def fit_hawkes(hv=hv):
                from simrel.adapters.hawkes_fit import fit

                return fit(fit_data, **C.HAWKES_VARIANTS[hv])

            specs[hv], info[f"{hv}_fit"] = cached(D / f"fit_{hv}.pkl", fit_hawkes)
        if "abides" in sims:
            def fit_abides():
                from simrel.adapters.abides_fit import grid_specs, spec_to_dict

                grid = grid_specs(int(round(mean_price)), W)
                rows = pool_map(twin_job, [("abides", W, C.SEED_CALIB + s, g, ctx) for g in grid for s in range(C.N_CALIB_SEEDS)], procs, "abides calibration")
                scores = {}
                for g in grid:
                    mine = [r for r in rows if r.get("tag") == g.name and not r.get("failed")]
                    scores[g.name] = realism_error(mine, heldout_fit)[0] if len(mine) >= 3 else float("inf")
                best = min(grid, key=lambda g: scores[g.name])
                return best, {"chosen": spec_to_dict(best), "grid_scores": scores}

            # calibration compares against the fitting episodes' statistics, not the held-out ones
            heldout_fit = cached(D / "fit_twins.pkl", lambda: pool_map(twin_job, [("reference", W, C.SEED_FIT + 100 + s, None, ctx) for s in range(20)], procs, "reference fit twins"))
            specs["abides"], info["abides_fit"] = cached(D / "fit_abides.pkl", fit_abides)
        if "mars" in sims:
            specs["mars"] = None

        # 4. each simulator: own volume, realism against held-out reference data, and all candidate policies
        # (skipped for the Goal 2 markets added later: Goal 2 runs its own candidates; addendum 3)
        for sim, spec in ({} if os.environ.get("SIMREL_REFERENCE_ONLY") else specs).items():
            vol_rows = cached(D / f"twins_{sim}.pkl", lambda: pool_map(twin_job, [(sim, W, C.SEED_VOLUME + s, spec, ctx) for s in range(C.N_VOLUME[C.family(sim)])], procs, f"{sim} twins"))
            ok = [r for r in vol_rows if not r.get("failed")]
            info[f"{sim}_twin_failures"] = len(vol_rows) - len(ok)
            # Gate D inside the experiment: a simulator whose unperturbed runs mostly fail, or whose mid moves by more
            # than 3% in ten minutes in over a quarter of them, is not asked for a recommendation.
            fail_share = 1.0 - len(ok) / max(len(vol_rows), 1)
            wild_share = float(np.mean([r["move_bps_abs"] > C.WILD_MOVE_BPS for r in ok])) if ok else 1.0
            info[f"{sim}_twin_fail_share"], info[f"{sim}_twin_wild_share"] = fail_share, wild_share
            info[f"{sim}_twin_failure_reasons"] = sorted({str(r["failed"])[:70] for r in vol_rows if r.get("failed")})
            if len(ok) < 3 or fail_share > C.MAX_FAIL_SHARE or wild_share > C.MAX_WILD_SHARE:
                info[f"{sim}_status"] = f"not evaluated: {fail_share:.0%} of unperturbed runs failed, {wild_share:.0%} of the rest moved more than {C.WILD_MOVE_BPS:.0f} bps in ten minutes"
                print(f"  {sim}: {info[f'{sim}_status']}", flush=True)
                if ok:
                    info[f"{sim}_volume"] = mean_volume(ok)
                continue
            info[f"{sim}_status"] = "evaluated"
            info[f"{sim}_volume"] = mean_volume(ok)
            info[f"{sim}_realism_error"], info[f"{sim}_realism_parts"] = realism_error(ok, [r for r in heldout if not r.get("failed")])
            cached(D / f"rows_{sim}.pkl", lambda: pool_map(policy_job, [(sim, W, s, info[f"{sim}_volume"], spec, ctx) for s in range(C.N_SIM[C.family(sim)])], procs, f"{sim} policies"))
        ipath = D / "info.json"  # merge with what earlier invocations (other simulators) recorded
        merged = {**(json.loads(ipath.read_text()) if ipath.exists() else {}), **info}
        ipath.write_text(json.dumps(merged, indent=2, default=str))
        print(json.dumps({k: v for k, v in info.items() if not k.endswith("_parts") and not k.endswith("_fit")}, indent=2, default=str), flush=True)
