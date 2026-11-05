"""
Phase 3 pilot: every candidate policy in every world, with unperturbed twins.

Stage 1 measures each world's mean background volume over the task window
(separate seeds). Stage 2 runs, for each world and seed, the twin and the 12
candidate policies at each parent size. Results are checkpointed; rerunning
skips finished (world, seed) pairs.

    python simrel/experiments/run_pilot.py <MarS checkout> <MarS data dir> [procs]
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simrel.configs.pilot import SIZES, TASK, VOLUME_SEED_OFFSET, WORLDS
from simrel.core import Task
from simrel.metrics.cost import episode_metrics
from simrel.policies.library import candidate_set

OUT = REPO_ROOT / "results" / "simrel"
MARS = {}


def _twin_sets(world: str, seed: int, jobs: dict[float, tuple[int, dict]]):
    """Run the twin and all policies for every size. jobs: size_frac -> (total_qty, policies). Returns size -> records."""
    family, key, _, _ = WORLDS[world]
    if family == "abides":
        from simrel.adapters.abides import run_twin_set

        return {f: run_twin_set(key, seed, Task(TASK.side, q, TASK.horizon_s, TASK.n_slices, TASK.post_s), pol) for f, (q, pol) in jobs.items()}
    if family == "hawkes":
        from simrel.adapters.hawkes import HawkesLOB, regimes
        from simrel.eventsim import run_twin_set

        P = regimes()[key]
        return {f: run_twin_set(lambda s: HawkesLOB(P, s), world, seed, Task(TASK.side, q, TASK.horizon_s, TASK.n_slices, TASK.post_s), pol)
                for f, (q, pol) in jobs.items()}
    from simrel.adapters.mars import run_twin_set
    from simrel.adapters.mars_local import setup_mars

    setup_mars(MARS["root"], MARS["data"])
    policies, sizes = {}, {}
    for f, (q, pol) in jobs.items():
        for name, p in pol.items():
            policies[f"{f}|{name}"], sizes[f"{f}|{name}"] = p, q
    recs = run_twin_set(seed, TASK, policies, repo_id=key, world=world, sizes=sizes)
    out = {f: {"none": recs["none"]} for f in jobs}
    for k, r in recs.items():
        if k != "none":
            f, name = k.split("|", 1)
            r.policy = name
            out[float(f)][name] = r
    return out


def volume_job(args):
    world, seed, mars = args
    warnings.filterwarnings("ignore")
    MARS.update(mars)
    try:
        r = _twin_sets(world, seed, {0.0: (0, {})})[0.0]["none"]
    except Exception as err:
        return {"world": world, "seed": seed, "volume": float("nan"), "failed": repr(err)[:200]}
    return {"world": world, "seed": seed, "volume": r.background_volume, "failed": r.failed}


def main_job(args):
    world, seed, volume, mars = args
    warnings.filterwarnings("ignore")
    MARS.update(mars)
    t0 = time.time()
    lot = 100 if WORLDS[world][0] != "abides" else 1
    jobs = {f: (max(int(f * volume) // lot * lot, lot), candidate_set(f)) for f in SIZES}
    rows = []
    try:
        sets = _twin_sets(world, seed, jobs)
    except Exception as err:
        return [{"world": world, "seed": seed, "failed": repr(err)[:200]}]
    for f, recs in sets.items():
        twin = recs["none"]
        pre = twin.grid_t < 0 if twin.grid_t is not None else None
        for name, r in recs.items():
            if name == "none":
                continue
            row = episode_metrics(r, twin)
            row["size_frac"] = f
            if pre is not None and not row.get("failed"):
                row["pre_move_bps"] = float((twin.grid_mid[pre][-1] / twin.grid_mid[pre][0] - 1) * 1e4)
                row["pre_spread_bps"] = float(np.nanmean(twin.grid_spread[pre]) / twin.arrival_mid * 1e4)
                m = min(len(r.grid_mid), len(twin.grid_mid))
                row["disp_path"] = (-(r.grid_mid[:m] - twin.grid_mid[:m]) / twin.arrival_mid * 1e4).astype(np.float32)
                row["grid_t"] = r.grid_t[:m].astype(np.float32)
            rows.append(row)
    for row in rows:
        row["job_wall_s"] = time.time() - t0
    return rows


def run_pool(fn, jobs, procs, label, path=None, done_rows=None):
    rows = list(done_rows or [])
    t0 = time.time()
    with Pool(procs) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
            rows.extend(res if isinstance(res, list) else [res])
            if path is not None and (i % 10 == 0 or i == len(jobs)):
                tmp = path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(rows, f)
                tmp.replace(path)
            if i % 20 == 0 or i == len(jobs):
                print(f"[{label}] {i}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    mars = {"root": sys.argv[1], "data": sys.argv[2]}
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    OUT.mkdir(parents=True, exist_ok=True)

    vpath = OUT / "pilot_volume.json"
    if vpath.exists():
        volume = json.loads(vpath.read_text())
    else:
        vraw = OUT / "raw" / "pilot_volume_rows.pkl"
        vraw.parent.mkdir(parents=True, exist_ok=True)
        have = pickle.load(open(vraw, "rb")) if vraw.exists() else []
        seen = {(r["world"], r["seed"]) for r in have}
        jobs = [(w, VOLUME_SEED_OFFSET + s, mars) for s in range(max(c[3] for c in WORLDS.values())) for w, c in WORLDS.items()
                if s < c[3] and (w, VOLUME_SEED_OFFSET + s) not in seen]
        res = run_pool(volume_job, jobs, procs, "volume", vraw, have)
        volume = {}
        for w in WORLDS:
            v = [r["volume"] for r in res if r["world"] == w and not r["failed"] and np.isfinite(r["volume"])]
            volume[w] = {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v), "failed": sum(1 for r in res if r["world"] == w and r["failed"]),
                         "failures": sorted({str(r["failed"])[:80] for r in res if r["world"] == w and r["failed"]})}
        vpath.write_text(json.dumps(volume, indent=2))
    print(json.dumps(volume, indent=2), flush=True)

    rpath = OUT / "raw" / "pilot_rows.pkl"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    done_rows = pickle.load(open(rpath, "rb")) if rpath.exists() else []
    done = {(r["world"], r["seed"]) for r in done_rows}
    jobs = [(w, s, volume[w]["mean"], mars) for s in range(max(c[2] for c in WORLDS.values())) for w, c in WORLDS.items()
            if s < c[2] and (w, s) not in done]
    print(f"{len(done)} (world, seed) pairs on disk, {len(jobs)} to run", flush=True)
    run_pool(main_job, jobs, procs, "pilot", rpath, done_rows)
