"""
Parallel episode runner. Each job is one (world, seed, task, policy) episode;
results are reduced to a compact record and pickled.
"""

from __future__ import annotations

import logging
import pickle
import signal
import time
import traceback
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "simbench"
RAW_DIR = RESULTS_DIR / "raw"


@dataclass(frozen=True)
class Job:
    world: str
    seed: int
    side: str
    size_frac: float  # parent size as a fraction of the world's mean background volume
    total_qty: int
    policy_name: str
    policy: Any
    horizon_s: int = 1800
    n_slices: int = 60
    post_s: int = 300


EPISODE_TIMEOUT_S = 90
EPISODE_ATTEMPTS = 2


class EpisodeTimeout(BaseException):  # BaseException so no "except Exception" inside ABIDES can swallow it
    pass


def _on_timeout(signum, frame):
    raise EpisodeTimeout("".join(traceback.format_stack(frame, limit=6)))


def _acf1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else np.nan


def _observables(ep) -> dict:
    """Stylised-fact statistics of an unperturbed episode (execution window)."""
    g = ep.grid_s
    m = (g >= 0) & (g <= ep.task["horizon_s"])
    mid = ep.mid[m]
    r10 = np.diff(np.log(mid[::10])) * 1e4
    tr = ep.trades
    tw = tr[(tr[:, 0] >= 0) & (tr[:, 0] <= ep.task["horizon_s"])] if len(tr) else tr
    bins = np.arange(0, ep.task["horizon_s"] + 1, 30)
    ofi = np.histogram(tw[:, 0], bins=bins, weights=tw[:, 1])[0] if len(tw) else np.zeros(len(bins) - 1)
    sd = float(r10.std())
    return {
        "obs_spread_bps": float(np.nanmean(ep.spread[m]) / mid[0] * 1e4),
        "obs_depth5": float(np.mean(0.5 * (ep.bid_depth[m] + ep.ask_depth[m]))),
        "obs_vol10s_bps": sd,
        "obs_ret_acf1": _acf1(r10),
        "obs_absret_acf1": _acf1(np.abs(r10)),
        "obs_kurtosis": float(((r10 - r10.mean()) ** 4).mean() / sd**4 - 3.0) if sd > 0 else np.nan,
        "obs_volume": float(np.abs(tw[:, 1]).sum()) if len(tw) else 0.0,
        "obs_n_trades": float(len(tw)),
        "obs_trade_size": float(np.abs(tw[:, 1]).mean()) if len(tw) else np.nan,
        "obs_ofi_std": float(ofi.std()),
        "obs_ofi_acf1": _acf1(ofi),
    }


def run_job(job: Job) -> dict:
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from .abides_world import Task, run_episode

    task = Task(side=job.side, total_qty=job.total_qty, horizon_s=job.horizon_s, n_slices=job.n_slices, post_s=job.post_s)
    t0 = time.time()
    # Occasionally an episode stalls: the process spins for many minutes on a run that
    # normally takes seconds, and the same seed completes normally when rerun. Each
    # attempt is cut off after EPISODE_TIMEOUT_S and retried; if every attempt stalls
    # the episode is recorded as a timeout rather than silently dropped.
    signal.signal(signal.SIGALRM, _on_timeout)
    ep, attempts, where = None, 0, ""
    while ep is None and attempts < EPISODE_ATTEMPTS:
        attempts += 1
        signal.setitimer(signal.ITIMER_REAL, EPISODE_TIMEOUT_S, 5.0)
        try:
            ep = run_episode(job.world, job.seed, task, job.policy, job.policy_name)
        except EpisodeTimeout as e:
            where = str(e)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    if ep is None:
        return {"world": job.world, "seed": job.seed, "side": job.side, "size_frac": job.size_frac,
                "total_qty": job.total_qty, "policy": job.policy_name, "horizon_s": job.horizon_s,
                "wall_s": time.time() - t0, "timeout": True, "attempts": attempts, "stalled_at": where}
    n = job.n_slices
    dt = job.horizon_s / n

    sl_mid = np.full(n, np.nan, dtype=np.float64)
    sl_req = np.zeros(n, dtype=np.float64)
    for s in ep.slices:
        sl_mid[s["k"]] = np.nan if s["mid"] is None else s["mid"]
        sl_req[s["k"]] = s["qty"]
    f_qty = np.zeros(n)
    f_not = np.zeros(n)
    if len(ep.fills):
        k = np.clip((ep.fills[:, 0] // dt).astype(int), 0, n - 1)
        np.add.at(f_qty, k, ep.fills[:, 2])
        np.add.at(f_not, k, ep.fills[:, 1] * ep.fills[:, 2])

    rec = {
        "world": job.world,
        "seed": job.seed,
        "side": job.side,
        "size_frac": job.size_frac,
        "total_qty": job.total_qty,
        "policy": job.policy_name,
        "horizon_s": job.horizon_s,
        "wall_s": time.time() - t0,
        "attempts": attempts,
        "stalled_at": where,
        **ep.scalars,
        "slice_mid": sl_mid.astype(np.float32),
        "slice_req": sl_req.astype(np.float32),
        "fill_qty": f_qty.astype(np.float32),
        "fill_notional": f_not,
        "mid30": ep.mid[(ep.grid_s >= -600)][::30].astype(np.float64),
    }
    if job.total_qty == 0:
        rec.update(_observables(ep))
    return rec


def _key(r) -> tuple:
    get = r.get if isinstance(r, dict) else lambda k: getattr(r, k if k != "policy" else "policy_name")
    return (get("world"), get("seed"), get("side"), get("size_frac"), get("policy"))


def run_jobs(jobs: list[Job], name: str, procs: int = 6, chunksize: int = 1, save_every: int = 100) -> list[dict]:
    """Run jobs in a process pool. Results are checkpointed, and a rerun with the
    same name skips episodes that are already on disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{name}.pkl"
    recs = load(name) if out.exists() else []
    done = {_key(r) for r in recs}
    todo = [j for j in jobs if _key(j) not in done]
    print(f"[{name}] {len(done)} on disk, {len(todo)} to run", flush=True)

    def save():
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(recs, f)
        tmp.replace(out)

    t0 = time.time()
    with Pool(procs) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_job, todo, chunksize=chunksize), 1):
            recs.append(rec)
            if i % save_every == 0 or i == len(todo):
                save()
                print(f"[{name}] {i}/{len(todo)} episodes, {time.time() - t0:.0f}s", flush=True)
    save()
    return recs


def load(name: str) -> list[dict]:
    with open(RAW_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
