"""
Phase 1 check (Gate D): is an unperturbed MarS market stable?

Official initialisation (30 minutes of noise agent), then 30 minutes of MarS
with no intervention. Reports the mid, spread, depth and volume in successive
5-minute windows for several seeds and model sizes.

    python simrel/experiments/pilot_mars_stability.py <MarS checkout> <data dir> [n_seeds] [minutes]
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simrel.adapters.mars_local import LocalModelClient, setup_mars


def run(args):
    mars_root, data_dir, repo_id, seed, minutes = args
    setup_mars(mars_root, data_dir)
    import logging

    logging.disable(logging.CRITICAL)
    from pandas import Timedelta, Timestamp

    import market_simulation.examples.market_impact as mi

    mi.ModelClient = lambda **kw: LocalModelClient(repo_id=repo_id, seed=seed)
    start, init_end = Timestamp("2024-01-01 09:30:00"), Timestamp("2024-01-01 10:00:00")
    end = init_end + Timedelta(minutes=minutes)
    t0 = time.time()
    infos = mi.execute_single_simulation(mi.RolloutTask(seed, "000000", start, init_end, end, seed, False, 0))
    rows = []
    for k in range(-1, minutes // 5):
        a, b = init_end + Timedelta(minutes=5 * k), init_end + Timedelta(minutes=5 * (k + 1))
        w = [x for x in infos if a <= x.order.time < b]
        if not w:
            rows.append(None)
            continue
        mids = np.array([x.lob_snapshot.float_mid_price for x in w])
        vol = sum(t.volume for x in w if x.order.type in "BS" for t in x.transactions)
        d1 = np.mean([(x.lob_snapshot.bid_volumes[0] if x.lob_snapshot.bid_volumes else 0) + (x.lob_snapshot.ask_volumes[0] if x.lob_snapshot.ask_volumes else 0) for x in w]) / 2
        rows.append({"window": k, "orders": len(w), "ret_bps": float((mids[-1] / mids[0] - 1) * 1e4), "spread_ticks": float(np.mean([x.lob_snapshot.spread for x in w]) / 100),
                     "depth1": float(d1), "volume": int(vol), "mid_end": float(mids[-1])})
    return {"model": repo_id, "seed": seed, "wall_s": round(time.time() - t0, 1), "windows": rows}


if __name__ == "__main__":
    mars_root, data_dir = sys.argv[1], sys.argv[2]
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    minutes = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    jobs = [(mars_root, data_dir, m, s, minutes) for m in ("Don-Don/mars-order-2m", "Don-Don/mars-order-10m") for s in range(n_seeds)]
    with Pool(6) as pool:
        res = pool.map(run, jobs, chunksize=1)
    (REPO_ROOT / "simrel" / "reports" / "pilot_mars_stability.json").write_text(json.dumps(res, indent=2))
    for r in res:
        print(r["model"][-3:], "seed", r["seed"], "wall", r["wall_s"], "| window: orders ret_bps spread depth1 volume")
        for w in r["windows"]:
            if w:
                print(f"   {w['window']:>3} {w['orders']:>6} {w['ret_bps']:>9.0f} {w['spread_ticks']:>7.1f} {w['depth1']:>8.0f} {w['volume']:>9}")
