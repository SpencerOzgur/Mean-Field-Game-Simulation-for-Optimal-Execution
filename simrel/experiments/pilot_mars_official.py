"""
Phase 1 pilot: reproduce MarS's official market-impact example
(market_simulation/examples/market_impact.py) with an in-process model.

Same setup as the official script: 30 minutes of noise-agent initialisation,
then a 5-minute MarS rollout, with and without the repository's own TWAP buy
agent at a fraction of baseline volume.

    python simrel/experiments/pilot_mars_official.py <MarS checkout> <data dir> [n_rollouts] [volume_ratio]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simrel.adapters.mars_local import LocalModelClient, setup_mars

if __name__ == "__main__":
    mars_root, data_dir = sys.argv[1], sys.argv[2]
    n_roll = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    setup_mars(mars_root, data_dir)

    from pandas import Timestamp

    import market_simulation.examples.market_impact as mi

    seeds = iter(range(10_000))
    mi.ModelClient = lambda **kw: LocalModelClient(seed=next(seeds))  # the only change to the official flow

    start, init_end, end = Timestamp("2024-01-01 09:30:00"), Timestamp("2024-01-01 10:00:00"), Timestamp("2024-01-01 10:05:00")

    def rollout(i, twap, vol):
        t0 = time.time()
        task = mi.RolloutTask(i, "000000", start, init_end, end, 0, twap, vol)
        infos = mi.execute_single_simulation(task)
        w = [x for x in infos if x.order.time >= init_end]
        mids = np.array([x.lob_snapshot.float_mid_price for x in w])
        spreads = np.array([x.lob_snapshot.spread for x in w])
        return infos, {"twap": twap, "orders_in_window": len(w), "mid_start": float(mids[0]), "mid_end": float(mids[-1]),
                       "ret_bps": float((mids[-1] / mids[0] - 1) * 1e4), "mean_spread_ticks": float(spreads.mean() / 100),
                       "max_spread_ticks": float(spreads.max() / 100), "wall_s": round(time.time() - t0, 1)}

    base = [rollout(i, False, 0) for i in range(n_roll)]
    avg_vol = mi.calculate_average_volume([b[0] for b in base], init_end, end)
    target = int(avg_vol * ratio) // 100 * 100
    twap = [rollout(i, True, target) for i in range(n_roll)]
    out = {"avg_baseline_volume": avg_vol, "twap_target_volume": target, "baseline": [b[1] for b in base], "twap_buy": [t[1] for t in twap],
           "mean_ret_bps_baseline": float(np.mean([b[1]["ret_bps"] for b in base])), "mean_ret_bps_twap": float(np.mean([t[1]["ret_bps"] for t in twap]))}
    print(json.dumps(out, indent=2))
    (REPO_ROOT / "simrel" / "reports" / "pilot_mars_official.json").write_text(json.dumps(out, indent=2))
