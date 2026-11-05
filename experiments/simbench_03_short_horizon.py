"""
Stage 3: the ABIDES worlds on the task the TRADES arm can afford.

TRADES episodes are expensive, so that arm uses a 10-minute sell of 10% of
background volume in 20 slices. This stage runs the same task in every ABIDES
world so the two families are compared on identical task definitions.

Run from repo root:
    python experiments/simbench_03_short_horizon.py [n_seeds]
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simbench.abides_world import WORLDS
from simbench.analysis import baseline_index, boot_ci, to_frame
from simbench.policies import Immediate, exp_schedule
from simbench.runner import Job, RESULTS_DIR, run_jobs

H, N, POST, SIZE = 600, 20, 120, 0.10
B_GRID = [-2.0, 0.0, 2.0, 8.0]

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    kw = dict(horizon_s=H, n_slices=N, post_s=POST)
    base = run_jobs([Job(w, s, "sell", 0.0, 0, "none", Immediate(), **kw) for s in range(n_seeds) for w in WORLDS], "baselines_h600")
    vol = pd.DataFrame([{"world": r["world"], "v": r["obs_volume"]} for r in base if not r.get("timeout")]).groupby("world").v.mean()

    pol = {f"exp_b{b:g}": exp_schedule(b, N) for b in B_GRID}
    pol["immediate"] = Immediate()
    jobs = [Job(w, s, "sell", SIZE, int(round(SIZE * vol[w])), name, p, **kw)
            for s in range(n_seeds) for w in WORLDS for name, p in pol.items()]
    recs = run_jobs(jobs, "policies_h600")

    df = to_frame(recs, baseline_index("baselines_h600"))
    rows = []
    for (w, p), d in df.groupby(["world", "policy"]):
        lo, hi = boot_ci(d["cost_pen_bps"].to_numpy())
        rows.append({"world": w, "policy": p, "cost_bps": d["cost_pen_bps"].mean(), "ci_lo": lo, "ci_hi": hi,
                     "fill_frac": d["fill_frac"].mean(), "disp_end_bps": d["disp_end_bps"].mean(),
                     "disp_post_bps": d["disp_post_bps"].mean(), "n": len(d)})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(RESULTS_DIR / "table_short_horizon_abides.csv", index=False)
    print(out.pivot(index="policy", columns="world", values="cost_bps").round(2).to_string())
