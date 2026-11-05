"""
Stage 0: unperturbed baselines for every world, plus gates G1 (reproducible)
and G2 (an injected order changes the subsequent market).

Run from repo root:
    python experiments/simbench_00_baselines.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simbench.abides_world import WORLDS
from simbench.policies import Immediate, twap
from simbench.runner import Job, RESULTS_DIR, run_job, run_jobs

N_SEEDS = 200

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # G1: same seed twice gives an identical mid path; G2: a TWAP sell changes it.
    gates = {}
    for w in WORLDS:
        a = run_job(Job(w, 7, "sell", 0.0, 0, "none", Immediate()))
        b = run_job(Job(w, 7, "sell", 0.0, 0, "none", Immediate()))
        c = run_job(Job(w, 7, "sell", 0.0, 4000, "twap", twap(60)))
        gates[w] = {
            "G1_identical_repeat": bool(np.array_equal(a["mid30"], b["mid30"])),
            "G2_pre_start_identical": bool(np.array_equal(a["mid30"][:20], c["mid30"][:20])),
            "G2_post_start_differs": bool(not np.array_equal(a["mid30"][21:], c["mid30"][21:])),
            "G2_filled": float(c["filled"]),
            "wall_s": round(a["wall_s"], 2),
        }
        print(w, gates[w], flush=True)
    (RESULTS_DIR / "gates.json").write_text(json.dumps(gates, indent=2))

    jobs = [Job(w, s, "sell", 0.0, 0, "none", Immediate()) for w in WORLDS for s in range(N_SEEDS)]
    recs = run_jobs(jobs, "baselines", procs=6)
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in recs])
    if "timeout" in df:
        print("runaway episodes cut off:", df[df.timeout == True][["world", "seed"]].values.tolist(), flush=True)
        df = df[df.timeout != True]
    obs = [c for c in df.columns if c.startswith("obs_")]
    summ = df.groupby("world")[obs + ["mid_move_end_bps"]].agg(["mean", "std"])
    summ.to_csv(RESULTS_DIR / "observables_by_world.csv")
    vol = df.groupby("world")["obs_volume"].mean().round().to_dict()
    (RESULTS_DIR / "world_volume.json").write_text(json.dumps(vol, indent=2))
    print(df.groupby("world")[obs].mean().round(3).T.to_string())
