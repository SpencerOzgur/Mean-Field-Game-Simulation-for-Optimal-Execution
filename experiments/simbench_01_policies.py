"""
Stage 1: run every execution policy in every world.

Sell side: immediate, POV, urgency rule, and the exponential schedule family
w_k ∝ exp(-b k/n) (b = 0 is TWAP). Buy side: TWAP only, for the buy/sell
asymmetry check. Parent sizes are fractions of each world's own mean
background volume over the 30-minute window (from stage 0).

Run from repo root (after simbench_00_baselines.py):
    python experiments/simbench_01_policies.py [n_seeds]
"""

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simbench.abides_world import WORLDS
from simbench.policies import POV, Immediate, Urgency, exp_schedule
from simbench.runner import Job, RESULTS_DIR, run_jobs

SIZES = [0.01, 0.05, 0.20]
B_GRID = [-2.0, 0.0, 2.0, 4.0, 8.0]
N_SLICES = 60


def sell_policies(size_frac: float) -> dict:
    pol = {f"exp_b{b:g}": exp_schedule(b, N_SLICES) for b in B_GRID}
    pol["immediate"] = Immediate()
    pol["pov"] = POV(rate=size_frac / (1.0 - size_frac))
    pol["urgency"] = Urgency()
    return pol


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    volume = json.loads((RESULTS_DIR / "world_volume.json").read_text())

    jobs = []
    for seed in range(n_seeds):  # seed-major order so a partial run is still balanced
        for w in WORLDS:
            for f in SIZES:
                qty = int(round(f * volume[w]))
                for name, pol in sell_policies(f).items():
                    jobs.append(Job(w, seed, "sell", f, qty, name, pol, n_slices=N_SLICES))
                jobs.append(Job(w, seed, "buy", f, qty, "exp_b0", exp_schedule(0.0, N_SLICES), n_slices=N_SLICES))
    print(f"{len(jobs)} episodes", flush=True)
    run_jobs(jobs, "policies", procs=6)
