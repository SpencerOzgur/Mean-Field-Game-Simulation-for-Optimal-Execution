"""
Run the TRADES arm: unperturbed twins first (to measure background volume),
then every schedule for every seed. Episodes are separate processes; existing
outputs are skipped, so the batch can be resumed.

    python simbench/trades/run_trades_batch.py --deepmarket /path/to/DeepMarket \
        --ckpt <ckpt> --seeds 8 --procs 6 --size 0.10
"""

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "simbench" / "raw" / "trades"
POLICIES = ["exp:-2", "exp:0", "exp:2", "exp:8", "immediate"]


def run_all(cmds, procs):
    running, t0, done = [], time.time(), 0
    cmds = list(cmds)
    while cmds or running:
        while cmds and len(running) < procs:
            running.append(subprocess.Popen(cmds.pop(0), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(2)
        still = [p for p in running if p.poll() is None]
        done += len(running) - len(still)
        if len(still) != len(running):
            print(f"{done} episodes finished, {time.time() - t0:.0f}s", flush=True)
        running = still


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepmarket", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--size", type=float, default=0.10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def cmd(seed, qty, policy, path):
        return [sys.executable, str(HERE / "run_trades_episode.py"), "--deepmarket", args.deepmarket, "--ckpt", args.ckpt,
                "--seed", str(seed), "--side", "sell", "--qty", str(qty), "--policy", policy, "--out", str(path)]

    base = {s: OUT / f"base_s{s}.pkl" for s in range(args.seeds)}
    run_all([cmd(s, 0, "exp:0", p) for s, p in base.items() if not p.exists()], args.procs)
    vols = []
    for p in base.values():
        v = pickle.load(open(p, "rb"))["window_volume"]
        vols.append(float(np.sum(v)))
    qty = int(round(args.size * np.mean(vols)))
    print("mean background volume in window:", np.mean(vols), "-> parent qty", qty, flush=True)

    jobs = []
    for s in range(args.seeds):  # seed-major, so a partial batch is balanced
        for pol in POLICIES:
            path = OUT / f"{pol.replace(':', '')}_s{s}.pkl"
            if not path.exists():
                jobs.append(cmd(s, qty, pol, path))
    run_all(jobs, args.procs)
