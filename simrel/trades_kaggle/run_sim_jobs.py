"""
Run TRADES episodes on a Kaggle GPU: unperturbed twins and policy episodes, several in parallel.

    python run_sim_jobs.py <DeepMarket dir> <checkpoint> <jobs file> <out dir> --procs 4 --hours 10.5

Each line of the jobs file is "<date> <seed> <qty> <policy>". Finished episodes are skipped on a rerun. No new episode
starts after --hours, and any episode still running 0.6 h later is killed, so the notebook always packages its output
inside the platform's 12-hour limit.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deepmarket")
    ap.add_argument("ckpt")
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--hours", type=float, default=10.5)
    ap.add_argument("--ddim-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda", help="one device, or a comma-separated list assigned to jobs in turn (e.g. cuda,cpu)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    jobs = [line.split() for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    t0 = time.time()

    running = {}
    hard_stop = t0 + (args.hours + 0.6) * 3600  # episodes still running then are killed, so the output is always saved

    devices = args.device.split(",")

    def run_guarded(indexed):
        n, job = indexed
        device = devices[n % len(devices)]
        date, seed, qty, policy = job
        stem = f"gen_{date}_s{seed}" if int(qty) == 0 else f"pol_{date}_s{seed}_{policy}"
        path = out / f"{stem}.pkl"
        if path.exists():
            return f"{stem} cached"
        if time.time() > deadline:
            return f"{stem} skipped (time limit)"
        cmd = [sys.executable, "-u", str(HERE / "run_trades_episode.py"), "--deepmarket", args.deepmarket, "--ckpt", args.ckpt,
               "--seed", seed, "--qty", qty, "--side", "sell", "--policy", policy, "--symbol", "INTC", "--date", date,
               "--ddim-steps", str(args.ddim_steps), "--device", device, "--out", str(path)]
        w0 = time.time()
        print(f"{stem} started on {device} (elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
        with open(out / f"{stem}.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            running[stem] = proc
            try:
                rc = proc.wait(timeout=max(hard_stop - time.time(), 1))
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = "killed at the hard stop"
            running.pop(stem, None)
        return f"{stem} on {device}: rc={rc} {(time.time() - w0) / 60:.1f} min (elapsed {(time.time() - t0) / 3600:.2f} h)"

    from concurrent.futures import as_completed

    with ThreadPoolExecutor(args.procs) as pool:
        futures = [pool.submit(run_guarded, (n, j)) for n, j in enumerate(jobs)]
        for f in as_completed(futures):  # report each episode when it finishes, in any order
            print(f.result(), flush=True)

if __name__ == "__main__":
    main()
