"""
Download Nasdaq ITCH sample days and convert chosen symbols to LOBSTER-style arrays.

The raw file (3.5-6 GB compressed) is streamed through gunzip into the parser
and never written to disk. Output: <out>/<SYMBOL>_<YYYY-MM-DD>.npz with
`message` (N x 6) and `book` (N x 40) int64 arrays for 09:30-16:00.

    python simrel/experiments/data_00_fetch_itch.py <out dir> [symbols,comma] [days,comma]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simrel.data.itch import parse_stream

BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
# file name on the server -> trading date; all are public sample days
DAYS = {
    "01302019.NASDAQ_ITCH50.gz": "2019-01-30",
    "03272019.NASDAQ_ITCH50.gz": "2019-03-27",
    "07302019.NASDAQ_ITCH50.gz": "2019-07-30",
    "08302019.NASDAQ_ITCH50.gz": "2019-08-30",
    "S101819-v50.txt.gz": "2019-10-18",
    "10302019.NASDAQ_ITCH50.gz": "2019-10-30",
    "12302019.NASDAQ_ITCH50.gz": "2019-12-30",
    "01302020.NASDAQ_ITCH50.gz": "2020-01-30",
}

if __name__ == "__main__":
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    symbols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["INTC"]
    days = sys.argv[3].split(",") if len(sys.argv) > 3 else list(DAYS.values())
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for fname, day in DAYS.items():
        if day not in days or all((out / f"{s}_{day}.npz").exists() for s in symbols):
            continue
        t0 = time.time()
        curl = subprocess.Popen(["curl", "-s", "-L", BASE + fname], stdout=subprocess.PIPE)
        gz = subprocess.Popen(["gzip", "-dc"], stdin=curl.stdout, stdout=subprocess.PIPE, bufsize=1 << 26)
        last = [0]

        def progress(n, last=last):
            if n - last[0] > 50_000_000:
                last[0] = n
                print(f"  {day}: {n / 1e6:.0f}M messages, {time.time() - t0:.0f}s", flush=True)

        books = parse_stream(gz.stdout, symbols, progress=progress)
        gz.wait()
        curl.wait()
        for s in symbols:
            if s not in books:
                print(f"  {day}: {s} not found in the stock directory", flush=True)
                continue
            b = books[s]
            msg, book = b.arrays()
            np.savez_compressed(out / f"{s}_{day}.npz", message=msg, book=book)
            manifest[f"{s}_{day}"] = {"source": BASE + fname, "date": day, "symbol": s, "events": int(len(msg)), "crossed_snapshots": b.n_crossed,
                                      "unknown_references": b.n_unknown_ref, "executions": int((msg[:, 1] == 4).sum()), "hidden_executions": int((msg[:, 1] == 5).sum())}
            print(f"{day} {s}: {len(msg):,} events, crossed {b.n_crossed}, unknown refs {b.n_unknown_ref}, {time.time() - t0:.0f}s", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
