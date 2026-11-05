"""
Randomly drawn reference markets (simrel/reports/lob_impact_design_addendum4.md).

Parameters are drawn log-uniformly (uniform for exponents and powers) over fixed ranges from one random stream; the
first five draws of each family that pass the viability rule are kept. Accepted markets are listed in
results/simrel/raw/sampled_markets.json, which registry() in data.py reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .agent_mix import AMParams
from .queue_reactive import QRParams

SEED = 20261103
LISTING = Path(__file__).resolve().parents[2] / "results" / "simrel" / "raw" / "sampled_markets.json"
QR_RANGES = [("join", 0.5, 2.5, "log"), ("q0", 800.0, 4000.0, "log"), ("deep", 0.25, 1.0, "log"), ("cancel", 0.3, 0.9, "log"),
             ("market", 0.08, 0.18, "log"), ("theta", 0.8, 3.0, "uniform"), ("init_qty", 400, 1500, "log")]
AM_RANGES = [("fund", 0.03, 0.30, "log"), ("fund_power", 1.2, 2.2, "uniform"), ("trend", 0.02, 0.12, "log"), ("trend_boost", 2.0, 7.0, "uniform"),
             ("regime_switch", 1 / 900, 1 / 200, "log"), ("withdraw", 1.0, 5.0, "log"), ("limit", 0.9, 1.8, "log"), ("fund_vol", 0.10, 0.30, "log")]


def draw(rng, ranges) -> dict:
    out = {}
    for name, lo, hi, kind in ranges:
        v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind == "log" else float(rng.uniform(lo, hi))
        out[name] = int(round(v)) if name == "init_qty" else v
    return out


def params(family: str, values: dict):
    return QRParams(**values) if family == "qr" else AMParams(**values)


def accepted() -> dict:
    """name -> (family, parameter values) of the accepted markets."""
    if not LISTING.exists():
        return {}
    return {k: (v["family"], v["params"]) for k, v in json.loads(LISTING.read_text())["accepted"].items()}
