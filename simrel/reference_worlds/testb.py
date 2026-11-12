"""
Test B markets of design addendum 10 (simrel/reports/lob_impact_design_addendum10.md, section 2).

B-U (unfiltered, primary): stream default_rng(20261112). Each family takes a fixed block of BLOCK_U draws from the
stream, in the order queue-reactive, agent-mix, latent; within a family the first 16 draws that pass the viability rule
(run_reference.py) are the test markets. Ranges are those of sampled.py (QR_RANGES, AM_RANGES) and latent.py (RANGES).

B-F (opportunity-rich, secondary): stream default_rng(20261113), blocks of BLOCK_F draws, agent-mix then latent; a draw
is kept if it is viable and the true generator at 100 seeds beats always-TWAP by at least 1.5 bp of regret averaged over
the two sizes; the first 8 kept per family are the markets.

Markets are registered under draw-order names (TBq_d01, TBa_d01, TBl_d01; TFa_d001, TFl_d001); the listing
results/simrel/raw/testb_markets.json records which draws were accepted and maps them, in order, to the addendum's names
(TBq01-16, TBa01-16, TBl01-16; TFa01-08, TFl01-08).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import latent, sampled

SEED_U, SEED_F = 20261112, 20261113
BLOCK_U, BLOCK_F = 40, 300
N_U, N_F = 16, 8
LISTING = Path(__file__).resolve().parents[2] / "results" / "simrel" / "raw" / "testb_markets.json"
PREFIX = {("U", "qr"): "TBq_d", ("U", "am"): "TBa_d", ("U", "latent"): "TBl_d", ("F", "am"): "TFa_d", ("F", "latent"): "TFl_d"}


def _one(rng, fam: str) -> dict:
    if fam == "qr":
        return sampled.draw(rng, sampled.QR_RANGES)
    if fam == "am":
        return sampled.draw(rng, sampled.AM_RANGES)
    return latent.draw(rng)


def draws(which: str = "U") -> dict:
    """draw name -> (family, parameter values), for every draw of the blocks, in stream order."""
    rng = np.random.default_rng(SEED_U if which == "U" else SEED_F)
    fams, block = (["qr", "am", "latent"], BLOCK_U) if which == "U" else (["am", "latent"], BLOCK_F)
    width = 2 if which == "U" else 3
    out = {}
    for fam in fams:
        for k in range(1, block + 1):
            out[f"{PREFIX[(which, fam)]}{k:0{width}d}"] = (fam, _one(rng, fam))
    return out


def all_draws() -> dict:
    return {**draws("U"), **draws("F")}


def accepted() -> dict:
    """addendum name (TBq01, ...) -> draw name, once the viability (and, for B-F, headroom) screen has been run."""
    if not LISTING.exists():
        return {}
    return json.loads(LISTING.read_text())["accepted"]
