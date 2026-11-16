"""
External-validity markets (simrel/reports/review_audit.md, section 4): latent-liquidity and agent-mix families with
widened parameter ranges, screened on observable statistics towards the Nasdaq ranges.

Stream default_rng(20261117); blocks of BLOCK draws per family, latent first, then agent-mix; names EVl_d001..,
EVa_d001... Widened ranges (the original ranges in parentheses) aim at faster relaxation, stronger replenishment and
longer order-flow sign runs, which lower the persistence of the price response and make impact more concave.
"""

from __future__ import annotations

import numpy as np

from . import latent, sampled

SEED, BLOCK = 20261117, 200
LATENT = [("tau", 2.0, 600.0, "log"),            # (20, 600)
          ("limit", 0.8, 5.0, "log"),            # (0.8, 2.5)
          ("market", 0.06, 0.20, "log"),
          ("run_alpha", 1.1, 2.5, "uniform"),    # (1.3, 2.5)
          ("level_p", 0.10, 0.6, "uniform"),     # (0.25, 0.6)
          ("fund_vol", 0.02, 0.3, "log")]        # (0.05, 0.3)
AGENT = [("fund", 0.005, 0.30, "log"),           # (0.03, 0.30)
         ("fund_power", 1.2, 2.2, "uniform"),
         ("trend", 0.002, 0.12, "log"),          # (0.02, 0.12)
         ("trend_boost", 2.0, 7.0, "uniform"),
         ("regime_switch", 1 / 900, 1 / 200, "log"),
         ("withdraw", 0.1, 5.0, "log"),          # (1.0, 5.0)
         ("limit", 0.9, 4.0, "log"),             # (0.9, 1.8)
         ("fund_vol", 0.02, 0.30, "log")]        # (0.10, 0.30)


def draws() -> dict:
    """name -> (family, parameter values), every draw of both blocks in stream order."""
    rng = np.random.default_rng(SEED)
    out = {}
    for fam, ranges, prefix in (("latent", LATENT, "EVl_d"), ("am", AGENT, "EVa_d")):
        for k in range(1, BLOCK + 1):
            out[f"{prefix}{k:03d}"] = (fam, sampled.draw(rng, ranges))
    return out


def params(fam: str, vals: dict):
    return latent.LatentParams(**vals) if fam == "latent" else sampled.params("am", vals)
