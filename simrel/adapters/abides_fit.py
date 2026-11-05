"""
Calibrate an ABIDES configuration to a target market by simulated moments.

ABIDES has no likelihood, so calibration is a search: each configuration on a
small grid is simulated without intervention for a few seeds, and the one
whose stylised facts are closest to the target's (simrel.metrics.realism) is
kept. The price level is set so one cent equals one tick of the target, which
matches the tick-to-price ratio; the grid then varies the number of value
traders, the number of market makers and the fundamental volatility.
"""

from __future__ import annotations

import dataclasses
import itertools

GRID = {
    "num_value_agents": [60, 102, 180],
    "num_mm": [0, 2],
    "fund_vol": [5e-5, 5e-4, 2e-3],  # at a mean price near 1000 cents these give fundamental moves of about 0.7, 7 and 27 cents per 30 minutes
}


def grid_specs(r_bar: int, prefix: str) -> list:
    from simbench.abides_world import WorldSpec

    specs = []
    for nv, nm, fv in itertools.product(*GRID.values()):
        specs.append(WorldSpec(name=f"{prefix}_v{nv}_m{nm}_f{fv:g}", description="calibration candidate", num_value_agents=nv, num_mm=nm, fund_vol=fv, r_bar=r_bar))
    return specs


def register(spec) -> str:
    """Make a WorldSpec available to simbench in this process (worlds are looked up by name)."""
    from simbench import abides_world

    abides_world.WORLDS[spec.name] = spec
    return spec.name


def spec_to_dict(spec) -> dict:
    return dataclasses.asdict(spec)
