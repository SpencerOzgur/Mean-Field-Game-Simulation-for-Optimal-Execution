"""Configuration of the controlled reference-market experiment (Phase 4)."""

from __future__ import annotations

from simrel.core import Task

TASK = Task(side="sell", horizon_s=600, n_slices=20, post_s=120)
SIZES = [0.05, 0.15]
LONG_TASK = Task(side="sell", horizon_s=1800, n_slices=60, post_s=120)  # task variant of addendum 8: 30 minutes, 60 decisions
LONG_SIZES = [0.02, 0.30]


def task_for(world: str | None) -> Task:
    """The execution task of a market: a market name ending in _long uses the 30-minute variant (addendum 8)."""
    return LONG_TASK if world and world.endswith("_long") else TASK


def sizes_for(world: str | None) -> list:
    return LONG_SIZES if world and world.endswith("_long") else SIZES

N_FIT_EPISODES = 6  # 30-minute unperturbed reference episodes each simulator may learn from
N_HELDOUT = 40  # held-out unperturbed reference episodes for the realism score
N_REFERENCE = 300  # reference-world seeds used to evaluate every candidate policy
N_SIM = {"hawkes": 100, "abides": 30, "mars": 10}  # seeds each fitted simulator uses to pick its policy
N_VOLUME = {"hawkes": 20, "abides": 12, "mars": 6}
N_CALIB_SEEDS = 6  # seeds per ABIDES grid point during calibration

# disjoint seed ranges
SEED_FIT, SEED_HELDOUT, SEED_VOLUME, SEED_MARS_INIT, SEED_CALIB = 1000, 2000, 10_000, 3000, 20_000

# Hawkes variants differ only in how much a market order excites later flow as a function of its size:
# weight = min((size / mean size) ** exponent, cap). Unperturbed data contain few orders large enough to tell them apart.
HAWKES_VARIANTS = {
    "hawkes": {"weight_exponent": 1.0, "weight_cap": 10.0},  # linear, capped at ten mean order sizes
    "hawkes_linear": {"weight_exponent": 1.0, "weight_cap": 100.0},  # linear, effectively uncapped
    "hawkes_sqrt": {"weight_exponent": 0.5, "weight_cap": 100.0},  # square-root (concave)
}


def family(sim: str) -> str:
    return "hawkes" if sim.startswith("hawkes") else sim

# stability screen applied to each fitted simulator's unperturbed runs before it is asked for a recommendation
WILD_MOVE_BPS, MAX_WILD_SHARE, MAX_FAIL_SHARE = 300.0, 0.25, 0.5
