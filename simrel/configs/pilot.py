"""Pilot configuration (Phase 3): worlds, task, sizes and seed counts."""

from simrel.core import Task

# The shared task: sell over 10 minutes in 20 decisions, then watch for 2 minutes.
TASK = Task(side="sell", horizon_s=600, n_slices=20, post_s=120)

# Parent size as a fraction of the world's mean background volume in the task window.
SIZES = [0.05, 0.15]

# world -> (family, family-specific key, seeds for the main pass, seeds for the volume pass)
WORLDS = {
    "A1_rmsc04": ("abides", "A1_rmsc04", 40, 20),
    "A5_deep_mm": ("abides", "A5_deep_mm", 40, 20),
    "H1_liquid": ("hawkes", "H1_liquid", 60, 20),
    "H2_thin": ("hawkes", "H2_thin", 60, 20),
    "M10_mars": ("mars", "Don-Don/mars-order-10m", 8, 12),  # seeds 0-7, 9, 10 were completed within the compute budget
    "M2_mars": ("mars", "Don-Don/mars-order-2m", 11, 12),  # seeds 0-10 completed
}

# Seeds: volume pass uses 10_000 + i so it never overlaps the main pass.
VOLUME_SEED_OFFSET = 10_000
