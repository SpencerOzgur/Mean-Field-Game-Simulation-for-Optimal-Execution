"""
Mechanism-ablation markets (simrel/reports/lob_impact_design_addendum6.md, experiment 2).

One mechanism varied at a time around two base markets (queue-reactive R1a defaults, agent-mix R2a defaults), every
other parameter at its base value. The order size is fixed in shares at the base market's, so only the mechanism changes.
"""

from __future__ import annotations

from .agent_mix import AMParams
from .queue_reactive import QRParams

BASE_VOLUME = {"qr": 42968.0, "am": 95845.0}  # reference volume of R1a_qr_resilient and R2a_am_anchored
KNOBS = {
    "join": ("qr", [0.3, 0.6, 1.2, 2.4, 4.8], 1.2),
    "theta": ("qr", [0.4, 0.8, 1.2, 2.0, 2.8], 1.2),
    "fund": ("am", [0.025, 0.05, 0.1, 0.2, 0.4], 0.1),
    "withdraw": ("am", [0.5, 1.0, 2.0, 4.0, 6.0], 2.0),
    "trend": ("am", [0.0125, 0.025, 0.05, 0.1, 0.2], 0.05),
}


def name(knob: str, level: float) -> str:
    fam, _, base = KNOBS[knob]
    return f"X{fam}_base" if level == base else f"X{fam}_{knob}_{level:g}"


def markets() -> dict:
    """name -> (family, params, knob, level)."""
    out = {}
    for knob, (fam, levels, base) in KNOBS.items():
        for lv in levels:
            p = QRParams(**{knob: lv}) if fam == "qr" else AMParams(**{knob: lv})
            out[name(knob, lv)] = (fam, p, knob, lv)
    return out
