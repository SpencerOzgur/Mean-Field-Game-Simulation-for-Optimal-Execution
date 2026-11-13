"""
Identifiability grid of design addendum 10, section 5 (simrel/reports/lob_impact_design_addendum10.md).

Agent-mix markets around the R2a defaults, 3^4 = 81: fundamental anchoring (fund), trend following (trend), liquidity
withdrawal (withdraw) and liquidity replenishment (limit). Every other parameter at its default; the order size is fixed
in shares at the agent-mix base market's, as in the ablation markets.
"""

from __future__ import annotations

import itertools

from .agent_mix import AMParams

LEVELS = {"fund": (0.025, 0.1, 0.4), "trend": (0.0125, 0.05, 0.2), "withdraw": (0.5, 2.0, 6.0), "limit": (0.7, 1.4, 2.8)}


def name(fund: float, trend: float, withdraw: float, limit: float) -> str:
    return f"IG_f{fund:g}_t{trend:g}_w{withdraw:g}_l{limit:g}"


def markets() -> dict:
    """name -> (parameters, levels)."""
    out = {}
    for f, t, w, lim in itertools.product(*LEVELS.values()):
        out[name(f, t, w, lim)] = (AMParams(fund=f, trend=t, withdraw=w, limit=lim), {"fund": f, "trend": t, "withdraw": w, "limit": lim})
    return out
