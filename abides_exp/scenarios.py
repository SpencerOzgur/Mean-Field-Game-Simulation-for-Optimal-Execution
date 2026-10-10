"""
Scenario library. A scenario builds one ABIDES config for (seed, treated):
treated=True has the strategy or event switched on; treated=False is its baseline,
with the same agents present (inert) so the two runs match until the event starts.

Each builder returns (cfg, handles):
  handles["kind"]     "execution" | "adversary" | "event"
  handles["window"]   (start_s, end_s) seconds after the open, where the action is
  handles["subject"]  the agent being evaluated (or None)
  handles["victim"]   a second agent whose outcome we track (predator scenario)
  handles["oracle"]   the run's oracle, to recover the true fundamental
  handles["meta"]     scenario parameters (also written to meta.json)

Add a scenario: write a builder and register it in SCENARIOS.
All times are seconds after the 09:30 open. The default simulation ends at 11:00.
"""
from typing import Dict

import numpy as np

from abides_core.utils import str_to_ns
from abides_markets.configs import rmsc04
from abides_markets.utils import config_add_agents

from abides_exp.agents import STRATEGIES
from abides_exp.agents.scripted import (
    ScriptedTrader,
    dump_script,
    ignition_script,
    predator_script,
    spoof_script,
)
from abides_exp.events import jump_offset, shift_fundamental, withdraw_market_makers

TICKER = "ABM"
NS = 10**9
END_TIME = "11:00:00"

# Shared defaults. ~61k shares trade in a 30-minute window of the default market.
EXEC = dict(start_s=600, duration_s=1800, n_slices=30, qty=12_000, side="sell")
EVENT_AT = 1800


def _base(seed: int, market: Dict):
    cfg = rmsc04.build_config(seed=seed, end_time=END_TIME, ticker=TICKER,
                              stdout_log_level="WARNING", log_orders=False, **market)
    mkt_open = cfg["agents"][0].mkt_open
    # Every run uses ShiftedOracle (zero offset unless an event sets one). Swapping the
    # class does not touch the oracle's RNG, so baselines are unaffected.
    shift_fundamental(cfg, lambda t: 0.0)
    return cfg, mkt_open


def _exec_agent(cfg, strategy, seed, qty, start_s, duration_s, n_slices, side, **kw):
    return STRATEGIES[strategy](
        id=len(cfg["agents"]), symbol=TICKER, total_qty=qty, side=side,
        start_delay=int(start_s * NS), duration=int(duration_s * NS), n_slices=n_slices,
        random_state=np.random.RandomState(seed + 1), **kw)


def _scripted(cfg, seed, script, active, name, offset=2):
    return ScriptedTrader(id=len(cfg["agents"]), symbol=TICKER, script=script, active=active,
                          name=name, random_state=np.random.RandomState(seed + offset))


# ---------------------------------------------------------------------------
# Execution strategies (subject sells EXEC["qty"]; baseline: same agent, qty 0)
# ---------------------------------------------------------------------------
def make_exec(strategy, **strategy_kw):
    def build(seed, treated, market, p=None):
        p = {**EXEC, **(p or {})}
        cfg, mkt_open = _base(seed, market)
        agent = _exec_agent(cfg, strategy, seed, p["qty"] if treated else 0, p["start_s"],
                            p["duration_s"], p["n_slices"], p["side"], **strategy_kw)
        cfg = config_add_agents(cfg, [agent])
        return cfg, {"kind": "execution", "window": (p["start_s"], p["start_s"] + p["duration_s"]),
                     "subject": agent, "victim": None, "oracle": cfg["custom_properties"]["oracle"],
                     "meta": {**p, "strategy": strategy, **strategy_kw}}
    return build


# ---------------------------------------------------------------------------
# Adversaries
# ---------------------------------------------------------------------------
def spoofing(seed, treated, market, p=None):
    p = {"start_s": 600, "duration_s": 1800, "cycle_s": 30, "spoof_qty": 5000,
         "trade_qty": 300, "ticks": 1, **(p or {})}
    cfg, _ = _base(seed, market)
    script = spoof_script(p["start_s"], p["duration_s"], p["cycle_s"], p["spoof_qty"],
                          p["trade_qty"], p["ticks"])
    agent = _scripted(cfg, seed, script, treated, "Spoofer")
    cfg = config_add_agents(cfg, [agent])
    return cfg, {"kind": "adversary", "window": (p["start_s"], p["start_s"] + p["duration_s"]),
                 "subject": agent, "victim": None, "oracle": cfg["custom_properties"]["oracle"],
                 "meta": p}


def momentum_ignition(seed, treated, market, p=None):
    p = {"start_s": 600, "burst_qty": 3000, "burst_s": 30, "hold_s": 60, "unwind_s": 600, **(p or {})}
    cfg, _ = _base(seed, market)
    script = ignition_script(p["start_s"], p["burst_qty"], p["burst_s"], hold_s=p["hold_s"],
                             unwind_s=p["unwind_s"])
    agent = _scripted(cfg, seed, script, treated, "Igniter")
    cfg = config_add_agents(cfg, [agent])
    end = p["start_s"] + p["burst_s"] + p["hold_s"] + p["unwind_s"]
    return cfg, {"kind": "adversary", "window": (p["start_s"], end), "subject": agent,
                 "victim": None, "oracle": cfg["custom_properties"]["oracle"], "meta": p}


def predatory(seed, treated, market, p=None):
    """A TWAP seller (present in both runs) and a predator who knows its schedule."""
    p = {**EXEC, "predator_qty": 3000, "frontrun_s": 120, **(p or {})}
    cfg, _ = _base(seed, market)
    victim = _exec_agent(cfg, "twap", seed, p["qty"], p["start_s"], p["duration_s"],
                         p["n_slices"], p["side"])
    cfg = config_add_agents(cfg, [victim])
    script = predator_script(p["start_s"], p["duration_s"], p["predator_qty"], p["frontrun_s"],
                             victim_side=p["side"])
    pred = _scripted(cfg, seed, script, treated, "Predator", offset=3)
    cfg = config_add_agents(cfg, [pred])
    return cfg, {"kind": "adversary", "window": (p["start_s"], p["start_s"] + p["duration_s"]),
                 "subject": pred, "victim": victim, "oracle": cfg["custom_properties"]["oracle"],
                 "meta": p}


# ---------------------------------------------------------------------------
# Market events
# ---------------------------------------------------------------------------
def fundamental_jump(seed, treated, market, p=None):
    p = {"at_s": EVENT_AT, "jump_bps": -50.0, **(p or {})}
    cfg, mkt_open = _base(seed, market)
    oracle = cfg["custom_properties"]["oracle"]
    if treated:
        oracle.offset_fn = jump_offset(mkt_open, p["at_s"], p["jump_bps"],
                                       r_bar=oracle.symbols[TICKER]["r_bar"])
    return cfg, {"kind": "event", "window": (p["at_s"], p["at_s"] + 1800), "subject": None,
                 "victim": None, "oracle": oracle, "meta": p}


def mm_withdrawal(seed, treated, market, p=None):
    p = {"start_s": EVENT_AT, "duration_s": 600, **(p or {})}
    cfg, mkt_open = _base(seed, market)
    if treated:
        withdraw_market_makers(cfg, mkt_open, p["start_s"], p["start_s"] + p["duration_s"])
    return cfg, {"kind": "event", "window": (p["start_s"], p["start_s"] + p["duration_s"]),
                 "subject": None, "victim": None, "oracle": cfg["custom_properties"]["oracle"],
                 "meta": p}


def flash_crash(seed, treated, market, p=None):
    """A fire sale (aggressive market sells) while market makers step away."""
    p = {"at_s": EVENT_AT, "qty": 18_000, "dump_s": 120, "mm_off_s": 300, **(p or {})}
    cfg, mkt_open = _base(seed, market)
    if treated:
        withdraw_market_makers(cfg, mkt_open, p["at_s"], p["at_s"] + p["mm_off_s"])
    agent = _scripted(cfg, seed, dump_script(p["at_s"], p["qty"], p["dump_s"]), treated, "FireSeller")
    cfg = config_add_agents(cfg, [agent])
    return cfg, {"kind": "event", "window": (p["at_s"], p["at_s"] + p["mm_off_s"]), "subject": agent,
                 "victim": None, "oracle": cfg["custom_properties"]["oracle"], "meta": p}


SCENARIOS = {
    "exec_twap": make_exec("twap"),
    "exec_ac": make_exec("ac", kappa_T=3.0),
    "exec_pov": make_exec("pov", pov=0.2),
    "exec_passive": make_exec("passive"),
    "spoofing": spoofing,
    "momentum_ignition": momentum_ignition,
    "predatory": predatory,
    "fundamental_jump": fundamental_jump,
    "mm_withdrawal": mm_withdrawal,
    "flash_crash": flash_crash,
}
