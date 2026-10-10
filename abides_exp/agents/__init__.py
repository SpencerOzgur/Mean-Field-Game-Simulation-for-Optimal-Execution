from .execution import (
    AlmgrenChrissAgent,
    ExecutionAgent,
    PassiveThenCrossAgent,
    POVAgent,
    TWAPAgent,
)

# Strategy name -> agent class. Register new strategies here (e.g. "mfg").
STRATEGIES = {
    "twap": TWAPAgent,
    "ac": AlmgrenChrissAgent,
    "pov": POVAgent,
    "passive": PassiveThenCrossAgent,
}
