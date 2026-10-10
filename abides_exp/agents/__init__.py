from .execution import ExecutionAgent, TWAPAgent

# Strategy name -> agent class. Register new strategies here (e.g. "mfg").
STRATEGIES = {
    "twap": TWAPAgent,
}
