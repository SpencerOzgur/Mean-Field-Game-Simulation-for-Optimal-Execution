# Exploratory: own executions, alone and with a simulator prior

Written 16 November 2026, before the analysis below was run. Motivation: passive data cannot identify the response to
the trader's own orders (Naviglio et al. 2026; Fujiwara, Sato and Kanazawa 2026), the optimal schedule depends on it
(Chen, Horst and Tran 2025), and misspecified simulators cannot be validated from passive data (this study). The
remaining evidence a desk has is its own past executions. Question: how many own executions does it take to choose the
right strategy, and does a misspecified simulator used as a prior reduce that number without misleading?

## Design (development markets only; no new simulation)

- Own executions: for each market, size and strategy, n executions drawn without replacement from reference seeds 0-149,
  independently per strategy (different days, so unpaired), n in {2, 5, 10, 20, 50, 100}; 200 random draws per n.
- Truth for regret: reference seeds 150-299 (cross-fitting), all 12 strategies.
- Rules, all choosing among the 12 strategies:
  - own data: the strategy with the lowest sample mean cost;
  - own data, gated: depart from TWAP only if the best alternative's estimated saving exceeds 2 standard errors;
  - simulator: the cheapest predicted strategy (mean of the four fixed designs, 100 decision seeds), and the corrected
    model alone;
  - hybrid: a Gaussian prior on each strategy's cost difference to TWAP centred on the simulator's prediction, with prior
    variance equal to the simulator's mean squared error on the other development markets (leave one market out),
    updated with the own executions (known-variance normal update, variance from the own sample); choose the lowest
    posterior difference (TWAP = 0).
- Reported: mean regret by family and n; the n at which each rule first beats always-TWAP; regret against TWAP.

## Gate (fixed before running)

Worth a design addendum and fresh confirmation markets only if, in agent-mix markets, the hybrid reaches always-TWAP's
regret with at most half the own executions the better of the two own-data rules needs, while never being worse than
the own-data rules by more than 0.5 bp at any n in any family.

## Result (16 November 2026, appended after the run; results/simrel/goal2/exploratory/own_execution.csv)

Agent-mix markets, mean regret (bp) at n = 2, 10, 50, 100 own executions per strategy (12 strategies): own data 5.20,
4.59, 3.62, 2.79; gated 3.54, 3.08, 2.90, 2.89; hybrid with the four-design ensemble as prior 7.17, 5.85, 4.37, 3.56;
always-TWAP 2.88. Own data alone first reaches TWAP's regret at n = 100 (about 1,200 executions); the hybrid never
does by n = 100 and is worse than own data at every n. In the Hawkes-built markets, where the candidates are correctly
specified, the hybrid reaches TWAP's regret at n = 5 against 20-50 for own data.

Gate not met (the hybrid is worse than own data in agent-mix markets at every n). No design addendum. Reported in the
evaluation paper: own executions are a weak source of evidence at realistic budgets, and a misspecified simulator prior
makes them weaker; a simulator prior helps only where the simulator's mechanisms are right.
