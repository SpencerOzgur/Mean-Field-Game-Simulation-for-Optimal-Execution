# Exploratory checks prompted by the literature review (4 November 2026)

None of these were pre-registered; all were run after every Goal 2 result had been seen. They test whether the
conclusions survive five objections a reviewer could raise.

## 1. Execution schedules against signal strategies

Two of the twelve strategies (signal momentum, signal reversal) trade on predicted price moves, which execution
simulators are not built for. Repeating the analysis with schedules only (simrel/experiments/goal2_exploratory.py split,
results/simrel/goal2/exploratory/split_schedules_signals.csv), on the 20 misspecified markets:

| | all strategies | schedules only |
|---|---|---|
| regret, always TWAP | 2.00 | 1.06 |
| regret, ensemble's cheapest strategy | 5.75 | 4.90 |
| choose by metaorder accuracy minus TWAP | +0.66 [+0.04, +1.78] | +0.89 [+0.16, +1.86] |
| metaorder accuracy, correlation with good advice | +0.20 [0.04, 0.25] | +0.17 [0.04, 0.27] |
| LOB-Bench order book, correlation | -0.10 [-0.21, 0.01] | -0.12 [-0.25, -0.00] |

The conclusions do not depend on the signal strategies.

## 2. Bayesian shrinkage instead of choosing the cheapest prediction (Smith and Winkler 2006)

Each strategy's predicted gain over TWAP is shrunk towards a prior fitted on the first ten markets (goal2_exploratory.py
shrink, bayesian_shrinkage.csv). The simulators' historical errors are so large that the weight on their prediction is
essentially zero, so the rule reduces to "use the strategy with the best prior", signal reversal:

| markets | always TWAP | ensemble, cheapest | Bayesian shrinkage | prior only (no simulator) |
|---|---|---|---|---|
| sampled (10) | 2.10 | 5.83 | 0.73 | 0.73 |
| well-specified (6) | 4.11 | 2.15 | 5.02 | 5.02 |

The simulators add nothing beyond knowing the family of market. The prior works where the market shares the training
markets' mechanism (mean reversion) and fails in the Hawkes-built markets. Caution for the paper: our two synthetic
families share mechanisms, so a prior learned on some of them transfers to others.

## 3. The real cost of selling at once is a bound

Hidden orders execute 12-14% of volume, 20-24% on the bid side (hidden_liquidity_share.csv). Enlarging every visible
level by the bid-side hidden share (a generous lower bound) lowers the real cost by about 7-17% (slice_16_immediate_bounds.py,
immediate_cost_bounds.csv). The Goal 1 mechanism is unchanged: ABIDES-original stays above the band (selling at once
too expensive), Hawkes-refill below it (too cheap).

## 4. Impact measured on our own data (synthetic metaorders, Maitrier, Loeper and Bouchaud, arXiv:2503.18199)

Every aggressive order assigned at random to one of 20 synthetic traders; a metaorder is a run of same-sign orders of
one trader (at least two); impact from the mid before the first child to the mid before the next order after the last
one, in units of the day's range (slice_17_synthetic_metaorders.py; 5 assignments pooled):

| stock | metaorders | exponent | amplitude Y at Q/V = 1e-3 |
|---|---|---|---|
| INTC | 99,967 per assignment | 0.35 | 0.86 |
| CSCO | 71,210 | 0.33 | 0.97 |
| MSFT | 247,797 | 0.43 | 0.92 |

The measured impact of the Goal 1 TWAP order (10% of ten-minute volume) is 5.6-6.5 bps, above the literature band
used in slice_14 (0.9-4.4 bps). Simulated TWAP impact divided by the measured impact (twap_vs_synthetic_metaorders.csv):

| version | INTC 10% / 30% | CSCO 10% / 30% | MSFT 10% / 30% |
|---|---|---|---|
| Hawkes-original | 1.43 / 1.95 | 1.34 / 2.26 | 1.86 / 2.81 |
| Hawkes-refill | 0.40 / 0.45 | 0.36 / 0.49 | 0.55 / 0.61 |
| ABIDES-original | 0.27 / 0.47 | 0.25 / 0.40 | 0.46 / 0.59 |
| ABIDES-calibrated | 0.07 / 0.27 | 0.09 / 0.27 | 0.73 / 1.31 |
| Hawkes-queue | 9.9 / 12.1 | 5.8 / 9.2 | 3.6 / 4.1 |

Which simulator "gets impact right" depends on the real-data benchmark: against the literature band Hawkes-refill was
inside and Hawkes-original 2-3 times too high; against measured synthetic metaorders Hawkes-original is the closest and
Hawkes-refill about half. The 30% sizes lie beyond the fitted range (extrapolated). Synthetic metaorders may carry
correlated flow of other traders, and our eight days give a higher amplitude than the paper's three years of Paris
data. The paper must report both benchmarks and state the Goal 1 impact claim conditionally.

## 5. A modern queue-reactive candidate

A 30th candidate in the spirit of Huang, Lehalle and Rosenbaum (2015) and Noble, Rosenbaum and Souilmi (2026):
intensities depend on the spread and the ten-level depth imbalance (15 states), no memory of past events
(simrel/adapters/qr_candidate.py; goal2_qr_candidate.py, results/simrel/goal2/exploratory/qr_candidate*.csv). Fitted to
every market's observable log like the others; the 29 pre-registered candidates and their results are untouched.

| mean regret (bps) | queue-reactive candidate | always TWAP | median Hawkes candidate | its rank among 30 |
|---|---|---|---|---|
| queue-reactive markets (10) | 0.54 | 0.41 | 0.58 | 15 |
| agent-mix markets (10) | 7.23 | 3.58 | 9.68 | 12 |
| Hawkes-built markets (6) | 8.64 | 4.11 | 1.70 | 24 |

It relaxes after a metaorder (LOB-Impact end over peak 0.15-0.72, where every Hawkes candidate stays at or above 1),
the property the recent literature emphasises; in the queue-reactive markets the true market does not relax at
LOB-Impact's scale (1.10), so it over-relaxes there. It does not improve decisions enough to beat TWAP in any group, and
its picks show the same pattern (selling at once most often, signal reversal never). Caveat: a simplified
implementation (the synthetic markets' logs give ten-level depths, not best-queue sizes), not the authors' model.
