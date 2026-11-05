# Goal 2 design: which kinds of realism predict good execution decisions?

Written 31 October 2026, before any Goal 2 simulator was fitted, run or scored.

## Question

Real data cannot say which simulator's advice is right. Here the "market" is one of four synthetic reference markets
whose rules we wrote (from the second study: two queue-reactive markets, R1a resilient and R1b fragile, and two
agent-mix markets, R2a anchored and R2b trending). In them the true expected cost of every strategy is known
(`results/simrel/raw/ref_*/rows_reference.pkl`: 12 strategies, two order sizes, 150+ seeds each).

We build a family of candidate simulators of each reference market, score each one's realism against held-out
reference data, ask each one for its recommended strategy, and measure that recommendation's **regret**: its true
cost in the reference market minus the true cost of the best strategy.

## Candidate simulators (per reference market, all fitted only on the six observable 30-minute fit episodes)

- **Hawkes family** (compiled simulator), fitted with the same code as for INTC:
  - structure: every combination of spread closing, gap filling and multi-timescale kernels (8 versions);
  - reaction to trades: impact factor 0.5, 1, 2 on each structure (24 versions);
  - stability cap: branching ratio 0.8 and 0.95 instead of 0.9, for the plain and the full structure (4 versions).
  The queue and size options need best-queue sizes that the reference markets' event logs do not record, so they are
  not part of the family.
- **ABIDES**: the configuration the second study calibrated to each market.

About 29 simulators per market, 116 in total.

## Task

As in the second study: sell over ten minutes in 20 decisions; order sizes 5% and 15% of the **reference market's**
mean volume (the trader knows the order size in shares, so the size does not depend on the simulator); all 12
strategies; 30 seeds per simulator, market and size. A simulator recommends the strategy with its lowest mean cost.

## Realism measures (each simulator against 12 held-out ten-minute reference windows)

1. Order book: LOB-Bench's 21 default statistics, mean L1.
2. Price: five return and volatility statistics, mean Wasserstein.
3. Impact response: LOB-Bench's impact score.
4. Trade response: mean price change ten seconds after the market's own aggressive orders, absolute error.
5. **Metaorder impact**: absolute error of the simulator's TWAP peak impact against the reference market's own TWAP
   peak impact at the same order size. This uses an intervention on the reference market, which real data cannot
   provide; it is the measure the real-data study could only approximate with the square-root law.

## Pre-specified analysis

- **A1.** Spearman rank correlation between regret and each realism measure, within each market and size, and pooled
  (ranks within market and size). Which measure predicts regret best?
- **A2.** Selection: in each market and size, pick the simulator with the best score on each measure and record the
  regret of its recommendation. Compare with the median regret over all simulators and with always using TWAP.
- **A3.** Ensembles: average the predicted costs over (a) all simulators, (b) the five best by order book, (c) the five
  best by metaorder impact; recommend the cheapest; record regret.
- **A4.** Robustness: repeat A1 and A2 excluding ABIDES, and excluding the two queue-reactive markets.

Expected direction (stated in advance, from the INTC study): metaorder impact predicts regret better than order-book
realism. The analysis reports whichever way it comes out.
