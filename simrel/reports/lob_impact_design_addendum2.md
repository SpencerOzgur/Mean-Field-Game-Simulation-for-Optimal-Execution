# LOB-Impact design, addendum 2: three checks requested in review

Written 2 November 2026, after the LOB-Impact results (lob_impact_design.md) and before any of the runs or analyses
below. Already seen: the per-child decomposition of the INTC/CSCO/MSFT build-up rollouts, which shows that the
100-share children almost never move the mid in any simulator (1-8% of children), that Hawkes impact accumulates
between children (+19 to +25 bps per metaorder), and that ABIDES-original's small mechanical jumps (+1.3 to +1.9 bps)
are undone between children (-1.6 to -2.9 bps).

## 1. Validation against the reference tool

The reference tool is run (by the user; it cannot run in this environment) on the INTC rollouts exported as LOBSTER
CSVs (Hawkes-original and ABIDES-original, 6 per arm and side), and compared with our port on the same files
(simrel/experiments/impact_03_crosscheck.py). Pass: identical point counts; delta, gamma_flow and H equal to three
decimals. A discrepancy is traced and fixed, and every LOB-Impact number is recomputed.

## 2. Why ABIDES shows no LOB-Impact response

Three candidate explanations: the market mechanism (agents anchored to an exogenous fundamental revert any
displacement), the intervention size (100-share children against ABIDES's book), and the adaptation to ABIDES's
clock (ABIDES-original's message rate puts children about 3 minutes apart, so reversion has more time). INTC,
ABIDES-original and ABIDES-calibrated, build-up arm, 24 rollouts per side on the main seeds, four arms:

| arm | child | spacing |
|---|---|---|
| beta (done) | median market order | dN messages of the configuration (time clock matched to its own message rate) |
| beta_t | median market order | the real stock's time between children, dN / real message rate (about 20 s) |
| beta_x5 | 5 x median | as beta |
| beta_t_x5 | 5 x median | as beta_t (close to the Goal 1 task: about 450 shares every 30 s) |

Reported per arm: peak impact after 100 children with its standard error, the mechanical jump and the
between-children move. Reading, fixed now: impact detected only with larger children -> intervention size; only
with real-time spacing -> clock adaptation; in neither -> mechanism. Hawkes-original is run on beta_t_x5 as well, as a
comparison with a simulator whose reaction is known.

## 3. Goal 2 intervals that respect the design's dependence

The 116 (market, simulator) rows are not independent: the 29 simulator designs recur in every market, there are only
four markets, and each simulator's recommendation rests on 30 seeds. A three-level bootstrap (2,000 resamples):
draw the four markets with replacement; draw the 29 simulator designs with replacement, the same draw applied in
every market; within each drawn (market, design) draw its 30 policy seeds with replacement and recompute its
predicted costs, its recommendation, its regret and its metaorder measure. The realism and LOB-Impact scores are
held fixed (their own sampling error is not propagated; stated as a limitation). Statistics: the pooled Spearman
correlation of regret with each measure; the mean regret of each selection rule; and the paired differences
(best by metaorder measure - always TWAP), (best by metaorder - best by order book), (best by LOB-Impact peak - best by
order book). With four markets the market-level draw is coarse; the intervals are reported as they come out.

No claim that simulator-based selection outperforms TWAP is made unless the paired difference excludes zero.
