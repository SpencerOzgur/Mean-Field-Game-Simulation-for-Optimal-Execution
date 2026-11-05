# Goal 2 at 100 decision seeds (design addendum 7)

Written 5 November 2026, after the oracle of addendum 6 showed that 30 decision seeds per candidate are too few
(sampling noise alone produces regret of the order of TWAP's), and before any of the additional seeds below exist.

## Change

Every candidate (the 29 pre-registered designs) in every one of the 26 markets is given 70 more policy seeds
(62030-62099), so that its predicted costs, recommendation, regret and metaorder measure rest on 100 seeds, as the
oracle's do. Nothing else changes: same task, strategies, tie rule, realism and LOB-Impact scores.

## Analyses, at 100 seeds, reported beside the 30-seed results

1. Oracle comparison: candidate regret against the oracle's (100 seeds) and against always TWAP, by family; the
   misspecification share (candidate regret beyond the oracle's).
2. The confirmatory hypotheses C1-C4 of addendum 4 on the ten drawn markets, recomputed. These are now secondary: the
   primary test stays the 30-seed result already reported; the 100-seed result shows whether its failure was a matter
   of power.
3. Correlations of every realism and counterfactual measure with regret, and selection regret, on all 26 markets and by
   family (three-level bootstrap, seeds resampled out of 100).
4. Experiment A of addendum 5: the gated rules, with h, b and k re-estimated on the first ten markets at 100 seeds by the
   same procedure, then frozen and tested on the drawn markets.
5. Ensemble and single-simulator regret against always TWAP in the well-specified and misspecified markets.

Expectation stated in advance: with less noise, candidate regret falls in the Hawkes-built markets (where the oracle
falls from 3.03 to 1.22) and little in the agent-mix markets (where misspecification dominates); whether any selection
rule then beats TWAP is reported either way.
