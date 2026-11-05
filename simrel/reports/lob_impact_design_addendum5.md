# Why TWAP wins: diagnostic, conservative decision rules and a validation profile (design addendum 5)

Written 3 November 2026, while the ten sampled markets of addendum 4 were still running and before any of their
results existed. Everything below that is estimated (strategy biases, historical errors, the conservatism parameter k)
was estimated on the first ten markets (the training markets) and frozen in results/simrel/goal2/rules/rules_frozen.json,
whose checksum is recorded with this file. The tests are run on the ten sampled markets only.

## Diagnostic on the training markets (exploratory, already seen)

simrel/experiments/goal2_error_diagnostic.py. For each strategy, the predicted gain over TWAP against the true gain:

- Systematic bias: simulators think selling at once costs the same as TWAP (true: 8.1 bps worse); it is their most
  frequent pick (133 of 576) and never the best. They undervalue signal reversal (bias -1.9 bps), the true best in 10
  of 20 cells.
- 29% of the error variance is shared across the 29 simulators; 71% is simulator-specific.
- Ranking before selection is poor: median rank correlation of predicted and true costs over the 12 strategies 0.22
  (queue-reactive 0.42, agent-mix 0.02).
- Selection amplifies error: the chosen strategy's over-promise is +10.4 bps against +1.6 for a typical strategy.
- Removing the shared bias (an oracle) would give mean regret 1.62 against 1.89 for always TWAP.

## Experiment A: deviate from TWAP only when confident

Choose alternative a only if D_a - k sigma_a > 0 (the largest such a), otherwise TWAP. D_a is the ensemble's predicted
gain over TWAP (mean over all candidate simulators of the market). sigma_a^2 = between-simulator variance of the gain +
seed variance of the ensemble mean + h_a^2, where h_a is the root mean square error of the ensemble's gain for strategy
a over the training markets (it carries the error all simulators share, which disagreement cannot show). Variants:
"bc" also subtracts the training bias b_a; "per simulator" applies the rule to each simulator alone with
sigma_s,a^2 = seed variance + (historical error of single simulators)^2, and averages the regret over simulators.

k is chosen on the training markets from {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}, by mean regret, ties going to the larger k.
Frozen values: ensemble 3, ensemble bc 0.5, per simulator 3; sensitivity at k = 1 is also reported. On the training
markets these rules almost always fall back to TWAP (training regret 1.88-1.89 against 1.89 for TWAP, 5.67 ungated).

Tests on the sampled markets (three-level bootstrap of addendum 2, 2,000 resamples):
- A1. Gated ensemble minus ungated ensemble regret: interval below zero.
- A2. Gated ensemble minus always TWAP: reported. A claim that the rule beats TWAP requires the interval below zero.
- A3. Gated per-simulator minus ungated per-simulator regret: interval below zero.

## Experiment B: a decision-relevant validation profile

Each an error of the candidate against its reference market:
1. immediate-execution cost (predicted cost of selling at once against its true cost);
2. TWAP impact (the metaorder measure);
3. impact relaxation (TWAP impact recovered in the 120 s after the deadline);
4. conditional price response (lag-one autocorrelation of 25-trade mid returns in free runs; negative = reversal);
5. cost-ranking accuracy (rank correlation of predicted and true costs over the 12 strategies; measurable only in
   synthetic markets, reported as an upper bound on what validation could achieve).

Profile = mean within-cell rank of dimensions 1-4 (the ones that real data or a real intervention could provide).

- B1. Correlation of the profile with regret minus that of TWAP impact alone: interval above zero.
- B2. The correlation of each dimension, and of the order-book score, reported.
- B3. The correlation of cost-ranking accuracy (oracle), reported.

## Experiment C: well-specified reference markets (run after the sampled markets)

To separate "the mechanism lies outside the simulators' family" from "simulators fail even when it does not", six
reference markets are built from the candidate family itself: the Hawkes designs h_k1 and h_c_f_m_k1, as fitted to
R1a_qr_resilient, R1b_qr_fragile and R2a_am_anchored. The true mechanism is then a member of the candidate family (the
29 candidates are refitted on the new market's own data; ABIDES is misspecified there). Same viability rule and
pipeline. Reported separately for well-specified and misspecified markets, with intervals: cost-ranking accuracy,
regret of the ungated ensemble, of the best-by-metaorder simulator and of the gated rule, each against always TWAP,
and the correlation of the metaorder measure with regret.

The confirmatory hypotheses C1-C4 of addendum 4 are unchanged.
