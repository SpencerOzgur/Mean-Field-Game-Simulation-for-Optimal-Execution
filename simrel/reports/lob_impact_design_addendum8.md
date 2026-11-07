# Noise against misspecification, a decision-aware metric, matched mechanisms and a held-out family (design addendum 8)

Written 6 November 2026, 12:00, after the interim 86-seed results of addendum 7 and before the 100-seed results are
complete (the last Kaggle part was still running) and before any of the new runs below exist. Prompted by review.

Positioning adopted: "When does market simulator realism translate into reliable execution decisions?", around three
questions: realism against advice (Goal 1), why advice fails (finite-sample decision uncertainty against missing
mechanisms), and whether failure can be detected beforehand. We stop tuning selection rules on the existing 26 markets
(no further work on Bayesian shrinkage or on combinations of existing realism scores).

## Step 1. Final 100-seed results, with split seeds

The 30-seed pre-registered results stay primary; the 100-seed results are secondary (addendum 7), reported for every
market group: oracle, always TWAP, ensemble and single-simulator regret under one protocol.

Split seeds. Wherever a validation measure is compared with regret, the measure is estimated from decision seeds 50-99
of the candidate and the recommendation (hence the regret) from seeds 0-49, so that no Monte Carlo noise is shared
between them. The full-seed analyses of addendum 7 are reported beside them.

Truth noise. Regret uses the reference market's true costs (300 seeds 0-299, independent of every simulator's seeds
62000+). As a sensitivity check, regret is also cross-fitted: the true best strategy is identified on reference seeds
0-149 and costs are evaluated on seeds 150-299.

## Step 2. Four decision makers per market

| | decision maker | seeds | error it contains |
|---|---|---|---|
| A | the market's own generator (oracle) | 100 | as little sampling error as feasible, no structural error |
| B | the oracle | 30 | sampling error |
| C | each candidate simulator | 100 | structural error, little sampling error |
| D | each candidate simulator | 30 | both |

Sampling contribution regret(B) - regret(A); structural contribution regret(C) - regret(A); combined regret(D) -
regret(A); reported by market group with 95% intervals (bootstrap over markets and seeds, and over designs for C and
D). These are diagnostic differences, not an additive decomposition; the interaction regret(D) - regret(B) -
regret(C) + regret(A) is reported separately. Success criterion: in the agent-mix markets the structural contribution
is positive with its 95% interval above zero.

## Step 3. Matched mechanisms with fixed candidate designs

Markets: the 22 ablation markets of addendum 6 plus a sixth agent-mix knob, flow_halflife (impact decay of the
liquidity providers' withdrawal; levels 5, 10, 20*, 40, 80 s; 4 new markets). For every one of these 26 matched
markets, four fixed candidate designs are fitted to its observable data: plain Hawkes (h_k1), refill Hawkes
(h_c_f_m_k1), ABIDES (grid-calibrated) and the queue-reactive candidate; 100 decision seeds each.

Measures, on pairwise cost differences rather than absolute costs: for every pair of strategies (i, j), the error
(C_hat_i - C_hat_j) - (C_i - C_j). Reported per knob and design: (a) whether the predicted difference against TWAP of
each strategy moves with the knob in the same direction as the true difference (rank correlation across the five
levels); (b) the decision-boundary error, whether the design's recommended strategy changes between levels where the
truth's does (agreement of recommended and true optimal strategy across levels); (c) the mean absolute pairwise
difference error. A mechanism "creates decision-boundary errors" for a design if (b) is below 50% while (c) exceeds the
error of the same design at the base market by more than 1 bp.

## Step 4. A decision-aware validation measure (oracle diagnostic)

E_decision = mean over all 66 pairs of strategies of |(C_hat_i - C_hat_j) - (C_i - C_j)|, the error in the cost
differences that determine the decision. Measured with split seeds (C_hat from seeds 50-99; regret from seeds 0-49).
Compared, on all 26 Goal 2 markets and by group, with LOB-Bench, LOB-Impact and the metaorder measure: rank correlation
with regret and selection regret, three-level bootstrap. It needs the true costs of every strategy, so it is an oracle
diagnostic; its observable two-pair proxy (the errors in C_immediate - C_TWAP alone, which real data can approach by
walking the book and measuring TWAP impact) is reported beside it. Hypothesis: E_decision predicts regret better than
the metaorder measure (paired difference of correlations, interval above zero); the proxy is reported either way.

## Step 5. A held-out market family

A new reference-market family, "R3 latent liquidity", not used to design any rule (simrel/reference_worlds/latent.py):
liquidity is revealed around a latent reference price that follows the traded price with a relaxation time tau
(transient impact as in propagator and latent-order-book models), market-order signs are persistent (runs of
Pareto-distributed length), and cancellations are proportional to depth. Six markets are drawn from fixed ranges
(seed 20261106; first six viable draws; viability rule of addendum 3):

| parameter | range |
|---|---|
| tau (s, log) | 20 - 600 |
| limit (orders/s per side, log) | 0.8 - 2.5 |
| market (orders/s per side, log) | 0.06 - 0.20 |
| run_alpha (Pareto exponent of sign runs, uniform) | 1.3 - 2.5 |
| level_p (geometric placement, uniform) | 0.25 - 0.6 |
| fund_vol (ticks per sqrt(s) of the reference, log) | 0.05 - 0.3 |

Full pipeline (truth with 300 seeds, ABIDES calibration, 29 candidates fitted, LOB-Bench and LOB-Impact, 100 decision
seeds). Without any retuning, reported: always TWAP; the oracle; ensemble and single-simulator regret; selection by
each measure; the gated rule of addendum 7 with its values frozen on the first ten markets at 100 seeds; and the Step 2
decomposition. A task variant on the same six markets (30-minute horizon in 60 decisions; sizes 2% and 30%) with the
four fixed designs of Step 3 and the oracle tests a different horizon and order-size regime.
