# Exploratory: estimating decision-critical responses directly (no simulator)

Written 16 November 2026, before the analysis below was run. Prompted by the second novelty review: can the market
responses that determine the execution decision be recovered without a fully accurate market simulator? This is a
one-off feasibility test on existing data. It is exploratory: Test B has already been read, so any confirmatory claim
would need newly drawn markets and its own design addendum.

## Estimator

For each market and order size: the market's observable features from one three-hour sample (the eight diagnostic
features of addendum 10, section 4: H, sign-memory gamma, R(100)/R(1), 25-trade block autocorrelation, variance ratios at
30, 120 and 600 s, depth response to one-sided flow) plus the order size. Target: each strategy's true cost difference
against TWAP (bp). One gradient-boosted regression per strategy (scikit-learn defaults, fixed seed), trained on the
training markets. Decision: the strategy with the most negative predicted difference if it is below zero, else TWAP.

Markets with known costs and features: 42 development (diagnostic market sample, seed 63000), 16 Test A (same), 73
viable identifiability-grid markets (first 3-hour sample), 48 Test B (market sample computed with the same code and seed
63000). Families: queue-reactive (R1*, Xqr*, TBq*), agent-mix (R2*, Xam*, IG*, TBa*), latent liquidity (R3s*, TBl*),
Hawkes-built (W1-W6).

## Evaluation

Leave one family out: train on the other families, evaluate on the held-out family. Compared with always-TWAP and with
the no-simulator rule trained on the same training families (lowest mean regret per size). Market-level means over the
two sizes; 95% market-bootstrap intervals (2,000 resamples). Regret from the full truth (300 reference seeds).

## Gate (fixed before running)

The direction is worth a design addendum only if, on at least one held-out family with headroom (agent-mix or latent),
the estimator beats both always-TWAP and the no-simulator rule with intervals excluding zero, while departing from TWAP
in at least 25% of market-size cells, and is not significantly worse than TWAP on any held-out family. Otherwise the
methodological pivot stops and the result is reported as further negative evidence.
