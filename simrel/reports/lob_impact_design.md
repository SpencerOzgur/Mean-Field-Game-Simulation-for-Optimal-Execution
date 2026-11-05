# Adding LOB-Impact to the realism profile (Goal 1 and Goal 2)

Written 1 November 2026, before any LOB-Impact score of any simulator was computed. Only calibration numbers and a
probe of eight free runs per simulator (counting mid-price changes, to check the pipeline) had been looked at.

## Why

LOB-Bench (ICML 2025) compares the *distribution* of generated order flow with real data: an observational, or
realism, measure. LOB-Impact (Nigmatulin et al., ICAIF 2026, arXiv:2610.27100, code
github.com/george-nigmatulin/lob_impact @0aaf6bf) injects a standardised metaorder and measures the *response*: a
counterfactual measure, and a published, simulator-agnostic one. Our current Goal 2 result is that only our own
metaorder measure (TWAP peak impact against the reference market's, at the task's order size) predicts decision
regret. The question here is whether a standardised, published counterfactual benchmark does the same, and how it
compares with LOB-Bench.

## Two flavours of metric

| flavour | metric | source |
|---|---|---|
| realism (observational) | LOB-Bench order book (21 statistics, L1), price (5 statistics, Wasserstein), impact score, trade response | as before |
| realism (observational) | LOB-Impact phase 1: trade-sign memory gamma_flow, Hurst exponent H, response ratio R(100)/R(1), free drift | new |
| counterfactual | LOB-Impact phase 2: build-up exponent delta, amplitude Y, peak impact after 100 children; phase 3: end-over-peak e/p | new |
| counterfactual | our metaorder measure (Goal 2 only; the task's own TWAP) | as before |

## Protocol (as in the paper, section 2)

Per stock and day: child = median market-order size; spacing dN = 9 / phi messages (participation 10%), phi = share
of market orders in the visible message stream. Arms on the same seeds: free run (101 dN messages), build-up (100
children), blind build-up (Hawkes only, see deviations), relaxation (10 children, 100 cooling windows). Buy and sell
are pooled by sign. Estimators: the reference code's, ported constant for constant into simrel/metrics/lob_impact.py
(signed binned estimator with 18 bins and a 400-resample day bootstrap; direct least-squares identifiability on
delta in [0.05, 1.5]; Y floor 0.05; detection at 3 standard errors; collapse filters 2000 bps and 5000 ticks).

## Deviations from the paper, fixed now

1. Rollouts: 64 per simulator, arm and side in Goal 1 (paper: 1,024 over 20 days), over all eight sample days; 32
   per candidate and 64 per reference market in Goal 2. Intervals will be wider than the paper's.
2. The reference code could not be executed in this environment; the estimators are ported. A subset of rollouts is
   exported as LOBSTER CSVs in the tool's layout so the tool can be run on them as a check.
3. ABIDES runs on a time clock: one child every dN / r seconds, r being the configuration's own visible message
   rate, so the spacing matches the message clock on average. ABIDES has no blind arm.
4. The Hawkes blind arm removes only the self-excitation by the children; the children still change the book, which
   queue- and gap-dependent intensities see. It is reported as "partly blind".
5. No Historic (real replay) metaorder arm: it is the paper's negative control, not a target.
6. The Hawkes simulators run with the compiled implementation (validated against the reference implementation).
7. Goal 2 markets have no calendar: pseudo-days are blocks of seeds (8), and the daily volume V and volatility used
   to normalise impact are the reference market's own (volume per 6.5 h; daily volatility from its 5-second returns).
   They are the same for the reference and every candidate, so errors against the reference do not depend on them.

## Goal 1 (real data: INTC, CSCO, MSFT; five versions)

No ground truth; descriptive, all reported:

- G1.1 The claim ladder C1–C5 of every version on every stock, with the paper's criteria (C1: gamma, H and drift
  within 0.10, 0.02 and 5 ticks of the stock's real stream; C2: law identified; C3: v/b >= 3; C4: |delta - 0.5| <=
  0.10 and Y in [0.5, 1]; C5: e/p at or below 0.87 (the paper's OW kernel value) with a non-positive slope and free
  drift under 20 ticks).
- G1.2 Do LOB-Impact's counterfactual numbers agree with our two real-data checks (TWAP impact against the
  square-root band; cost of selling at once against walking the real book)? Rank correlation over the five versions
  of peak impact and Y with our TWAP peak impact.
- G1.3 Do the two versions that give opposite advice (Hawkes-original, ABIDES-original) differ more in LOB-Impact's
  counterfactual numbers than in LOB-Bench? Reported as the two versions' values side by side.

## Goal 2 (synthetic reference markets with known best strategy)

The protocol is run on each reference market itself (it can be intervened on) and on each of its 29 candidates.
Measures, each an error against the reference market (lower is better); an undefined value (no identified law, no
gamma) where the reference has one counts as 10 tolerances:

- li_flow_error: mean of |gamma - gamma_ref| / 0.10, |H - H_ref| / 0.02, |drift - drift_ref| / 5 ticks (realism).
- li_response_error: |R(100)/R(1) - reference| (realism).
- li_peak_error: |peak - peak_ref| in bps after 100 children (counterfactual).
- li_law_error: mean of |delta - delta_ref| / 0.10, |ln(Y / Y_ref)| / ln 2, |e/p - e/p_ref| / 0.10 (counterfactual,
  against the market's own impact law).
- li_law_error_literature: the same against the empirical values delta = 0.5, Y in [0.5, 1] (distance to the band),
  e/p = 2/3: what a practitioner without an intervention on the real market would use.

Analyses, as in the Goal 2 design:

- B1. Pooled Spearman correlation of regret with each measure (old and new), ranks within market and size, with a
  95% bootstrap interval (resampling simulators within market and size, 2,000 resamples).
- B2. Regret of the simulator chosen by each measure; B3. regret of the five-best ensemble by each measure.
- B4. Robustness: excluding ABIDES; excluding the queue-reactive markets.

Expectations stated in advance: the counterfactual LOB-Impact measures (li_peak_error, li_law_error) predict regret
better than LOB-Bench's; li_law_error_literature does worse than li_law_error, because the reference markets need
not follow the square-root law; none beats our task-specific metaorder measure, which uses the task's own order size
and schedule. The analysis reports whichever way it comes out.
