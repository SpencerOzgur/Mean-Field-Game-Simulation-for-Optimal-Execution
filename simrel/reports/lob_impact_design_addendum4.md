# Goal 2 confirmation: ten randomly drawn reference markets (design addendum 4)

Written 3 November 2026, after the ten-market results of addendum 3 and before any of the markets below was drawn or
simulated. Reason: in the six hand-designed markets of addendum 3 the association between the metaorder measure and
decision regret was positive but not clearly so (+0.13, 95% interval [-0.06, 0.29]). A fresh set of markets, drawn
at random rather than designed, tests whether the association is real.

## Drawing the markets

Parameters are drawn independently, log-uniformly over the ranges below (uniform for exponents and powers), from
numpy.random.default_rng(20261103), first all queue-reactive draws and then all agent-mix draws, in the order listed.
Parameters not listed keep their family defaults. Draws are taken in sequence; each is checked with the viability rule
of addendum 3 (40 held-out twins; excluded if more than a quarter move over 300 bps in ten minutes or more than half
fail); the first five viable draws of each family are kept, and every draw, kept or excluded, is reported.

| family | parameter | range |
|---|---|---|
| queue-reactive | join | 0.5 - 2.5 |
| | q0 | 800 - 4000 |
| | deep | 0.25 - 1.0 |
| | cancel | 0.3 - 0.9 |
| | market | 0.08 - 0.18 |
| | theta (uniform) | 0.8 - 3.0 |
| | init_qty | 400 - 1500 |
| agent-mix | fund | 0.03 - 0.30 |
| | fund_power (uniform) | 1.2 - 2.2 |
| | trend | 0.02 - 0.12 |
| | trend_boost (uniform) | 2 - 7 |
| | regime_switch | 1/900 - 1/200 |
| | withdraw | 1.0 - 5.0 |
| | limit | 0.9 - 1.8 |
| | fund_vol | 0.10 - 0.30 |

Names: R1s01-R1s05 and R2s01-R2s05 (s for sampled), numbered in order of acceptance.

## Procedure

As in addendum 3: reference stage, 29 candidates (fit, LOB-Bench realism, 30 policy seeds), LOB-Impact.

## Confirmatory hypotheses (on the ten new markets only)

- C1 (primary). The rank correlation between regret and the metaorder impact error is positive: the 95% interval of
  the three-level bootstrap (addendum 2; markets, designs, seeds) lies above zero.
- C2. Choosing the simulator with the best metaorder impact error gives lower mean regret than choosing by LOB-Bench's
  order-book score: the 95% interval of the paired difference lies below zero.
- C3. No realism measure (LOB-Bench order book, price, impact score; trade response; LOB-Impact flow and response
  errors) has a 95% interval lying above zero.
- C4 (descriptive). Choosing by metaorder impact error against always TWAP: reported with its interval.

Exact ties in predicted costs go to the first strategy by name (as fixed on 3 November in every code path).
All twenty markets are also analysed together, as a secondary result.
