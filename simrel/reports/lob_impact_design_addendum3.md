# Goal 2 extension: six more reference markets (design addendum 3)

Written 2 November 2026, before any of the markets below was simulated. Reason: with four markets, the market-level
draw of the three-level bootstrap (addendum 2) sets the width of every Goal 2 interval.

## The six markets

Each changes one mechanism dial of an existing family, chosen to span resilience, amplification, liquidity and
volatility. Parameters not listed keep their family defaults (simrel/reference_worlds).

| market | family | change | intended mechanism |
|---|---|---|---|
| R1c_qr_thin | queue-reactive | join 0.6, q0 1000, deep 0.3, init_qty 400 | thin book, slow replenishment |
| R1d_qr_momentum | queue-reactive | theta 2.8, market 0.16 | market orders chase queue imbalance: amplification |
| R1e_qr_deep | queue-reactive | join 2.5, q0 4000, deep 0.9, cancel 0.4, init_qty 1500 | deep book, fast replenishment |
| R2c_am_anchor | agent-mix | fund 0.30, fund_power 2.0, trend 0.02, withdraw 1.0 | strong fundamental anchoring: resilience |
| R2d_am_toxic | agent-mix | withdraw 5.0, v_ref 1000, limit 1.0 | liquidity providers flee one-sided flow: convex impact |
| R2e_am_calm | agent-mix | fund_vol 0.10, trend 0.03, trend_boost 3.0, regime_switch 1/800 | calm, low volatility |

## Procedure (unchanged from the first four markets)

Reference stage as in simrel/experiments/run_reference.py (6 fitting episodes, 40 held-out twins, every strategy on
300 seeds, ABIDES calibrated on its 18-point grid); the ABIDES policy runs of that older study are skipped, since
Goal 2 runs its own. Then goal2_run.py (29 candidates: fit, LOB-Bench realism, 30 policy seeds) and goal2_impact.py
(LOB-Impact, 32 rollouts per arm and side per candidate, 64 for the market itself).

## Viability rule, fixed now

A market whose unperturbed ten-minute mid move exceeds 300 bps in more than a quarter of its 40 held-out twins, or
whose twins fail in more than half the seeds, is excluded and reported as such (the stability gate used for every
simulator in this study). Excluded markets are not replaced or re-tuned.

## Analysis

The pre-specified Goal 2 analyses (B1-B4 of lob_impact_design.md, and the three-level bootstrap of addendum 2) are run
on all viable markets together, and reported separately for the original four and the new markets.
