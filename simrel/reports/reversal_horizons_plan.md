# Horizon sweep for the reversal check: definitions (exploratory)

14 November 2026. **Exploratory follow-up.** The committed check (simrel/reports/reversal_check_plan.md, commit
cff8418; report simrel/reports/reversal_check.md) used one horizon, 30 s, and its result is known: strong
order-flow continuation in the simulated families (S3 > 0, S4 < 0) and none in INTC, CSCO and MSFT. This file was
written with that result in hand. It fixes the definitions of the sweep before any number at another horizon is
computed. Nothing in the committed plan, report or results files is changed; all outputs here are new files.

## 1. Data

Exactly the data of the committed check.

- Real: the same 24 files (INTC, CSCO, MSFT; eight days), the same segments, 09:45 to 15:45.
- Simulated: the same 24 markets (results/simrel/goal2/reversal_check/markets.json) and the same passive episodes:
  seeds 70000 to 70019, 600 s warm-up, 3,600 s segment. The stored summaries hold 30 s bins only, so the episodes are
  regenerated from the same seeds; the engine is deterministic and the reproduction check (section 4) confirms that
  they are the same episodes.
- No Test B market is read or simulated. No truth run is made: S5 is not part of the sweep, because the engine's
  decision interval belongs to the study's task.

The quote grid is 5 s on both sides, so at h = 5 a bin is one grid step.

## 2. Horizons and statistics

Horizons: h = 5, 10, 30, 60 and 120 s. For each h the statistics are those of the committed plan with only the bin
length changed.

- **S2(h).** Lag-1 autocorrelation of non-overlapping h-second mid changes (grid points h seconds apart from the
  segment start), demeaned within the segment, pooled over segments by sums.
- **S3(h).** For each bin j >= 1: x = P6's signal for a sell parent computed from bin j-1, signal = 0.5 x (imb + ret),
  imb = signed volume / volume of the bin's trades (0 if no volume), ret = sign of the bin's mid change; y = mid
  change over bin j in bp of the mid at its start. Pooled least squares with an intercept; slope and t-statistic
  with standard errors clustered by segment. Positive means continuation.
- **S4(h).** The impact-free timing backtest with the 600 s window kept and the decision interval set to h, so
  600 / h decisions per window (120, 60, 20, 10, 5). At each decision the strategy sees the flows and the mid change
  of the previous h seconds (zeros at the first decision). Child sizes come from the study's objects
  (candidate_set(0.15), P0_twap and P6_signal_reversal) through core.child_quantity. Children are valued at the mid
  at their decision time. S4(h) = P6 minus TWAP timing cost in bp of the arrival mid, sell parent; negative means P6
  is cheaper. **Except at h = 30 this is P6 re-parameterised to a different slice length, not the study's P6.** With
  more decisions the same clip (0.25 to 3 times the even pace) lets the schedule move more of the order in total, so
  S4 at different h are not the same strategy at different speeds; the profile is read with that in mind.
- **Parent size.** The Q of the committed check for every horizon: 15% of the mean traded volume per 600 s window,
  rounded down to 100 shares.
- **Lot.** S4 is reported with lot = 100 (as in the committed check) and with lot = 1, at every horizon. With lot 100
  and many decisions the even pace falls below one lot (for example 9,100 shares over 120 decisions) and both
  schedules send nothing until late in the window, so lot 100 measures rounding at short horizons. **The reading in
  the report is based on lot = 1.** Lot = 100 at h = 30 is the reproduction check.

Component split, at every horizon:

- **S3, flow alone and return alone.** The same regression with x = imb, and with x = ret. Slopes are per unit of
  the component itself (the combined signal gives each a weight of one half).
- **S4, flow alone and return alone.** The same backtest with P6 shown only one component: for flow alone the mid
  change of the previous bin is set to zero, so P6's signal is 0.5 x imb; for return alone the volumes are set to
  zero, so its signal is 0.5 x ret. Each component keeps the weight it has in the combined signal. The policy object
  is not changed; only the observation is.

Zero-signal bins, at every horizon: the share of bins in which the combined signal is exactly zero, and the shares
with no traded volume and with no mid change.

Units: every S3 and S4 value is given in bp and in ticks (bp divided by the tick size in bp of the stock or of the
simulated market: about 1.85, 2.04 and 0.72 bp for INTC, CSCO and MSFT, 10 bp simulated).

## 3. Uncertainty and multiple horizons

Real: bootstrap over the 8 days of a stock. Simulated: bootstrap over the markets of a family with seeds resampled
within markets. 2,000 resamples, percentile 95% intervals, as in the committed check. Bootstrap slopes are computed
from per-segment sums.

Five horizons are reported with 95% intervals. No horizon is singled out as significant on its own. With five
horizons, three regressors and two lots, some intervals will exclude zero by chance; the report describes the profile
across horizons.

## 4. Reproduction check

At h = 30 with lot = 100 and the combined signal, the point estimates must equal the committed check: S2, the S3
slope and t-statistic, S4, the number of windows and the share of windows in which P6 is cheaper, for each stock
(real_table.csv, rows "8 days") and each simulated family (sim_table.csv, rows "pooled"). They are computed with the
same functions as the committed check. If any differs, the work stops and the difference is reported before anything
else. The bootstrap intervals are drawn afresh (the committed ones came from a random stream shared with the S5
bootstraps); the committed intervals are shown next to the new ones.

## 5. What would change the reading (fixed before computing)

- If simulated continuation (S3 > 0, S4 < 0) appears at every horizon while the real data show none, the
  benchmark-property conclusion is strengthened.
- If real data show predictability at short horizons (5 to 10 s) that is absent at 30 s and above, it is reported
  as such. It is a real short-horizon effect that P6's 30 s slices do not use.
- If real data show continuation at some horizon comparable in size to the simulated value, the conclusion is
  weakened at that horizon. This is said plainly.
- How fast the effect decays with horizon is compared between real and simulated data.

"Comparable in size" is judged in ticks, family against stock, with the intervals. The reading uses S3 and S4 with
lot = 1 and the combined signal; the components and lot = 100 are description.

## 6. Outputs

- results/simrel/goal2/reversal_check/horizons.csv: one row per side, family or stock, and horizon, with S2, S3
  (bp, ticks, t) and S4 (bp, ticks) for the combined signal and each component, both lots, intervals, zero-signal
  shares, and the numbers of segments and windows.
- results/simrel/goal2/reversal_check/fig_reversal_horizons.png: S3 and S4 against horizon (log axis, ticks), one
  line per stock and per simulated family, intervals as bands, a zero line.
- simrel/reports/reversal_horizons.md: the table, the figure, the reproduction check, which reading applies, limits.
- Code: a bin-length and a lot parameter in simrel/metrics/reversal.py with defaults that leave existing calls
  unchanged; simrel/experiments/reversal_horizons.py; tests in simrel/tests/test_reversal.py.
