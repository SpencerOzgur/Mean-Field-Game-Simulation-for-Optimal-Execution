# Is the signal-strategy advantage of the reference markets present in real data? Analysis plan

14 November 2026. Descriptive check. This plan is committed before any statistic is computed on real data; when it
was written the real data files were not on the machine. Nothing here changes a rule, an addendum or a market.

## 1. Question

In every reference family run so far, P6_signal_reversal is usually the true best strategy. The lambda pilot showed
that this does not depend on impact fading. If the advantage comes from a property of the simulation engine that
real markets lack, part of the benchmark rewards exploiting the benchmark. P6's gain can come from two channels:

1. Background predictability: the next 30 s mid return is predictable from the public signal of the previous 30 s,
   whoever caused it. This can be measured without own impact, on real and simulated data alike.
2. Own impact: how the market responds to our own children (resilience, and the feedback of our own trades into
   the signal). This cannot be measured from passive real data.

## 2. What P6_signal_reversal does (read from the code before writing this plan)

simrel/policies/library.py, SignalAware with gamma = -1, lo = 0.25, hi = 3.0. At each decision it requests
clip(1 - signal, 0.25, 3) times the even pace (remaining / decisions left), where

    signal = 0.5 x (imb + ret)
    imb    = -sgn x signed volume of the previous slice / total volume of the previous slice   (0 if no volume)
    ret    = -sgn x sign(mid now - mid one slice ago)
    sgn    = -1 for a sell parent, +1 for a buy parent

For a sell parent this gives imb = signed volume / volume (buyer-initiated positive) and ret = sign of the mid
change. So after a slice in which the price fell and sellers dominated, signal is negative and P6_signal_reversal
sells faster (up to 1.9 times the even pace in a direct check: signed volume -80% of volume, mid down); after a slice
in which the price rose it sells slower (0.25 times). For a buy parent it is the mirror image.

**This is the opposite of the strategy's name and of its docstring** ("signal > 0 means the last slice moved against
the parent order ... gamma < 0 waits"). As coded, signal > 0 means the last slice moved in favour of the parent
order, and gamma = -1 trades faster after an adverse move. P6_signal_reversal therefore trades with the last move.
Without own impact it gains when 30 s moves continue and loses when they revert. P6_signal_momentum (gamma = +1) is
the one that waits after an adverse move.

Consequences for this check, fixed now:

- All statistics use the strategy as coded. No policy is changed.
- S3 and S4 are defined by the code's signal, so their signs carry the meaning: a positive S3 slope (continuation)
  and a negative S4 favour P6_signal_reversal as coded.
- The engine adds the trader's own child to the previous slice's volume and signed volume. A sell child makes the
  next signal more negative, so P6_signal_reversal as coded also speeds up after its own trades. This is a
  front-loading feedback that exists only with own orders. It is measured separately (S4b and S5).
- The words "reversal advantage" in the task are kept as the name of the question. The report will say plainly which
  direction the data and the code support.

## 3. Data

### 3.1 Real

Nasdaq ITCH sample days converted by simrel/experiments/data_00_fetch_itch.py, symbols INTC, CSCO, MSFT, all eight
days of simrel/configs/slices.py: 2019-01-30, 2019-03-27, 2019-07-30, 2019-08-30, 2019-10-18, 2019-10-30, 2019-12-30,
2020-01-30. Fit and test days are not distinguished here; nothing is fitted. For each file the report records its
size in bytes and its number of messages.

Session 09:30 to 16:00; the first and last 15 minutes are excluded, so each day is one segment from 09:45:00 to
15:45:00 (21,600 s). A day with a trading halt or an empty side of the book inside the segment keeps its valid part
(grid points with a missing bid or ask are dropped from returns that touch them) and is listed.

Trades: executions of visible and hidden orders (message types 4 and 5). The aggressor is the side opposite to the
resting order (signed volume = -direction x size), as in simrel/data/lobster.py features().

### 3.2 Simulated

Unperturbed passive episodes: 600 s warm-up, then one segment of 3,600 s. 20 seeds per market, 70000 to 70019
(not used anywhere else in the study). Trades are the world's own tape (time, signed shares).

Markets used, and why each is allowed:

| markets | family | why allowed |
|---|---|---|
| R1a_qr_resilient, R1b_qr_fragile, R1c_qr_thin, R1d_qr_momentum, R1e_qr_deep | queue-reactive | hand-built development markets of the study (queue_reactive.regimes()) |
| R2a_am_anchored, R2b_am_trending, R2c_am_anchor, R2d_am_toxic, R2e_am_calm | agent-mix | hand-built development markets (agent_mix.regimes()) |
| XRP_pilot_a01 to a03 | agent-mix | development pilot markets of addendum 11 (stream 20261120), predator off, which is the base world; parameters read from origin/reactive-family with git show |
| XRP_pilot_l01 to l03 | latent | same pilot, predator off |
| XLP_pilot_a01, a04, a06, a09, a11, a15 | agent-mix | development pilot markets of the lambda pilot (stream 20261130), at lambda = 0, which is AgentMix bit for bit; parameters read from origin/lambda-family with git show |
| XLP_pilot_l01, l02 | latent | same pilot, reference markets |

24 markets: 5 queue-reactive, 14 agent-mix, 5 latent. No new market is drawn and no stream is advanced.

Not used:

- Test B markets (simrel/reference_worlds/testb.py and anything under results/simrel/goal2/testB or
  results/simrel/raw/testb*): forbidden. They are neither read nor simulated. The registry() function of
  reference_worlds/data.py builds the Test B draws, so it is not called; worlds are constructed directly from their
  parameter classes.
- R1s01 to 05, R2s01 to 05 and R3s01 to 06 (Test A of addendum 10): their parameter listings
  (results/simrel/raw/sampled_markets.json, latent_markets.json) are not on this machine. They would be allowed
  (already read once) and can be added later with the same code.
- The Hawkes-built markets W1 to W6, the ablation markets and the identifiability grid: not needed for the question.

## 4. Statistics

All statistics are computed by one module (simrel/metrics/reversal.py) from two inputs per segment: the 5 s quote
grid (the quote in force at each grid time, as simrel/data/lobster.py grid_quotes builds it for real data and as the
engine and reference_worlds/data.py observe() build it for simulated data), and the list of trades (time, signed
shares). Mid = (bid + ask) / 2. A segment is one real day (09:45 to 15:45) or one simulated episode (3,600 s).

- **S1.** Lag-1 autocorrelation of 5 s mid changes, by the formula of simrel/metrics/realism.py (_acf1), pooled over
  segments: changes are demeaned within the segment, and S1 = sum of products of consecutive changes / sum of
  squares, the sums running over all segments. Products do not cross segment boundaries.
- **S2.** The same for non-overlapping 30 s mid changes (every sixth grid point from the segment start). This is
  P6's horizon.
- **S3.** For each 30 s bin j >= 1 of a segment: x_j = P6's signal for a sell parent computed from bin j-1 as in
  section 2 (imb from the trades of bin j-1, ret from the mid change over bin j-1), and y_j = mid change over bin j in
  bp of the mid at the start of bin j. Ordinary least squares of y on x with an intercept, pooled over segments.
  Reported: the slope (bp per unit of signal) and its t-statistic, with standard errors clustered by segment. A
  positive slope means continuation. For a buy parent the signal changes sign and the slope does too, so nothing new
  is learned; only the sell form is reported.
- **S4, impact-free timing backtest.** Each segment is cut into non-overlapping 600 s windows (36 per real day, 6 per
  simulated episode). In each window the study's task is run with no own impact: 20 decisions, 30 s apart, sell,
  parent size Q. At decision k the strategy sees the public observation only: the quote at the decision time, the
  volume and signed volume of the trades in the previous 30 s, and the mid change over the previous 30 s (zeros at
  k = 0). Child sizes come from the study's own objects, candidate_set(0.15)["P0_twap"] and
  ["P6_signal_reversal"], passed through core.child_quantity with lot 100 (which includes the rule that the last
  decision sends everything left). Each child is valued at the mid at its decision time.
  Timing cost = sum over children of q_k x (m_0 - m_k) / (Q x m_0) x 10^4, in bp of the arrival mid m_0.
  S4 = cost of P6_signal_reversal minus cost of TWAP. Negative means P6 is cheaper. Reported: mean, standard error,
  and the share of windows in which P6 is strictly cheaper (ties reported separately).
  Q = 15% of the market's or stock's mean traded volume per 600 s window over all its windows in this check, rounded
  down to 100 shares. Impact-free, Q matters only through rounding.
  The buy parent is run as a check and reported in a secondary column; the sell parent is the headline, as in the
  study.
- **S4b (description).** The same backtest, with the engine's rule that the trader's own previous child is added to
  the previous slice's volume and signed volume, and still no effect on prices. S4b minus S4 is what the own-flow
  feedback in the signal does to timing when prices do not react.
- **S5 (simulation only): own impact.** For each simulated market, truth runs with eventsim.run_twin_set on seeds
  0 to 99: the standard task (sell, 600 s, 20 decisions, 120 s post window), Q as above, policies P0_twap,
  P6_signal_reversal and a spy. The spy is a policy that requests nothing and records the observation at each
  decision; the harness sends its whole order at the last decision, after the record is complete, so the spy sees
  the unperturbed twin's path. G = mean over seeds of (implementation shortfall of P6 minus that of TWAP), in bp.
  S4_twin and S4b_twin = the impact-free backtests on the spy's observations of the same seeds.
  S5 = G - S4_twin: the part of P6's true advantage that is not background predictability. It is split into
  S4b_twin - S4_twin (own-flow feedback on timing, prices not reacting) and G - S4b_twin (the rest: how prices respond
  to the orders). The study's stored truth runs are not on this machine, so G comes from these new runs; for the
  pilot markets the seeds and the task are those of the pilots.
- **Context.** For every stock and market: tick size in bp of the mean mid; mean spread in ticks and in bp; share of
  grid points at which the spread is exactly one tick; S1 and S2 are reported next to these. Large-tick mid-quote
  bounce mechanically produces negative short-horizon autocorrelation.

Uncertainty. Real: bootstrap over the 8 days of a stock (2,000 resamples), percentile 95% intervals; the three
stocks are reported one by one and not pooled. Simulated: per market, bootstrap over seeds; per family, bootstrap
over markets with seeds resampled within markets (2,000 resamples). With eight days the real intervals are rough;
this is stated in the report.

## 5. Comparison of real and simulated

One table: S1 to S4 (and S4b) for each stock and for each simulated family, pooled and per market, with intervals;
S5 and its split for the simulated markets; the tick and spread context. Simulated families are compared with each
stock separately. Where tick size differs a lot, like is compared with like: stocks whose spread is one tick most of
the time (expected: INTC and CSCO) against the simulated markets (one tick is 10 bp of the price and the spread is
about one tick), and the report says how much the comparison depends on tick size.

"Shows an impact-free gain" means: the 95% interval of S4 lies below zero.

## 6. Interpretation (fixed now, applied in the report)

- **Benchmark property.** If a simulated family shows an impact-free gain (S4 interval below zero) but no real stock
  does, and for S2 or S3 the real estimates have the opposite sign or are less than one third of the simulated size,
  then the part of P6's advantage that comes from background predictability is likely a property of the benchmark
  and not of real markets.
- **Realistic on this point.** If at least two of the three stocks show an impact-free gain, with S4 at least one
  third of the simulated size and S2 and S3 of the same sign as simulated, the benchmark is realistic on this point.
- **Mixed.** Anything else is reported as mixed, with the numbers.
- **No background channel in simulation.** If the simulated families do not show an impact-free gain themselves,
  then P6's simulated advantage is not background predictability at all; it is own impact (S5), and real passive data
  cannot confirm or refute it. The real statistics are then reported as description only.
- Either way, S5 says how much of P6's simulated advantage comes from own impact, which this check cannot test on
  real data. The report says so plainly.
- Because P6_signal_reversal as coded trades with the last move (section 2), the report states for each family and
  stock whether the predictability found is continuation or reversal, and does not use the strategy's name as
  evidence of either.

## 7. Outputs

- Code: simrel/metrics/reversal.py, simrel/experiments/reversal_check.py, simrel/tests/test_reversal.py.
- Results: results/simrel/goal2/reversal_check/ (tables as CSV; raw files not committed).
- Report: simrel/reports/reversal_check.md.
- The simulated side is computed first and committed as a table only, without interpretation, before the real data
  are available.
