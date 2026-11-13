# Reactive family: pilot report (addendum 11, development stage)

Written 13 November 2026. Development only. No candidate simulator was fitted to a reactive market, no trust rule was
applied, no held-out market was drawn, and nothing under results/simrel/goal2/testB/ or results/simrel/raw/testb* was
read. The pilot markets use their own stream, default_rng(20261120), and the prefix XRP_pilot_. That stream must not be
used again.

Outputs: results/simrel/goal2/addendum11_pilot/ (CSV per experiment, four figures, pilot_markets.json,
reference_volume.json). Raw pickles are in the raw/ subfolder and are not committed (62 MB).

## 1. What was built

| file | content |
|---|---|
| simrel/reference_worlds/reactive.py | the predator, its parameters, the detector, the wrapper world, the pilot market draw |
| simrel/policies/reactive.py | P0r_twap_random and reactive_candidate_set |
| simrel/tests/test_reactive.py | 22 tests |
| simrel/experiments/reactive_pilot.py | the pilot runs (timing, run, report) |
| simrel/experiments/reactive_pilot_report.py | tables and figures |

No shared file was changed: eventsim.py, core.py, lob.py, candidate_set and the existing worlds are untouched. Tests:
42 pass (20 existing, 22 new) under .venv-abides with `python -m pytest simrel/tests`.

One change to the environment: the existing LOBSTER converter used by the diagnostics imports sortedcontainers, which
was not installed. It was added with `pip install --no-deps sortedcontainers` (2.4.0). numpy (1.26.4) and pandas
(1.5.3) were not touched.

### 1.1 The predator

The predator wraps an agent-mix or latent world. It sees one thing: the public trade log, as rows of (time, side,
size). Every trade, the trader's or the background's, reaches it through the same call, with no label. A test checks
that relabelling who sent a trade changes nothing.

- Time stamps. Each trade time is shifted by Gaussian noise (s.d. `ts_noise_s`, 0.2 s by default, clipped at three
  s.d.) before the predator sees it. The noise comes from the predator's own stream.
- Detection. Within the last `window_s` (240 s), the detector looks for at least `min_matches` same-side trades that
  are evenly spaced (coefficient of variation of gaps below `gap_cv_tol`) and of similar size (coefficient of variation
  of sizes below `size_cv_tol`).
- Front-running. Once triggered, it sends a market order on the same side `lead_s` (2 s) before each expected child,
  of size `aggression` times the estimated child size, up to `inventory_cap` shares.
- Unwinding. It starts to unwind after 1.5 periods without a matching trade, or after `max_hold_s`. It unwinds with
  half-life `unwind_halflife_s` and must be flat 45 s after it starts.
- Effect on the market. Its orders go through the same path as a background market order: they hit the book, appear
  on the tape and feed the base world's flow signals.
- Random numbers. The predator has its own stream, default_rng([seed, 11, 0]). With aggression 0 or cap 0 the wrapper
  passes straight through and the market is bit-identical to the base world (tested for both families, including the
  event log).

### 1.2 Detectors

| name | role | ts_noise_s | gap_cv_tol | size_cv_tol | min_matches |
|---|---|---|---|---|---|
| noisy | family default | 0.2 | 0.03 | 0.15 | 8 |
| tight | comparison arm | 0 | 0.0005 | 0.5 | 5 |
| loose | negative control | 0 | 0.01 | 0.5 | 5 |

The first version was the tight detector. It relied on the engine placing children on an exact 30 s grid, and its size
test did almost nothing. After review the default became the noisy detector: time stamps carry 200 ms of noise, the gap
test is wide enough to absorb it, and size regularity does most of the work. The tight detector is kept as three
comparison arms and the loose one as a negative control.

### 1.3 Strengths

| strength | aggression | inventory cap (share of 10-minute reference volume) |
|---|---|---|
| off | 0 | 0 |
| weak | 0.5 | 1% |
| medium | 1.0 | 3% |
| strong | 2.0 | 8% |

Strength changes only aggression and the cap. The detector is the same at every strength. The cap and the order size
are both taken from the predator-off market's volume, so the order is the same number of shares at every strength.

### 1.4 P0r_twap_random

A TWAP that skips each slice but the last with probability 0.3 and weights the others by 1 + 0.5 x uniform(-1, 1).
The plan is drawn once from default_rng([crc32(world), seed, 0x503072]) and does not touch a market stream. It is
offered only through `reactive_candidate_set`; `candidate_set` is unchanged. Children can only be sent at slice
starts, so the randomisation is in sizes and skipped slices, not in timing within a slice.

## 2. Design decisions and why

1. Hard flatten deadline. The predator must be flat 45 s after it starts to unwind. Inventory left at the end of the
   post window is marked at the final mid and reported. This happened in 1.5% to 1.8% of true-cost runs.
2. The wrapper does not split the base world's `advance` call when the predator is inactive. Extra ticks change
   floating-point decay in the thinned worlds, which would break bit-identity at strength 0.
3. Per-branch predator state is read through a small Recorder object that survives the deep copy `run_twin_set` makes.
   This avoids any change to the engine. It is a workaround and is listed under open problems.
4. The detector picks, among trades that fit the time grid, the one closest in size. This makes it follow a schedule
   through background trades that happen to fall on the grid.
5. After a max-hold exit the predator does not re-trigger on the same pattern, so that a long schedule is not
   front-run twice.
6. The diagnostics (E3) need a classifier and sklearn is not installed. A ridge logistic regression with stratified
   five-fold cross-validation was written in numpy and scipy.

## 3. Pilot design

Six markets on stream default_rng(20261120): three agent-mix (XRP_pilot_a01 to a03) and three latent (l01 to l03),
drawn from the existing ranges. Standard task (sell, 10 minutes, 20 slices, 2 minutes post window), sizes 5% and 15%.

Arms: off, weak, medium, strong (noisy detector); weak_tight, medium_tight, strong_tight (tight detector);
strong_loose (loose detector, passive data only). The full run took about 55 minutes on 7 processes.

| experiment | runs |
|---|---|
| E4 viability | 40 unperturbed twins per market and arm |
| E1 detection | detector replayed on the off arm (the predator watches but does not trade), 100 seeds, 13 strategies and no order; 46 passive 30-minute episodes per market and arm |
| E2 true costs | 100 seeds per market, arm, size and strategy |
| E3 observability | 20 passive three-hour samples per market and arm, plus 20 more predator-off samples on other seeds |

## 4. Results

### 4.1 E4: viability

All 48 market and arm cells pass the existing rule (no failed runs; share of twins moving more than 300 bp at most
0.025). Volume, spread and volatility change by less than 2% between off and strong in five markets. In a03 the
predator-on arms have about 5% less volume and lower volatility than off over the same seeds, which is path divergence
in a market with crash-like episodes, not a systematic effect.

### 4.2 E1: how each strategy looks to the detector

Trigger rate, predator not trading, six markets pooled (figure fig_e1_trigger_rates.png, table
e1_detection_by_strategy.csv). Family detector: 100 seeds. Tight detector: 30 seeds.

| strategy | children sent | size CV of children | family 5% | family 15% | tight 5% | tight 15% |
|---|---|---|---|---|---|---|
| P0_twap | 20 | 0.23 / 0.13 | 0.54 | 0.87 | 1.00 | 1.00 |
| P5_liquidity | 17 to 20 | 0.5 | 0.40 | 0.33 | 0.99 | 1.00 |
| P7_risk_0.5 | 14 to 16 | 0.7 to 0.9 | 0.25 | 0.25 | 0.74 | 0.77 |
| P2_front_b2 | 18 to 20 | 0.5 | 0.23 | 0.09 | 1.00 | 1.00 |
| P3_back_b-2 | 17 to 20 | 0.5 | 0.23 | 0.08 | 1.00 | 1.00 |
| P7_risk_2 | 9 to 10 | 1.1 | 0.20 | 0.21 | 0.20 | 0.28 |
| P6_signal_momentum | 16 to 20 | 0.6 | 0.17 | 0.11 | 0.92 | 1.00 |
| P6_signal_reversal | 16 to 20 | 0.5 | 0.17 | 0.15 | 0.94 | 1.00 |
| P4_pov_1x | 15 to 16 | 1.0 / 0.5 | 0.16 | 0.10 | 0.78 | 0.99 |
| P2_front_b4 | 14 to 18 | 0.7 to 0.9 | 0.12 | 0.09 | 1.00 | 1.00 |
| P0r_twap_random | 14 | 0.3 | 0.10 | 0.06 | 0.83 | 0.84 |
| P4_pov_2x | 8 to 11 | 0.5 | 0.07 | 0.07 | 0.89 | 0.92 |
| P1_immediate | 1 to 2 | - | 0.05 | 0.06 | 0.02 | 0.09 |
| no order | - | - | 0.05 | 0.05 | 0.01 | 0.01 |

Median time of the first trigger under the family detector is 210 to 240 s for TWAP (it needs eight children) and 270
to 430 s for the schedules that are caught by chance.

Reading:

- At 15% the family detector is TWAP-specific: 0.87 for TWAP against 0.07 to 0.10 for POV and 0.09 for both
  front-loaded schedules.
- At 5% it is much less so: 0.54 for TWAP against 0.40 for P5_liquidity and 0.23 for P2_front_b2 and P3_back.
- The tight detector is not TWAP-specific at all. Every strategy that sends a child at most slice starts is caught
  nearly always. Only P1_immediate and P7_risk_2 escape. Under it the 13 strategies fall into two groups, "on the grid"
  and "not on the grid", and inside the first group they are indistinguishable.
- P5_liquidity and the two P7 strategies trigger the family detector in a fifth to two fifths of runs. They often send
  a run of equal children (P5 when the book is steady; P7 in its tail).

TWAP's trigger rate depends on lot rounding, market by market:

| market | TWAP child at 5% (shares) | trigger rate 5% | TWAP child at 15% (shares) | trigger rate 15% |
|---|---|---|---|---|
| a01 | 222 | 0.49 | 665 | 0.99 |
| a02 | 110 | 1.00 | 331 | 1.00 |
| a03 | 270 | 0.54 | 811 | 0.97 |
| l01 | 142 | 0.06 | 426 | 1.00 |
| l02 | 93 | 1.00 | 280 | 0.24 |
| l03 | 139 | 0.17 | 837 | 1.00 |

Children are rounded to lots of 100 shares. A TWAP whose child is close to one lot sends equal children and is always
caught. One whose child is about 1.4 or 2.8 lots sends a mix of 100 and 200, or 200 and 300, which fails a size test
of 0.15, and is rarely caught. So whether TWAP is detected is decided mostly by the arithmetic of order size and lot
size, not by the strategy.

Sensitivity of the detector (replay on 30 seeds, e1_detection_by_strategy.csv, columns rate_*):

| setting | TWAP 5% | TWAP 15% | P2_front_b2 5% / 15% | P3_back 5% / 15% | no order |
|---|---|---|---|---|---|
| default (size 0.15, 8 matches) | 0.53 | 0.87 | 0.24 / 0.09 | 0.24 / 0.12 | 0.06 |
| noise 0 s | 0.55 | 0.87 | 0.22 / 0.12 | 0.24 / 0.13 | 0.06 |
| noise 0.3 s | 0.51 | 0.87 | 0.23 / 0.09 | 0.24 / 0.12 | 0.06 |
| size 0.10 | 0.48 | 0.54 | 0.22 / 0.06 | 0.22 / 0.05 | 0.06 |
| size 0.25 | 0.73 | 1.00 | 0.73 / 0.95 | 0.62 / 0.94 | 0.11 |
| size 0.50 | 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.38 |
| 6 matches | 0.77 | 1.00 | 0.76 / 0.75 | 0.77 / 0.78 | 0.30 |
| 7 matches | 0.61 | 1.00 | 0.53 / 0.31 | 0.50 / 0.31 | 0.15 |

The result does not depend on the noise level between 0 and 0.3 s. It depends strongly on the size tolerance: there is
no setting that catches small TWAPs reliably without also catching the tilted schedules.

### 4.3 E1: false triggers on passive data

46 passive 30-minute episodes per market (the six observable episodes plus 40 more), no order in the market
(e1_false_triggers_passive.csv).

| market | family detector: triggers per episode | share of episodes with a trigger | predator share of volume, strong | tight detector: triggers per episode | loose detector: triggers per episode |
|---|---|---|---|---|---|
| a01 | 0.61 | 0.46 | 0.2% | 0.13 | 12.5 |
| a02 | 0.07 | 0.04 | 0.0% | 0.02 | 5.2 |
| a03 | 1.17 | 0.30 | 0.2% | 0.67 | 13.4 |
| l01 | 0.02 | 0.02 | 0.0% | 0.00 | 8.2 |
| l02 | 0.00 | 0.00 | 0.0% | 0.02 | 4.4 |
| l03 | 0.15 | 0.13 | 0.1% | 0.02 | 10.4 |

Mean over the six markets: 0.34 false triggers per 30-minute episode under the family detector, 0.14 under the tight
detector. The loose detector triggers 4 to 13 times per episode and the predator is then 4% to 7% of volume.

Two markets leak:

- a03. This is recorded as a finding and the detector was not tightened to remove it. The market has crash-like
  stretches in which the bid side empties and sell trades of similar size arrive several times per second. Any
  detector of regular same-side trades fires there. Under the tight detector the rate is 0.67 per episode in 13% of
  episodes (one episode had 17 triggers). Under the family detector it is 1.17 per episode in 30% of episodes. The
  predator therefore does trade in this market's passive data.
- a01. The move to a size-led detector made this market worse: 0.61 per episode, in 46% of episodes, against 0.13
  under the tight detector. A wider gap tolerance lets more background runs through.

The amount traded is small (at most 0.2% of volume at strong), but the predator is not absent from the passive data of
these two markets. In the three-hour samples of E3 the predator changed the path of the market in 95% of samples in
a03, 90% in a01, 55% in l03, 30% in l01, 15% in a02 and 5% in l02.

In the true-cost runs with no order at all, the detector triggers in 5% of 12-minute windows.

### 4.4 E2: true costs

100 seeds per cell (e2_costs.csv, e2_summary.csv, figures fig_e2_cost_vs_twap.png and fig_e2_headroom.png).

Market a03 is set aside in the averages below. Its costs have standard errors of 3 to 8 bp at 100 seeds, so nothing
about a 1 bp effect can be read from it. It is in the CSV files and in the figures.

Mean over the other five markets, family detector:

| size | strength | TWAP cost minus off (bp) | always-TWAP regret (bp) | regret, cross-fitted (bp) | P0r minus TWAP (bp) | predator P&L against TWAP (bp of order value) |
|---|---|---|---|---|---|---|
| 5% | off | 0.00 | 1.04 | 0.63 | 0.10 | 0.0 |
| 5% | weak | -0.08 | 1.11 | 0.78 | 0.08 | -1.1 |
| 5% | medium | 0.17 | 1.34 | 1.00 | -0.22 | -1.3 |
| 5% | strong | -0.03 | 1.02 | 0.65 | 0.03 | -2.7 |
| 15% | off | 0.00 | 0.77 | 0.57 | 0.33 | 0.0 |
| 15% | weak | 0.03 | 0.81 | 0.62 | 0.28 | -0.6 |
| 15% | medium | 0.37 | 1.15 | 0.86 | -0.09 | -2.2 |
| 15% | strong | 1.28 | 1.89 | 1.58 | -0.95 | -7.5 |

Always-TWAP regret is the headroom: the true cost of TWAP minus the true cost of the best strategy. The plain figure
picks the best strategy and evaluates it on the same 100 seeds and is biased upwards. The cross-fitted figure picks on
one half and evaluates on the other.

Reading:

- At 15% the predator raises the cost of TWAP in order of strength: 0.03, 0.37 and 1.28 bp. Per market at strong:
  a01 1.45 (s.e. 0.53), a02 1.85 (0.31), l01 0.88 (0.13), l02 0.13 (0.10), l03 2.09 (0.30). The exception, l02, is
  the market where lot rounding hides TWAP at 15%.
- The effect is on TWAP only. At strong and 15% every other strategy's mean cost moves by 0.33 bp or less.
- Headroom at 15% rises from 0.8 bp (off) to 1.9 bp (strong). Weak adds nothing measurable.
- At 5% there is no measurable effect at any strength. TWAP is detected in half the runs and late, and the orders the
  predator sends are small.
- The best strategy does not change. In nine of the ten cells it is the same at off and strong (P6_signal_reversal or
  P5_liquidity). The predator makes TWAP worse; it does not change what is best.
- P0r is 0.95 bp cheaper than TWAP at strong and 15%, and no different from TWAP with the predator off. It avoids the
  predator, as intended.
- The predator loses money. Against TWAP at strong and 15% it loses 7.5 bp of the order's value while costing the
  trader 1.3 bp. It trades with market orders and pays the spread (10 to 14 bp) on each round trip, and a child of a
  15% order does not move the price that far.

Tight detector, same five markets: at strong the TWAP cost rises by 0.97 bp at 5% and 1.66 bp at 15%, the predator
loses 22 bp and 9 bp, and the best strategy changes to P0r_twap_random in three of five markets at 15%. Because the
tight detector catches nearly every strategy, it raises all their costs and adds little headroom (cross-fitted regret
0.78 and 0.87 bp at strong).

### 4.5 E3: can the predator be seen in passive data

Eight diagnostic features of addendum 10 (H, gamma_flow, R100_over_R1, block_ac25, vr30, vr120, vr600, depth_flow) on
three-hour passive samples; 20 predator-on samples against 20 predator-off samples on other seeds per market;
features standardised within market and pooled; cross-validated logistic regression (e3_auc.csv, figure
fig_e3_features.png).

| arm | pooled AUC | permutation 95th percentile | permutation p | predator share of volume |
|---|---|---|---|---|
| off against off (null) | 0.52 | 0.57 | 0.32 | 0 |
| weak | 0.56 | 0.57 | 0.10 | 0.1% |
| medium | 0.55 | 0.58 | 0.16 | 0.1% |
| strong | 0.53 | 0.57 | 0.30 | 0.2% |
| weak_tight | 0.53 | 0.57 | 0.23 | 0.0% |
| medium_tight | 0.55 | 0.58 | 0.14 | 0.0% |
| strong_tight | 0.54 | 0.56 | 0.22 | 0.1% |
| strong_loose (negative control) | 0.62 | 0.57 | 0.005 | 6.1% |

Reading:

- With the family or the tight detector the pooled AUC is 0.53 to 0.56, no different from the null of 0.52. The
  eight features do not see the predator.
- The negative control is detected (0.62), so the test has some power. It is not strong: a predator that is 6% of
  volume gives only 0.62.
- Per market the AUC is too noisy to use with 20 samples per class. The null itself ranges from 0.40 to 0.65 across
  markets, and a03 gives values as low as 0.26.
- "Not visible in eight summary features" is weaker than "absent". Section 4.3 shows the predator trades in the
  passive data of a01 and a03. A detector applied to the passive tape, rather than these features, would find it.

## 5. Open problems

1. Lot rounding decides whether TWAP is detected (section 4.2). In two of twelve market and size cells TWAP is caught
   in under a quarter of runs. Options for the draft addendum: accept and report it; set order sizes so that the TWAP
   child is a whole number of lots; or give the detector a size test in lots rather than a coefficient of variation.
2. The family detector is TWAP-specific only at 15%. At 5% it also catches P5_liquidity (0.40) and the tilted
   schedules (0.23), and the predator has no measurable effect on costs.
3. The predator leaks into passive data in a01 and a03 (section 4.3). The size-led detector is worse than the tight
   one on this count.
4. The predator is unprofitable at every strength. It is a cost imposed on regular schedules, not a rational
   front-runner. Making it profitable would need passive exits or a larger price response to children.
5. Late trigger. Eight matches means the first trigger is at about 210 s of a 600 s order, so at most 12 of 20
   children are front-run.
6. Headroom is small: 1.1 bp added at strong and 15%, nothing at weak or at 5%. With between-market noise of the size
   seen here, a confirmatory test on the weak region would have little power.
7. a03-like markets pass the viability rule but have true costs too noisy for 100 seeds.
8. Residual inventory. In 1.5% to 1.8% of true-cost runs the predator is not flat at the end of the post window,
   because a background trade on the grid keeps the tracking alive. It is marked at the final mid.
9. Engineering. The Recorder relies on deep-copy behaviour of `run_twin_set`. `reactive_candidate_set` needs the world
   name and the seed, which `candidate_set` does not. The E3 sampler steps the market on a 5 s grid for every arm
   rather than in one long call. The AUC code is our own and not sklearn's.
10. Paired on and off feature differences mostly reflect path divergence after the predator's first trade, not a
    systematic shift. The AUC uses independent seeds for that reason.
