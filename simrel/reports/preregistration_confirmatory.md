# Pre-registration: confirmatory replication on CSCO and MSFT

Written 31 October 2026, before any simulator was fitted to, run on or scored against CSCO or MSFT data, and before
any CSCO or MSFT window was described or sliced. The day files for both symbols were converted on 12 October 2026
alongside INTC; only their message counts and integrity checks (crossed books, unknown references) have been viewed.

Everything below is fixed. Any deviation will be reported as a deviation, with its reason.

## 1. Purpose

The INTC study was exploratory: about 25 Hawkes versions, fixes chosen after inspecting realism scores, and an
evaluation definition extended twice. This replication tests whether its main findings hold on data that played no
part in any of those choices.

## 2. Data

- Symbols: **CSCO** (primary; a large-tick stock like INTC) and **MSFT** (secondary; a smaller tick relative to its
  price, to test generalisation).
- Days: the same eight Nasdaq TotalView-ITCH 5.0 sample days. Fit days 2019-01-30, 2019-03-27, 2019-07-30,
  2019-08-30, 2019-10-18; held-out days 2019-10-30, 2019-12-30, 2020-01-30.
- Windows: 09:45 to 15:45 cut into ten-minute windows; only events that change the visible ten-level book, plus all
  executions (as in a ten-level LOBSTER file).

## 3. Slices

- Same three descriptors as INTC: volatility (s.d. of 5-second mid returns), visible executed volume, absolute
  ten-minute mid move. Each split at its median over **that symbol's** 288 windows.
- A slice is used if it has at least 4 fit windows and at least 4 held-out windows.

## 4. Simulator versions (frozen; code as of this date)

| Label in this document | Version name in the code | Why it is included |
|---|---|---|
| Hawkes-original | `hawkes` | Original statistical model |
| Hawkes-queue | `hawkes_sd_q_c` | Best order-book score on INTC; most extreme advice |
| Hawkes-refill | `hawkes_sd_c_f_m` | Best version without the queue fix; converged with ABIDES-calibrated on INTC |
| ABIDES-original | `abides` | Original agent-based model |
| ABIDES-calibrated | `abides_quoting` | Agent-based model with data-fitted quoting agents |

Fitting procedures, grids, warm-up, seeds and all constants are exactly those in `simrel/configs/slices.py` as of this
date. Each symbol gets its own fits; nothing fitted on INTC is reused. TRADES is not part of the confirmatory study
(there is no trained CSCO or MSFT model, and its INTC training was unstable across seeds); it stays an INTC case study.

## 5. Task and strategies

- Sell a parent order of **10%** and of **30%** of the slice's mean visible executed volume per window, over ten
  minutes in 20 decisions, remainder forced at the last decision.
- Strategies: TWAP, immediate, liquidity-aware, signal-momentum (the INTC "version subset").
- 20 seeds per slice, version and size (seeds 40000–40019), paired with an unperturbed twin.

## 6. Measures

- **Decision statistic D**: mean over seeds and slices of (cost of TWAP − cost of immediate), implementation
  shortfall in bps. Positive = the simulator recommends immediate execution.
- **Realism**, scored against the held-out windows exactly as for INTC: order book (LOB-Bench 21 default statistics,
  mean L1), price (five return and volatility statistics, mean Wasserstein), impact (LOB-Bench impact score).

## 7. Hypotheses and decision rules

Intervals are 95% paired bootstrap intervals over seed-by-slice pairs (2,000 resamples). "Holds" is judged per symbol;
CSCO is primary, MSFT is reported in the same way as a secondary test.

- **H1 (original simulators disagree).** At 10%: D(Hawkes-original) > 0 and D(ABIDES-original) < 0, both intervals
  excluding 0, and D(Hawkes-original) − D(ABIDES-original) ≥ 2 bps.
- **H2 (a better order-book score can mean more extreme advice).** Hawkes-queue has a lower order-book L1 than
  Hawkes-original (mean over slices), and D(Hawkes-queue) − D(Hawkes-original) > 0 with its interval excluding 0,
  at 10%.
- **H3 (refill-realistic versions roughly agree).** At 10%: |D(Hawkes-refill) − D(ABIDES-calibrated)| < 1 bp, tested
  as equivalence (two one-sided tests at the 5% level, margin ±1 bp on the paired-by-slice difference of the two
  versions' slice means, bootstrap over slices and seeds). Reported for 30% too; at 30% this is exploratory.
- **H4 (impact is not solved).** No simulator version's impact score is within twice the real-data floor (real data
  from the fit days scored against the held-out days).
- **H5 (order-book realism does not predict consistent advice).** Across the five versions, the Spearman rank
  correlation between order-book L1 and D at 10% is negative. (With five points this is descriptive, not a test.)

## 8. What will be reported whatever the outcome

All five hypotheses for both symbols and both sizes, all intervals, per-slice tables, every failed or excluded run,
and any deviation from this document. Results that contradict the INTC findings will be reported as such.

## 9. Integrity

This file's SHA-256 checksum is recorded in `preregistration_confirmatory.sha256` next to it. Committing both to
version control before running would make the date verifiable.
