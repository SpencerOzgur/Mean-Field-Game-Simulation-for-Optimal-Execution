# Continuation check: plan (exploratory, written before continuation is joined with any regret)

Status: exploratory. Every regret used here comes from markets whose results have already been read (Test B of
addendum 10, and the development markets). Nothing below is a confirmatory test. This plan is committed on its own
before any statistic that combines continuation with candidate regret is computed.

## 1. Question

The main finding is that selecting a strategy with a fitted simulator has higher regret than always using TWAP. The
reversal check (reversal_check.md, reversal_horizons.md) found that every reference family shows order-flow continuation
that INTC, CSCO and MSFT lack, and the external-validity screen (review_audit.md, sections 4 and 7) found 0 of 400 draws
with a Nasdaq-like price response. Question: does the main finding hold in the reference markets that have little
continuation, or is it driven by the markets that have a lot of it?

## 2. Sources and arms

Primary source: Test B (results/simrel/goal2/addendum10/test_b.csv), 48 markets (16 queue-reactive, 16 agent-mix,
16 latent), two sizes (5% and 15%). Regret columns: `regret_full` is full-truth regret (300 seeds, never negative);
`regret` is cross-fitted regret (best identified on seeds 0 to 149, evaluated on 150 to 299; review_audit.md section 1).
Note: in test_a.csv the names are the other way round (`regret` full, `regret_cf` cross-fitted); Test A is not used.

| Row | Test B arm | Role |
|---|---|---|
| fitted: original Hawkes | original h_c_f_m_k1 | fitted simulator design |
| fitted: compound Hawkes | compound Hawkes | fitted simulator design |
| fitted: corrected | corrected | fitted simulator design |
| fitted: mean of the three | per market, mean of the three rows above | primary summary of the fitted designs |
| frozen rule (H10.1) | rule | reported separately, not a fitted design |
| no-simulator rule | no-simulator | reported separately |
| always TWAP | always TWAP | reference arm |

No ensemble arm is stored for Test B, so none is reported for it.

Development source (development only, never pooled with Test B for inference): the 16 core development markets of
results/simrel/goal2/addendum10/dev_fourway.csv (R1a to R1e queue-reactive, R2a to R2e agent-mix, W1 to W6 Hawkes-built).
One regret definition (the development truth; no cross-fitted version). Fitted rows: original h_k1, original
h_c_f_m_k1, recalibrated h_k1, recalibrated h_c_f_m_k1, compound Hawkes, corrected, and their per-market mean; the
ensemble of all designs (ensemble_all) from results/simrel/goal2/heldout/goal2_A2_A3_selection.csv, whose always-TWAP
column matches dev_fourway.csv. Rule rows: frozen rule (design set D) and corrected + frozen rule, separately. There is no
no-simulator arm in this source.

Not used:
- Test A (R1s01 to 05, R2s01 to 05, R3s01 to 06): the parameter listings results/simrel/raw/sampled_markets.json and
  latent_markets.json are not on this machine, and rebuilding them would require rerunning the viability screen. If
  the files are supplied, Test A is added later under exactly the rules of this plan.
- The 26 ablation markets (Xqr_*, Xam_*): perturbations of two base markets, not independent markets. If cheap, their
  S4 is reported in a descriptive table only (which mechanism removal reduces continuation), outside all comparisons.

## 3. Continuation measure

Computed exactly as in the reversal check, with simrel/metrics/reversal.py unchanged and the passive-episode loop of
reversal_check.passive_job (600 s warm-up, one 3,600 s segment, 5 s grid), worlds built by reference_worlds.data.make:

- 20 passive episodes per market, seeds 70000 to 70019 (the reversal check's seeds).
- Parent quantity: reversal.parent_qty (15% of the mean passive volume per 600 s window, in lots of 100).
- **S4 (primary), in ticks:** the mean over all 600 s windows of the impact-free P6-minus-TWAP timing cost (sell,
  reversal.window_gaps "s4"), in bp, divided by the market's tick in bp (reversal.context tick_bp). Negative = P6 cheaper
  = continuation.
- **S3 slope (secondary), in ticks per unit of signal:** reversal.s3_slope divided by tick_bp.
- Per-market 95% intervals by bootstrap over seeds (reversal_check.describe, 2,000 resamples).
- For R1a to R1e and R2a to R2e, the stored passive summaries of the reversal check
  (results/simrel/goal2/reversal_check/raw/sim_passive.pkl, read only) are used; they were produced by the same code and
  seeds. As a check, S4 recomputed from them must equal sim_table.csv.

Real-stock reference (reversal check, 8 days, S4 in ticks with 95% intervals): INTC -0.018 [-0.058, +0.027],
CSCO +0.015 [-0.027, +0.058], MSFT -0.028 [-0.089, +0.032].

## 4. Classification rule (fixed from the real data, not from regret)

- A market is **low-continuation** if S4 >= -0.05 ticks (point estimate). Otherwise it is in **the rest**.
- Sensitivity: the same with -0.10 ticks. Everything in sections 5 to 7 is reported for both thresholds; the -0.05
  threshold is the primary one.

## 5. Order of checks (each before any regret by group is seen)

1. Compute S3 and S4 for every eligible market. Write the per-market table and the counts per family and source.
2. **Feasibility stop:** if fewer than 8 Test B markets are low-continuation (primary threshold), stop. Report the count
   by family and the distribution of S4 across markets, and state that the benchmark cannot test the question, as with
   the external-validity result. No regret is joined. The development source is too small to replace Test B; it is
   reported under the same count only, and joined with regret only if Test B passes.
3. **Composition check:** report the low-continuation subset's composition by family. If it is mostly (more than half)
   queue-reactive, this is stated, and step 4 decides whether to go on.
4. **Headroom condition:** the only regret looked at here is always-TWAP's full-truth regret in the low-continuation
   subset (mean over markets of the two sizes' mean). If it is below 0.5 bp, the comparison is uninformative (nothing to
   gain): the report says so, does not claim that simulators lose or win, and the comparisons of section 6 are not run.
   If the subset is mostly queue-reactive and fails headroom, stop and report that.
5. Only then: the comparisons of section 6.

## 6. Comparisons

Unit: the market. For each market and arm, regret is averaged over the two sizes; the per-market difference is that
average minus always-TWAP's. Computed separately in the low-continuation subset, the rest, and their difference
(low minus rest), for full-truth and cross-fitted regret (Test B) and the development truth (development source):

- each fitted row minus always TWAP, and the mean of the fitted designs minus always TWAP;
- frozen rule minus always TWAP (separate row);
- no-simulator rule minus always TWAP (Test B only; expected to lose its advantage in the low-continuation subset).

Intervals: 95% percentile intervals from a market bootstrap stratified by family (within each subset, markets are
resampled with replacement within each family present; the difference is the difference of independent resamples of
the two subsets), 2,000 resamples, numpy default_rng(0). A family with one market in a subset is kept with that market.

## 7. Readings (fixed in advance)

Primary statistic: the mean of the three fitted designs minus always TWAP, Test B, full-truth regret, primary threshold.
The individual designs, cross-fitted regret and the -0.10 threshold are secondary and are reported whether or not they
agree; disagreements are stated.

- **Not dependent on continuation:** headroom holds, the low-continuation estimate is positive with its 95% interval
  above zero, and the low-minus-rest difference is not negative with an interval below zero. Reading: simulators are still
  worse than TWAP in the low-continuation subset; the main finding does not depend on the benchmark's continuation.
- **Partly driven by continuation:** the low-continuation estimate is zero or negative, or the low-minus-rest difference
  is negative with its interval below zero (the gap shrinks or reverses). Reading stated plainly: the main finding is
  partly driven by continuation. If the estimate is still positive with an interval above zero but the difference is
  significantly negative, both facts are stated (still worse, by less).
- **Cannot answer:** fewer than 8 low-continuation markets, or no headroom. A limitation, not a result. If the
  low-continuation estimate is positive but its interval includes zero and the difference interval includes zero, the
  subset is too noisy to distinguish the readings, and this is reported as inconclusive.

## 8. Outputs

- results/simrel/goal2/continuation_check/: per-market S3, S4 and classification (markets.csv); counts; comparison
  tables; ablation S4 table if run; figure: each Test B market's fitted-design regret minus TWAP (full-truth, one panel per
  design and one for their mean) against its S4 in ticks, coloured by family, with the union of the real stocks' 95%
  intervals (-0.089 to +0.058 ticks) shaded and the two thresholds marked.
- simrel/reports/continuation_check.md: inventory, counts, feasibility and headroom, comparisons, the reading that
  applies, limits. It states that it is exploratory and uses already-read test markets.

No existing report, addendum, note, freeze, checksum or results file is edited; eventsim.py, core.py, lob.py, the
policies, the worlds and simrel/metrics/reversal.py are reused unchanged.
