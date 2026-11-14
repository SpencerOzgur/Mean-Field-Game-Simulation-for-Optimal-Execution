# Lambda pilot: design, levels and stopping rule (fixed before the main run)

13 November 2026. Development only. This file is committed before the main run of the pilot
(simrel/experiments/lambda_pilot_run.py). When it was written, the only results in existence were calibration results
on calibration seeds: volatility matching on passive episodes, and the displacement of the mid caused by TWAP. No
cost comparison between strategies, no regret and no best strategy had been computed for any market or arm.

## 1. Question

In the agent-mix reference markets the fundamental F is never moved by trading, so all impact fades. The pilot asks
whether two conclusions of the study depend on that: that P6_signal_reversal is often the true best strategy, and
that leaving TWAP gains little. The lambda family is agent-mix with one change: every executed market order,
background or trader, moves F permanently by lambda times its signed size (simrel/reference_worlds/lambda_world.py).

## 2. Scope: the family covers only markets whose volatility can be matched

**Acceptance rate: 6 of the 15 agent-mix draws examined were kept (40%).** All 15 pass the existing viability rule.
Nine were dropped because their volatility could not be shown to match lambda = 0 at every level.

All-flow lambda adds variance to F, so each market's fund_vol is lowered until the variance of 600 s mid returns
equals that of lambda = 0. This fails in two kinds of market:

- collapse-prone markets, in which the book sometimes empties and the price runs hundreds of ticks (1% to 12% of
  600 s windows move more than 300 bp even at lambda = 0). Their variance cannot be estimated tightly enough to
  match;
- markets with a low fund_vol, in which the flow-driven variance at lambda 0.6 or 1.0 is already close to or above
  the whole lambda = 0 variance.

The family is therefore restricted to calmer markets whose volatility can be matched. It excludes collapse-prone
markets, and nothing in this pilot says anything about them.

## 3. Markets

Parameter stream: numpy default_rng(20261130), used for this pilot only and never reused. Draw order: a block of 12
agent-mix draws (sampled.AM_RANGES), 4 latent draws (latent.RANGES), then further agent-mix draws (a13 onward).
Names XLP_pilot_aNN and XLP_pilot_lNN. Streams 20261120, 20261121 and the Test B streams were not touched.

Selection rule for the agent-mix markets: in draw order from a01, keep the first six draws that (i) pass the existing
viability rule at lambda = 0 (40 twins, share of failed runs at most 0.5, share of twins moving more than 300 bp at
most 0.25) and (ii) are volatility-matched at all three levels (section 5). The search is capped at a20; with fewer
than six kept by then, the pilot would have run with those kept (at least four). The sixth market was found at a15,
so there is no shortfall and a16 to a20 were not matched.

Latent reference markets: the first two viable latent draws, XLP_pilot_l01 and XLP_pilot_l02, run at their own
parameters only (the latent family already has persistent impact). They are a reference point and take no part in
the rule.

Pilot agent-mix markets: **a01, a04, a06, a09, a11, a15**.

| market | reference volume (shares per 10 min) | fund_vol at lambda 0 | at 0.25 | at 0.6 | at 1.0 |
|---|---|---|---|---|---|
| a01 | 68040 | 0.211 | 0.195 | 0.169 | 0.136 |
| a04 | 66552 | 0.252 | 0.225 | 0.186 | 0.129 |
| a06 | 56448 | 0.199 | 0.174 | 0.135 | 0.080 |
| a09 | 49185 | 0.158 | 0.140 | 0.114 | 0.080 |
| a11 | 63180 | 0.120 | 0.106 | 0.075 | 0.035 |
| a15 | 72958 | 0.195 | 0.176 | 0.145 | 0.096 |

Market parameters, reference volumes and the viability of every draw: results/simrel/goal2/lambda_pilot/pilot_markets.json.
Arms and matched fund_vol per market: results/simrel/goal2/lambda_pilot/arms.json.

## 4. Levels

Unit: lambda is the permanent price move, in bp, caused by trading 1% of the market's 10-minute reference volume
(lambda = 0 volume of the 40 viability twins). It is converted to ticks per share for each market.

The levels are fixed:

| arm | lambda | role |
|---|---|---|
| lam0 | 0 | baseline (AgentMix, bit for bit) |
| all_dose | 0.25 | all-flow, unjudged dose-response level |
| all_band_low | 0.6 | all-flow, judged |
| all_max_feasible | 1.0 | all-flow, judged |
| trader_band_low | 0.6 | COMPARISON, unjudged: only the trader's orders move F. Uses identity, so it is unrealistic, and its passive data are bit-identical to lambda = 0 by construction. The market's own fund_vol is used |

How the levels were chosen.

- There is no real-data measurement of TWAP impact persistence in the Goal 1 outputs. The reference is a band taken
  from the literature for the share of peak impact that remains after a metaorder ends, 2/3 to 0.87. Neither end was
  measured on our stocks. Sources:
  - 2/3. simrel/reports/lob_impact_design.md uses it without naming a source ("the empirical values delta = 0.5, Y
    in [0.5, 1] (distance to the band), e/p = 2/3: what a practitioner without an intervention on the real market
    would use"). The sources we can identify are: Farmer, Gerig, Lillo and Waelbroeck, "How efficiency shapes market
    impact", Quantitative Finance 13 (2013), whose fair-pricing theory predicts a plateau at 2/3 of peak impact;
    Bershova and Rakhlin, "The non-linear market impact of large trades: evidence from buy-side order flow",
    Quantitative Finance 13 (2013), who report reversion to about 2/3 of peak; and Bucci, Benzaquen, Lillo and
    Bouchaud, "Slow decay of impact in equity markets: insights from the ANcerno database", Market Microstructure and
    Liquidity (2019), arXiv:1901.05332, who find about 2/3 of peak at the end of the same day and further decay over
    the following days. These were checked against secondary summaries (the review arXiv:2105.00521 and the abstract
    of arXiv:1901.05332), not re-read in full for this file.
  - 0.87. This is not an empirical measurement. lob_impact_design.md gives it as "e/p at or below 0.87 (the paper's
    OW kernel value)", and simrel/experiments/impact_02_analyse.py as "the paper's closed-form end-over-peak of the
    Obizhaeva-Wang kernel at this schedule (C5 threshold)". The paper is Nigmatulin et al., "LOB-Impact Benchmark:
    Emergent Market Impact in Generative Limit Order Book Models", ICAIF 2026, arXiv:2610.27100; the kernel is that of
    Obizhaeva and Wang, "Optimal trading strategy and supply/demand dynamics", Journal of Financial Markets 16 (2013).
    The value is a model value for that benchmark's schedule (10 children, 100 cooling windows), used there as an
    upper threshold for relaxation. It was taken from our own files; we could not retrieve the paper to confirm it
    when this file was written.
  - So the lower end has empirical and theoretical support at the horizon of the end of the trading day, with
    evidence of further decay later, and the upper end is a model threshold. No source gives a value for our
    horizon of 120 s after a 10-minute order.
- Simulated persistence is the 120 s retention: the mean displacement of the mid, against the unperturbed twin,
  120 s after the deadline, over the mean displacement at the deadline, for TWAP at the 15% size (the ratio form of
  addendum 5, item 3). At the 5% size the displacements are too small to measure.
- First pass (first six draws, lambda 0.25 to 1.5, 300 calibration seeds per market, the four draws that could be
  matched pooled): retention 0.02, 0.34, 0.69, 0.66, 0.80 at lambda 0, 0.25, 0.5, 0.75, 1.0. About 3.2 bp of the
  deadline displacement recovers at every lambda and the part that stays grows by about 10.9 bp per unit of lambda,
  which gives 2/3 at lambda 0.58, the midpoint of the band at 0.96 and 0.87 at 1.95.
- band_low (0.6) is the common lambda whose pooled retention is about 2/3 on that curve. It is one lambda for all
  markets, not a retention of 2/3 in every market.
- max_feasible (1.0) is the highest level compatible with volatility matching: at 1.5, three of the first six draws
  stay above the lambda = 0 variance even with fund_vol at zero. It is not the 0.87 target.
- **The upper part of the literature band, above about 0.80, is unreachable under volatility matching in these
  markets.** The 120 s retention should be at least as large as the long-horizon ratio, so these markets sit at the
  low-persistence end of the band.

Retention re-measured on the final six markets (300 calibration seeds each, 15% size, bootstrap 95% intervals). This
is description only. The levels are not re-picked from it.

| market | lambda 0 | 0.25 | 0.6 | 1.0 |
|---|---|---|---|---|
| a01 | -0.29 [-0.94, 0.19] | 0.45 [0.17, 0.83] | 0.58 [0.35, 0.85] | 0.63 [0.41, 0.84] |
| a04 | 0.02 [-0.44, 0.49] | 0.26 [-0.08, 0.71] | 0.90 [0.37, 1.78] | 0.76 [0.41, 1.31] |
| a06 | -0.00 [-0.44, 0.34] | 0.59 [0.33, 0.92] | 0.72 [0.52, 0.95] | 0.90 [0.75, 1.07] |
| a09 | -0.29 [-7.96, 6.98] | -0.18 [-1.74, 0.79] | 0.63 [0.35, 0.99] | 0.78 [0.53, 1.07] |
| a11 | 1.98 [-4.02, 8.40] | 0.58 [0.29, 0.93] | 0.88 [0.77, 1.05] | 0.95 [0.84, 1.07] |
| a15 | 0.41 [-0.95, 1.02] | 0.63 [0.23, 0.97] | 0.70 [0.42, 1.02] | 0.65 [0.45, 0.86] |
| pooled | -0.01 [-0.43, 0.31] | 0.39 [0.16, 0.59] | 0.76 [0.64, 0.89] | 0.79 [0.70, 0.89] |
| pooled: displacement at the deadline / 120 s later (bp) | 2.8 / -0.0 | 5.5 / 2.1 | 8.3 / 6.3 | 14.3 / 11.3 |

On the final six the pooled retention at 0.6 is 0.76, above the 0.67 of the first-pass curve, and close to the 0.79
at 1.0. The two judged levels differ clearly in how much displacement stays (6.3 against 11.3 bp) and little in the
retention ratio. At lambda = 0 the deadline displacement is small (2.8 bp), so the ratio there is a ratio of two
small numbers; single-market ratios at lambda 0 mean little.

## 5. Volatility matching

Quantity matched: the mean squared 600 s change of the mid (overlapping windows 60 s apart) in passive episodes of
one hour after a 600 s warm-up. Windows that move more than 300 bp (the wild-move threshold of the viability rule)
are left out and their share is reported; without this, one book collapse in 48 episodes multiplies the variance by
ten.

Search: fund_vol is lowered by a secant search until the ratio to lambda = 0 is within 2% on 192 calibration seeds
(SEED_CALIB + 0..191; the same seeds for both arms). The result is then checked on 48 check seeds that the search
never used (40000..40047).

Criterion (final). A market is matched at a level if the matching is feasible and the bootstrap 95% interval of its
check-seed variance ratio lies entirely within [0.80, 1.25]. A market is kept only if it is matched at all three
levels.

History of the criterion, recorded because it changed after matching results had been seen:

1. First pass: 48 calibration seeds, 5% tolerance on the calibration seeds, no test on the check seeds.
2. Second version: 192 calibration seeds, 2% tolerance, and "the 95% interval of the check-seed ratio includes 1".
3. The second version was replaced by the equivalence criterion above. Reason: it admitted a poorly matched market
   whose interval was very wide (a12, for example 0.71 [0.27, 1.63] at lambda 1.0) and dropped well-matched markets
   by chance across many tests (a03, a09 and a11, whose intervals are narrow and lie just beside 1).

The change from 2 to 3 was made after the matching results of a01 to a13 had been seen, and before any cost, regret
or decision result existed for any market. Under version 2 the first six kept would have been a01, a04, a06, a07,
a12, a13.

Every draw examined, under the final criterion (check-seed ratio with its 95% interval):

| draw | viable | wild windows at lambda 0 | 0.25 | 0.6 | 1.0 | outcome |
|---|---|---|---|---|---|---|
| a01 | yes | 0.2% | 1.01 [0.98, 1.03] | 1.01 [0.97, 1.07] | 0.98 [0.94, 1.03] | kept |
| a02 | yes | 4.1% | 0.65 [0.49, 0.88] | 0.75 [0.52, 1.06] | 0.86 [0.60, 1.23] | dropped: not matched at 0.25, 0.6, 1.0 (collapse-prone) |
| a03 | yes | 0.0% | 1.02 [0.96, 1.08] | 0.90 [0.83, 0.98] | 0.88 [0.77, 1.00] | dropped: not matched at 1.0 (lower bound 0.77) |
| a04 | yes | 0.0% | 1.00 [0.91, 1.10] | 1.04 [0.95, 1.15] | 0.95 [0.86, 1.07] | kept |
| a05 | yes | 5.3% | 0.80 [0.61, 1.06] | 1.05 [0.78, 1.42] | 0.90 [0.70, 1.16] | dropped: not matched at 0.25, 0.6, 1.0 (collapse-prone) |
| a06 | yes | 0.0% | 1.02 [0.98, 1.05] | 0.97 [0.92, 1.02] | 0.97 [0.87, 1.09] | kept |
| a07 | yes | 0.0% | 1.13 [0.98, 1.32] | 1.03 [0.95, 1.16] | 1.04 [0.95, 1.16] | dropped: not matched at 0.25 (upper bound 1.32) |
| a08 | yes | 3.2% | 1.31 [1.04, 1.65] | 1.12 [0.90, 1.38] | 1.26 [0.95, 1.66] | dropped: not matched at 0.25, 0.6, 1.0 (collapse-prone) |
| a09 | yes | 0.1% | 1.00 [0.96, 1.04] | 0.97 [0.90, 1.03] | 0.88 [0.81, 0.97] | kept |
| a10 | yes | 0.1% | 0.99 [0.86, 1.21] | 0.83 [0.75, 0.91] | 0.91 [0.74, 1.12] | dropped: not matched at 0.6, 1.0 |
| a11 | yes | 0.2% | 1.06 [1.04, 1.10] | 0.97 [0.92, 1.02] | 0.95 [0.86, 1.06] | kept |
| a12 | yes | 1.0% | 0.81 [0.34, 1.88] | 1.16 [0.45, 2.73] | 0.71 [0.27, 1.63] | dropped: not matched at 0.25, 0.6, 1.0 (collapse-prone) |
| a13 | yes | 12.0% | 1.11 [0.76, 1.58] | 1.14 [0.80, 1.57] | 1.19 [0.87, 1.58] | dropped: not matched at 0.25, 0.6, 1.0 (collapse-prone) |
| a14 | yes | 0.0% | 1.00 [0.89, 1.12] | 0.74 [0.61, 0.89] | 2.52 [2.12, 3.00] | dropped: not matched at 0.6; infeasible at 1.0 (above target with fund_vol at zero) |
| a15 | yes | 0.0% | 1.05 [0.99, 1.10] | 1.08 [1.00, 1.21] | 1.00 [0.91, 1.09] | kept |

Two kept markets have check-seed ratios whose interval does not include 1 at one level (a09 at 1.0: 0.88; a11 at
0.25: 1.06). They are within the equivalence bounds and are kept.

Other stylised facts of the six kept markets, mean over markets, calibration and check seeds (full table:
results/simrel/goal2/lambda_pilot/calib_final.csv):

| lambda | volume per 10 min | 5 s volatility (ticks) | spread (ticks) | depth, 10 levels | 5 s return autocorrelation | 60 s return autocorrelation | trade-sign autocorrelation |
|---|---|---|---|---|---|---|---|
| 0 | 63965 | 0.579 | 1.08 | 10401 | -0.103 | 0.054 | 0.237 |
| 0.25 | 63480 | 0.585 | 1.10 | 10496 | -0.106 | 0.061 | 0.234 |
| 0.6 | 61908 | 0.579 | 1.08 | 10556 | -0.102 | 0.080 | 0.228 |
| 1.0 | 59736 | 0.575 | 1.07 | 10814 | -0.100 | 0.090 | 0.216 |

Volume falls by 7% at lambda 1.0 and the 60 s return autocorrelation rises from 0.05 to 0.09. Spread, depth and
short-horizon volatility are unchanged.

## 6. Pilot design

- Task: the standard task (sell, 10 minutes, 20 decisions, 2 minutes post window).
- Sizes: 5% and 15% of the market's lambda = 0 reference volume.
- Strategies: the 12 strategies of candidate_set (simrel/policies/library.py). The task text said 13; candidate_set
  holds 12, and the thirteenth of the reactive pilot was that family's randomised TWAP, which is not used here.
- Arms: section 4. The agent-mix markets run all five arms; the two latent markets run at their own parameters only.
- Seeds: truth 0..99 (twin and the 12 strategies on the same seed); viability twins 2000..2039; observability
  samples 66000..66019 (each arm) and 67000..67019 (lambda = 0 reference). Calibration used 20000..20191 and
  40000..40047 (passive episodes) and 50000..50299 (twin and TWAP). No calibration seed is an evaluation seed.
- No candidate simulator is fitted, no trust rule is applied, no held-out market is drawn.
- Experiments: E1 costs (true best strategy, always-TWAP regret plain and cross-fitted, always P6_signal_reversal
  against the best, each strategy against TWAP); E2 observability (addendum 10 diagnostic features on three-hour
  passive samples, each arm against lambda = 0, cross-validated AUC with a permutation reference); E3 viability at
  each arm; E4 the 120 s retention at each arm on the evaluation seeds.

## 7. The rule

> The lambda family proceeds to a held-out design if, on the pilot agent-mix markets, for the all-flow arm at the
> band_low level (lambda 0.6) or the max_feasible level (lambda 1.0), each judged separately, at least one holds:
> (a) in at least half of the non-excluded market and size cells, keeping the lambda = 0 best strategy costs at least
> 0.5 bp more, cross-fitted, than the level's best strategy; or
> (b) always-TWAP cross-fitted regret rises by at least 1.0 bp relative to lambda = 0, pooled over markets, averaged
> over sizes.
> The rule passes if either judged level meets (a) or (b).
> Otherwise the family stops, and the result is written up as: impact persistence does not change the decisions in
> these markets.
> Exclusion: a market is excluded and reported separately if the standard error of its drift-adjusted lambda = 0
> TWAP cost, scaled to 100 seeds, exceeds 1.0 bp at either size (section 8). It is computed from the lambda = 0
> calibration runs only, before the main run.
> No threshold, market set, arm, level or metric may change after this file is committed.

Readings, fixed now:

- Cost is implementation shortfall against the arrival mid (is_bps), in bp. The true best strategy of a cell is the
  strategy with the lowest mean cost over the 100 truth seeds.
- (a) A cell is a market and a size, for the markets that are not excluded. The lambda = 0 best strategy of a cell
  is its true best at lambda = 0 (all 100 seeds). At the judged level: the level's best strategy is chosen on seeds
  0..49 and the cost of the lambda = 0 best strategy minus the cost of that strategy is measured on seeds 50..99, at
  the judged level; then the reverse; the two are averaged. The cell counts as changed if this average is at least
  0.5 bp. "At least half" means the count of changed cells is at least half the number of cells. How often the
  best-strategy label itself differs from the lambda = 0 label is reported as description only.
- (b) Cross-fitted regret of a cell: the best strategy is chosen on seeds 0..49 and TWAP's cost minus its cost is
  measured on seeds 50..99, and the reverse; the two are averaged. It is averaged over the two sizes within a market
  and then over the markets that are not excluded. The rise is that mean at the judged level minus the same mean at
  lambda = 0.
- Point estimates decide. No significance is required. How often the best strategy of a cell changes when seeds are
  resampled is reported as description and is not part of the rule.
- all_dose and trader_band_low are reported with the same numbers and are not judged.
- The exclusion is decided in section 8, before the main run, and is not recomputed from the truth seeds. No market
  is excluded, so (a) is judged on 12 cells and (b) on six markets.

## 8. Exclusion

Drift-adjusted TWAP cost of a run: TWAP's implementation shortfall minus the unperturbed twin's mid change weighted
by the TWAP schedule, on the same seed. This is TWAP's fills measured against the twin's mid at each decision time
(impact_bps in simrel/metrics/cost.py). Data: the existing lambda = 0 calibration runs (twin and TWAP, seeds
50000..50299). The standard deviation over the 300 seeds is divided by 10 to give the standard error at 100 seeds. A
market is excluded if this exceeds 1.0 bp at either size.

| market | size | drift-adjusted cost (bp) | its s.e. at 100 seeds (bp) | raw cost s.e. at 100 seeds (bp) | excluded |
|---|---|---|---|---|---|
| a01 | 5% | 6.09 | 0.24 | 3.14 | no |
| a01 | 15% | 7.88 | 0.21 | 3.11 | no |
| a04 | 5% | 6.51 | 0.73 | 4.40 | no |
| a04 | 15% | 11.14 | 0.80 | 4.37 | no |
| a06 | 5% | 5.83 | 0.33 | 2.84 | no |
| a06 | 15% | 7.46 | 0.37 | 2.83 | no |
| a09 | 5% | 6.03 | 0.42 | 2.59 | no |
| a09 | 15% | 8.20 | 0.46 | 2.58 | no |
| a11 | 5% | 5.55 | 0.14 | 2.29 | no |
| a11 | 15% | 6.49 | 0.15 | 2.35 | no |
| a15 | 5% | 6.77 | 0.40 | 2.94 | no |
| a15 | 15% | 9.77 | 0.85 | 3.02 | no |

No market is excluded. Table: results/simrel/goal2/lambda_pilot/calib_exclusion.csv.

Why the earlier exclusion was replaced. The rule first proposed excluded a market whose lambda = 0 TWAP cost had a
standard error above 2 bp at 100 seeds. That measured absolute cost, which includes the market's own drift over the
10 minutes (the raw column above: 2.3 to 4.4 bp in all six markets, so all six would have been excluded). The drift
is the same for every strategy on a seed and cancels in every comparison the rule makes: (a) and (b) both use
differences between strategies on the same seeds. The collapse-prone markets that the exclusion was meant to catch
were already removed by the volatility-matching criterion. The replacement was made before any cost comparison,
regret or decision result existed; the numbers above are TWAP alone at lambda = 0 on calibration seeds.
