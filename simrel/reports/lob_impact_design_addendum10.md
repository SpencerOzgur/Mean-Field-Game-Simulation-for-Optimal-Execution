# Decision reliability and mechanism correction (design addendum 10)

Written 12 November 2026, after all results of addenda 1-9 and before any of the analyses or runs below. Prompted by
review: test when a simulator's execution advice can be trusted, using evidence available in real markets, and whether
correcting decision-critical mechanisms makes the advice better than always-TWAP on markets never used in development.
Related work: simrel/reports/novelty_audit.md. No new simulator architecture is built under this addendum beyond the
two specified in section 6.

## 1. Questions

Q1. Can a trust rule using evidence available in real markets choose when to depart from TWAP so that regret falls below
always-TWAP and below a rule that uses no simulator, on markets not used in development?
Q2. Can different combinations of mechanisms produce observable data that cannot be told apart at our data size, while
implying different best strategies (decision-relevant non-identifiability)?
Q3. Does correcting a decision-critical mechanism (fundamental anchoring, liquidity withdrawal) reduce regret beyond
recalibration and beyond a modern Hawkes baseline, and does any realism gain translate into decision gain?

## 2. Market sets and leakage control

| set | markets | role |
|---|---|---|
| Development | 10 designed (R1a-R1e, R2a-R2e), 6 well-specified (W1-W6), 30 mechanism-ablation (X*) | all rule and model development; all tuning |
| Test A (secondary) | 10 drawn (R1s01-05, R2s01-05), 6 held-out latent (R3s01-06) | read once, after the rules are frozen; secondary evidence only |
| Test B-U (primary) | 48 new markets, unfiltered (section 2.1) | the confirmatory test |
| Test B-F (secondary) | 16 new markets, opportunity-rich (section 2.2) | stress test: can a method capture available gain? |

Test A has already influenced our thinking (its results are known in aggregate; addendum 5 tested earlier gated rules
on the drawn markets), so it cannot be confirmatory. Its result does not change any frozen rule.

Test B is generated from the ranges and seeds fixed here. Its runs may execute on Kaggle before stage 4, but until the
stage-4 freeze file is checksummed only a completeness check is run on its outputs (file counts and failed-run flags);
no cost, regret or diagnostic is computed or viewed. The Test B analysis script is written and checksummed before it is
run. Ranges, sizes, seeds, the viability rule and the evaluation criteria of Test B do not change after this addendum.

### 2.1 Test B-U: unfiltered

16 markets from each of three families, drawn with the ranges already in the code: queue-reactive and agent-mix
(simrel/reference_worlds/sampled.py, QR_RANGES and AM_RANGES), latent liquidity (simrel/reference_worlds/latent.py,
RANGES). Draw stream: numpy default_rng(20261112), families in the order queue-reactive, agent-mix, latent; within a
family, draws in order, the first 16 that pass the existing viability rule (run_reference.py) are kept; rejected draws
are listed with their reason. Names TBq01-16, TBa01-16, TBl01-16. Standard task (10 minutes, 20 decisions), sizes 5%
and 15% of reference volume. Truth: 300 seeds, cross-fitted (best identified on seeds 0-149, evaluated on 150-299).
Candidates: six observable 30-minute episodes, 100 decision seeds.

### 2.2 Test B-F: opportunity-rich

8 agent-mix and 8 latent markets, stream default_rng(20261113), same ranges. A draw is kept if it is viable and the
true generator at 100 seeds beats always-TWAP by at least 1.5 bp in regret averaged over the two sizes (truth from 300
seeds). The screen uses only the truth and the true generator, never a candidate or a rule. At most 300 draws per
family; the acceptance rate is reported. Results on B-F are reported as conditional on available gain and never as
evidence that a method beats TWAP in general.

### 2.3 Number of markets

The unit of replication is the market. Per-market paired differences are averaged over the two sizes. From the 100-seed
results, the true generator's gain over TWAP has between-market s.d. 0.2-1.1 bp within the latent and agent-mix families
and 2.1 bp pooled outside queue-reactive; its mean over an equal mix of the three families is about 0.65 bp (agent-mix
1.68, latent 0.39, queue-reactive -0.13). This is the ceiling for any method on B-U.

Minimum meaningful improvement over always-TWAP on B-U: 0.5 bp averaged over the family mix (about three quarters of
the ceiling; smaller averages are within the noise of realised costs). With an assumed s.d. of 1.2 bp for a rule's
per-market gain, one-sided alpha 0.025 and power 0.8, n = 7.84 x (1.2 / 0.5)^2 = 46, so 48 markets. If the observed
s.d. on Test A is larger than 1.6 bp, this is reported and the power of the B-U test is restated; n is not changed.

## 3. Evidence tiers

Every trust rule is labelled by the evidence it needs.

| tier | evidence | available in real markets | use |
|---|---|---|---|
| 1 observational | message and quote data of the market (the six episodes) | yes; computed for INTC, CSCO, MSFT | primary rules |
| 2 own-execution | realised costs of the trader's own past executions in this market | only with proprietary execution logs | secondary; flagged as such |
| 3 synthetic intervention | probe orders sent into the reference market | no | upper bound on the value of experimentation; never evidence about real markets |

Real data: only tier-1 diagnostics are applied to Nasdaq. They show applicability and which Goal 1 simulators pass and
whether those that pass agree with each other. No claim of real-market decision accuracy is made.

## 4. Trust rules (Q1)

Design set D (available in every market set): h_k1, h_c_f_m_k1, abides, qr; plus compound Hawkes and the corrected
models of section 6 once built. Every rule outputs either a strategy or "TWAP (abstain)". Within a market, statistics
that decide whether to trust use decision seeds 0-49; the recommended strategy uses seeds 50-99.

| rule | tier | departs from TWAP when |
|---|---|---|
| always TWAP | - | never |
| no-simulator | - | never uses a simulator: the strategy with the lowest mean regret on the development markets for this order size |
| gated (addendum 5) | 1 | the ensemble's estimated gain over TWAP minus k standard errors is positive |
| agreement | 1 | at least a share q of the designs in D recommend a strategy other than TWAP with the same sign of gain, and the ensemble's gain exceeds g |
| sensitivity | 1 | the recommendation is unchanged across the three fitted variants (caps 0.8, 0.9, 0.95) of each Hawkes design and the gain exceeds g |
| diagnostic | 1 | at least one design passes the mechanism diagnostics (below); the trusted designs' ensemble then decides, with gain above g |
| own-execution | 2 | the designs whose predicted TWAP cost lies within the 90% interval of the realised mean cost of six past TWAP executions (separate seeds in the reference market) are trusted; their ensemble decides |
| probe | 3 | the designs that reproduce the reference market's response to six small probe orders (1% of volume, alternating sides; mid-price move at 1, 5 and 30 s) within the 90% interval are trusted |

Mechanism diagnostics (tier 1), computed on the six episodes and on each design's simulated windows: the features of
the mechanism-validation table (H, sign-memory gamma, R(100)/R(1), 25-trade block autocorrelation), plus variance
ratios of mid-price changes at 30, 120 and 600 s (mean reversion from anchoring) and the response of best-level depth
to the preceding 30 s of signed flow (withdrawal). A design passes if every feature of the market lies in the design's
90% band for a six-episode sample (from 50 simulated six-episode samples of the design).

Selection and freezing: on the development markets each rule's parameters (k, q, g) are chosen over pre-stated grids
(k in {0, 1, 2, 3}; q in {0.5, 0.75, 1}; g in {0, 0.5, 1, 2} bp) by mean regret. One tier-1 rule is carried to Test B:
the one with the lowest development regret among those that depart from TWAP in at least 25% of market-size cells
(ties: fewer parameters). Values and the chosen rule go into results/simrel/goal2/addendum10/rules10_frozen.json,
checksummed before Test A is read.

Reported for every rule: regret; share of the TWAP-to-oracle gap captured; harmful-decision rate (regret above
always-TWAP's in the same cell); abstention rate.

Gate 1 (after Test A): the chosen tier-1 rule has lower regret than always-TWAP and than the no-simulator rule on Test A
with 95% market-bootstrap intervals excluding zero. If it fails, Q1 is answered negatively on Test A and the rule is
still carried to Test B unchanged; no new rule is developed after Test A. Stage 3 proceeds regardless.

## 5. Identifiability (Q2)

Grid: agent-mix around the R2a defaults, 3^4 = 81 markets: fund in {0.025, 0.1, 0.4} (anchoring), trend in
{0.0125, 0.05, 0.2} (trend following), withdraw in {0.5, 2, 6} (withdrawal), limit in {0.7, 1.4, 2.8} (liquidity
replenishment). Order size fixed in shares as in the ablation markets. Per market: truth at 200 seeds; 20 independent
six-episode observable samples, and 20 twenty-four-episode samples.

Observational equivalence: for each pair of markets, a cross-validated logistic classifier on the standardised
diagnostic vector of section 4 (one vector per sample). A pair is equivalent at a data size if the AUC is below 0.65.
Decision-relevant: the pair's true best strategies differ and using one market's best in the other costs at least 1 bp.
Wrong-mechanism test: the corrected model of section 6 is fitted to every grid market; report the correlation of
estimated with true anchoring and the share of markets where the fit attributes the mean reversion to the wrong
mechanism (estimated anchoring in the wrong tercile while trend or replenishment is off-base).

Pre-stated outcomes, by the share of decision-relevant pairs that are equivalent at six episodes:
I1 below 10%: correction from observables is feasible. I2 10-50%: partially identifiable. I3 above 50%: correcting the
mechanism from observable data of this size is ill-posed; reported as a principal finding. The 24-episode result shows
whether more data resolves it.

## 6. Mechanism correction (Q3)

Models, on the development markets:

| version | change |
|---|---|
| original | h_k1 and h_c_f_m_k1 as in the study |
| recalibrated | the same designs fitted to 24 episodes instead of six |
| compound Hawkes | Jain et al. (arXiv 2312.08927): event sizes sampled from the fitted empirical distributions, spread-dependent intensities; time-of-day terms omitted (30-minute windows) |
| corrected | h_c_f_m_k1 plus a latent fundamental value (random walk) and fundamentalist market orders with intensity f x mispricing^p, and depth withdrawal proportional to recent signed flow; f, p, the walk's volatility and the withdrawal gain estimated from the six episodes by simulated method of moments on the diagnostics of section 4 |
| corrected + rule | the corrected model inside the frozen tier-1 rule |

Each new model is first checked for parameter recovery on data simulated from itself (bias and spread of estimates from
six episodes). Compared against always-TWAP, the no-simulator rule and the true generator.

Gate 3 (development markets): the corrected model's regret is below both the original and compound Hawkes with 95%
market-bootstrap intervals excluding zero. If not, no further model development; the corrected and compound Hawkes
models are still run on Test B as specified (H10.3, H10.4).

## 7. Realism against decision value

For original, recalibrated, compound Hawkes and corrected models: realism (LOB-Bench order-book score; distance to the
market on the mechanism-validation features) and regret. Pre-stated outcomes for each model against original:
N1 realism and regret both improve; N2 realism improves, regret does not (interval includes zero or worse): reported as
"realism without decision value"; N3 neither improves; N4 regret improves without realism gain. Realism improves means
the mechanism-feature distance falls by at least 25% and the LOB-Bench score does not worsen. All four outcomes are
reported whichever occurs.

## 8. Baselines not included

TRADES is not trained per synthetic market (about 10 GPU hours per market). If gate 3 passes, TRADES is added on a
representative subset of six development markets (two per family) as a robustness comparison, specified in a further
addendum before training. A method paper claiming superiority over generative simulators would need it.

## 9. Confirmatory hypotheses on Test B-U

All one-sided, per-market paired differences averaged over sizes, inference by market bootstrap (2,000 resamples, decision
seeds resampled within markets), Holm adjustment over the four at family-wise 0.025:

H10.1 the frozen tier-1 rule has lower regret than always-TWAP.
H10.2 the frozen tier-1 rule has lower regret than the no-simulator rule.
H10.3 the corrected model (plain recommendation, no rule) has lower regret than original h_c_f_m_k1.
H10.4 the corrected model has lower regret than compound Hawkes.

A method-paper claim requires H10.1 and H10.2 together with a departure rate of at least 25%. Secondary, unadjusted:
per family, Test B-F, harmful-decision and abstention rates, tier-2 and tier-3 rules, the longer task (six B-U markets
per family under the 30-minute task, if compute allows; decided before stage 4).

## 10. Stopping rules and outputs

- Gate 1 and gate 3 as above; neither stops the B-U evaluation of what was frozen.
- No rule or model is changed after Test A is read, except fixes of defects, which are listed with their effect.
- If H10.1 fails, the paper is the evaluation paper; the stage 1-3 results are reported as negative evidence, including
  outcomes I3 and N2 if they occur.
- Outputs: results/simrel/goal2/addendum10/ (development, Test A), results/simrel/goal2/testB/ (Test B).
