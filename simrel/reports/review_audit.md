# Response to review (weak reject): statistical audit, corrections, and an external-validity design

17 November 2026. Reproducible: `python simrel/experiments/audit_statistics.py` (outputs in results/simrel/goal2/audit/).
Re-analysis of existing results only; no new simulation.

## 1. Statistical validity

| Item | What the code does | Finding | Consequence for the paper |
|---|---|---|---|
| Cross-fitted regret | best strategy chosen on reference seeds 0-149; regret = cost on seeds 150-299 of the chosen strategy minus that of the half-sample best (goal2_addendum8.regret_of) | Negative values are expected: 188 of 576 Test B cells, minimum -22.15 bp, arise when a strategy beats the half-sample "best" on the other half. The offset (the half-sample best's cost) is identical for every arm in a market and size, so it cancels exactly in every between-arm difference | Report cross-fitted regret as a level only with this explanation; all hypotheses are differences and are unaffected |
| Full-truth regret | 300 seeds, minimum over the same means | Never negative. The between-arm differences are also unbiased and use twice the seeds | Report both. Estimates: H10.1 +1.00 (cross-fitted) and +0.81 (full); H10.2 +1.75 and +1.45; H10.3 -0.82 and -0.72; H10.4 -1.34 and -0.94 |
| Truth sampling noise | Test B bootstrap resampled markets and decision seeds; the truth was held fixed | SE added by the truth's own noise (paired per-seed differences): 0.14-0.23 bp, against 0.32-0.71 from markets and seeds. Combined intervals: H10.1 [+0.31, +1.69], H10.2 [+0.77, +2.73], H10.3 [-2.16, +0.51], H10.4 [-2.80, +0.12] | No conclusion changes. Report the combined intervals as the main ones |
| One-sided tests | p = share of resampled differences >= 0 | H10.1 and H10.2 were tested in the pre-registered direction (rule better) and failed with p = 1.00. "The rule is significantly worse than TWAP" is not a pre-registered test | State the worse-than-TWAP result as a two-sided interval (descriptive), not as a confirmatory finding |
| Holm | step-down over four hypotheses at family-wise 0.025: thresholds 0.00625, 0.0083, 0.0125, 0.025 | Implemented correctly | None |
| Market independence | Test B: 48 independent draws from fixed streams, 16 per family | Independent; the family strata are fixed by design, but the bootstrap was not stratified | Add a stratified (by family) bootstrap as a check |
| Development markets | 42 markets, 26 of them ablation variants of two base markets | Not independent; gate 3's intervals are optimistic (already stated) | Keep the 16-core-market results alongside; never present development intervals as inference |
| Earlier Goal 2 bootstrap | three levels: markets, designs shared across markets, seeds | Realism scores and the truth held fixed | State this; the cross-fitted decomposition was the truth-noise check |

## 2. Own-execution study

- Prior and posterior (per market, size and strategy p, on the cost difference to TWAP, d_p):
  prior d_p ~ N(m_p, tau^2), where m_p is the four-design ensemble's predicted difference (100 decision seeds) and tau^2
  is the ensemble's mean squared error in d over the other development markets of the same size, pooled over
  strategies. Own estimate d_hat_p = xbar_p - xbar_TWAP from n unpaired executions each, variance
  v_p = (s_p^2 + s_TWAP^2) / n (sample variances). Posterior mean (m_p / tau^2 + d_hat_p / v_p) / (1 / tau^2 + 1 / v_p);
  choose the lowest posterior mean, TWAP = 0. Strategies treated as independent.
- Leakage: own executions come from reference seeds 0-149, regret is evaluated on seeds 150-299 (disjoint). tau^2 never
  uses the market being scored (leave one out on development; fixed at the development value on Test A). tau^2 does use
  other markets' true costs, which a desk would not have: it is favourable to the hybrid, which still lost.
- Allocation: uniform, n per strategy, a different day per execution (unpaired); executions are drawn from the same
  stationary market (optimistic: real days differ).
- Uncertainty around "about 1,200 executions" (agent-mix, market bootstrap over markets): on development markets own
  data reaches TWAP's regret by n = 100 per strategy in only 58% of resamples; on Test A in 6%. At n = 100 own data minus
  TWAP is -0.09 [-0.98, +0.73] (development) and +0.73 [-0.15, +1.64] (Test A). The defensible statement is "not
  reliably below TWAP with up to 100 executions per strategy (1,200 in total)", not a threshold of 1,200.
- Non-simulator learning baselines (agent-mix; total budget N over the 12 strategies):
  successive halving reaches 2.49 bp at N = 1,200 and 2.06 at N = 1,800 on development markets (TWAP 2.88), but 4.20 and
  3.93 on Test A (TWAP 3.04); an A/B test of TWAP against the development-best strategy (switch if z > 2) never departs
  usefully (3.01-3.04 on Test A). A standard bandit allocation beats uniform allocation on development markets but not on
  Test A. Report these baselines in the paper.

## 3. Corrections to the narrative (manuscript)

- Identifiability. The grid shows mechanism distinguishability: a classifier separates markets that differ in mechanism
  from observable features (AUC). It does not show parameter identification of any candidate model (the corrected
  model's parameters are not identified, note 17), and it does not show counterfactual identification (that the cost of
  an unobserved strategy, or the response to one's own orders, is determined by observational data; Naviglio et al.
  2026 and Fujiwara et al. 2026 argue the opposite for impact). The paper will use the three terms separately and claim
  only distinguishability.
- Error decomposition. "Structural" (candidates minus true generator at the same budget) contains mechanism error,
  parameter estimation, calibration and model selection. Only the removed-mechanism experiment (addendum 9) isolates
  mechanism error, and only the well-specified markets isolate parameter estimation. The paper will call the remainder
  "model error" and give the decomposition as an upper bound on the mechanism share.
- Novelty. Genuinely new empirical evidence: regret of execution advice across simulator families with known costs;
  realism scores and LOB-Impact scores failing to predict that regret under pre-registration; a removed-mechanism
  experiment in which more simulation increases regret; pre-registered failure of observable trust rules on untouched
  markets; the own-execution comparison. Demonstrations of established principles: the optimizer's curse (Smith and
  Winkler 2006); that observational fit does not imply interventional fidelity (LOB-Impact; Naviglio et al. 2026); that
  decision regret is the right yardstick (CrashGym); that a model need only be accurate where the policy goes (Dong et
  al. 2023; Dann et al. 2026). The paper will say so explicitly.

## 4. External-validity sensitivity experiment (design for review; not run)

Problem. The synthetic families' price response to trades is far more persistent than Nasdaq's and their impact is
close to linear in size. Feasibility check on the 179 existing markets with features and truth: median R(100)/R(1) 7-38
by family against 1.9 on Nasdaq; TWAP-impact exponent from the 5% and 15% sizes about 1.0 against 0.33-0.43; only 1
market lies in the box R(100)/R(1) in [1.0, 3.5] and exponent in [0.25, 0.65].

Question. Do the main findings, that fitted-simulator selection is worse than TWAP and that the excess is model error,
persist when the synthetic markets' impact persistence and concavity are brought into the empirical ranges?

Stage 0, calibration (cheap, local or one notebook). Draw candidate parameter sets from widened ranges of the latent and
agent-mix families (latent: shorter relaxation tau, longer sign runs; agent-mix: weaker fundamental pull, stronger
replenishment). For each draw run a probe (TWAP at 5% and 15%, 20 seeds, plus one 3-hour observation sample) and keep a
draw if R(100)/R(1) is in [1.0, 3.5], the exponent in [0.25, 0.65], H in [0.45, 0.55] and viability holds. The screen
uses only observable statistics and TWAP impact, never strategy costs or regret. Draw in a fixed order from a fixed
stream until 12 are accepted (6 per family) or 400 draws are used.

Stage 1, evaluation (about 6-8 hours of Kaggle with 5 notebooks). For the accepted markets: truth (300 seeds, 12
strategies, 2 sizes); the four fixed designs and the corrected model (fitted to six episodes, 100 decision seeds); the
true generator (100 seeds); always-TWAP; the no-simulator rule frozen from the development markets.

Hypotheses (one-sided, market bootstrap, Holm over E1 and E2 at 0.05; E3 descriptive):
- E1: choosing by the four-design ensemble has higher regret than always-TWAP.
- E2: the true generator has lower regret than always-TWAP by at least 0.5 bp (the decision is not flat).
- E3: candidates minus true generator (model error) is positive.

Stopping criteria, fixed in advance:
- Fewer than 8 accepted markets after 400 draws: stop; report that the synthetic families cannot reach the empirical
  ranges, and state the limitation instead of the sensitivity result.
- E2 fails (the true generator gains less than 0.5 bp over TWAP): the main finding is uninformative in this regime;
  report that execution decisions are nearly flat when impact is concave and transient, which bounds what any
  simulator can gain or lose there.
- E1 and E3 hold with E2: the main finding persists in the empirical regime.
- E1 fails with E2 holding: the main finding is regime-dependent; report it as such and restrict the paper's claims to
  persistent-impact regimes.

## 5. Manuscript revision (after the experiment decision)

Proportionate claims per sections 1-3; methods section with the full protocol; figures (budget curves, removed
mechanism, Test B by family, own-execution curves with intervals); Table 1 rebuilt to fit the column (two tables:
regrets and hypotheses); complete references (verify Joshi et al. 2024; add Naviglio, Fujiwara, Chen-Horst-Tran, Dann
et al., already verified); reproducibility section with repository, seeds, checksums and the data source of the ITCH
sample files; no TODOs.

## 6. Stage 0 implementation (16 November 2026, 18:21 EST, before any draw was evaluated)

- Draws: simrel/reference_worlds/extvalid.py (stream 20261117; 200 draws per family, latent then agent-mix; ranges
  listed there with the original ranges).
- Probe per draw (simrel/experiments/extvalid_screen.py): one 3-hour observation sample (seed 63000) giving R(100)/R(1)
  and H with the study's feature code; the reference volume taken as the mean traded volume per 10 minutes in that
  sample; TWAP at 5% and 15% of it, 20 seeds each (seeds 90000-90019), with the unperturbed twin; impact = the twin-paired
  displacement at the end of execution (disp_end_bps, as in the earlier feasibility check); exponent = log(I15 / I5) /
  log 3; viability = the study's rule applied to the 20 twins (failed share and share of twins moving more than 300 bp).
- Acceptance: R(100)/R(1) in [1.0, 3.5], exponent in [0.25, 0.65], H in [0.45, 0.55], viable. The first 6 accepted
  draws per family in draw order are the markets. No strategy cost other than TWAP's impact is computed.
