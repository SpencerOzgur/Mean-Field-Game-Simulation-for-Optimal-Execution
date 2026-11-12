# Novelty audit (stage 0 of design addendum 10)

12 November 2026. What recent work already shows, what it does not, and where this study differs. Based on the papers'
abstracts and introductions; entries marked "to check" need the full text before the manuscript cites them in detail.

## 1. Closest work

| work | what it shows | what it does not address | how this study differs |
|---|---|---|---|
| CrashGym (Lu, Cheng, Chen, Zhu; arXiv 2610.29245, 21 Oct 2026; apparently also NeurIPS 2026 as "MarketScientistBench") | Replayable synthetic LOB worlds with four amplification channels (persistent impact, liquidity withdrawal, margin liquidation, evolving strategy mix); an LLM agent picks a deleveraging or backstop intervention; regret against the best counterfactual intervention on an independently seeded target; laboratory access cut mean regret 8.88 to 3.79 on 24 prospective worlds (one deployment complete) | Misspecified simulators: its laboratory is a copy of the true world, i.e. our oracle; optimal execution; choosing between simulators; realism or validation measures | Our candidates are fitted, misspecified simulators; we measure whether realism and validation identify those that give good advice, and decompose regret into sampling and structural parts. Addendum 9: more simulation of a wrong model makes advice more confidently wrong, which a true-world laboratory cannot show. Shared methodology (mechanism worlds, seeded targets, counterfactual regret) is acknowledged as the closest precedent |
| Market Impact Estimation via Causal Foundation Models (Thumm, Chen; arXiv 2610.12537, 8 Oct 2026) | A continuous-time causal PFN trained on synthetic markets with confounding information flow; transfer to crypto perpetuals and 3,800 Nasdaq stocks; conformal recalibration; a recurrent variant tracks impact build-up (corr. with metaorder impact 0.68 to 0.92-0.96) | Whether better impact estimates change execution decisions or regret | We measure decision value directly. Its estimator is a candidate "impact-model correction" for stage 3 if code is available (to check) |
| LOB-Impact (Nigmatulin et al.; arXiv 2610.27100, 19 Oct 2026) | Benchmark of generators' response to injected metaorders: build-up, peak, relaxation | Whether the scores identify simulators that give good execution advice | Used as our counterfactual measure; C1/C2 test its decision value (suggestive, not significant after adjustment) |
| TRADES (Berti et al.; arXiv 2502.07071, ECAI 2025) | Diffusion LOB generator coupled to an interactive exchange; notes the need for better strategy-evaluation protocols | Decision reliability of the simulations | Included in Goal 1 as an INTC case study only (one trained model; a second training seed failed to learn); not in Goal 2 (cost; see addendum 10, section 8) |
| Compound Hawkes LOB model (Jain, Firoozye, Kochems, Treleaven; arXiv 2312.08927 v5, Aug 2024; submitted to Finance Research Letters, publication not confirmed) | Hawkes events with sampled sizes (round-number clustering), non-negative spreads, time-of-day intensities, concave impact | Execution decisions | Becomes the modern Hawkes baseline (addendum 10): a corrected model must beat it, not only our six-event variants |
| DiffLOB (Wang, Ventre; IJCAI 2026, pp. 6564-6572; arXiv 2602.03776) | Regime-conditioned diffusion (trend, volatility, liquidity, imbalance); counterfactual regime trajectories improve regime prediction | Interventions by a trader (own-order impact); execution decisions | Regime counterfactuals are not the effect of our own order; not a baseline for the execution task |
| EvoMarket (Zhong et al.; arXiv 2604.18046, 2026) | Scalable simulator with in-run self-calibration that adds corrective order flow to track recorded data | Decisions; whether tracking the record implies correct counterfactual response | Its correction targets replay fidelity; ours targets decision-critical mechanisms |

## 2. Claims this study can still make, and claims it should not

Can make (with the existing evidence):

1. Similarly realistic simulators recommend opposite execution strategies on real data (Goal 1, H1 on three stocks).
2. No observational realism score predicts good execution advice across markets with known answers (C3, replicated in a
   held-out family).
3. Regret of fitted simulators is mostly structural where mechanisms are missing; more seeds do not repair it, and removing
   one mechanism from the true generator reproduces the effect (addendum 9).
4. Simulator-based selection did not beat always-TWAP in any market set; the headroom over TWAP is small (0.4 bp in the
   held-out family, 1.7 bp in agent-mix), which bounds what any decision method can gain.

Should not make:

- A new impact-realism benchmark (LOB-Impact exists).
- A new generic or self-correcting simulator (TRADES, DiffLOB, EvoMarket).
- That experimentation in a market improves decisions in general (CrashGym shows this for a true-world laboratory).
- That conservative gating is a solution: it works by never leaving TWAP.

## 3. Open contribution (addendum 10)

When a simulator's execution advice can be trusted, using evidence available in real markets, and whether correcting
decision-critical mechanisms makes the advice better than TWAP on markets never used in development.
