# Research Execution Plan: Decision Reliability Across ABIDES, MarS, and Hawkes Market Simulators

**Status:** Execution-ready research specification, with mandatory feasibility gates  
**Date:** 2026-10-10  
**Audience:** Claude Code / research engineering team

## 1. Core research question

**Do market simulators that reproduce realistic limit-order-book (LOB) statistics also produce reliable execution-cost predictions and robust execution-policy recommendations?**

Separate three claims rigorously:

1. **Observational realism:** How closely does each simulator reproduce held-out market data?
2. **Cross-simulator robustness:** Does a policy optimized in one simulator remain competitive in the other simulators?
3. **Controlled decision validity:** When the data-generating market is known, does a simulator recommend policies that perform well in that reference environment?

**Never equate agreement among simulators with real-world correctness.** A synthetic reference market supplies ground truth only within that controlled experiment. Real-world counterfactual optimal policies generally cannot be recovered from historical LOB data.

## 2. Model shortlist

| Family | Candidate | Purpose | Feasibility requirement |
|---|---|---|---|
| Agent-based | **ABIDES** | Behavioral-agent baseline; agent interaction and order matching | Reproduce stable episodes; inspect exact fork/version and market-impact mechanism |
| Modern generative | **Microsoft MarS** | Learned order-flow simulator, ideally with interactive TWAP intervention | Verify official released checkpoint, download terms, market-impact example, GPU needs, and ability to insert external orders |
| Statistical | **Multivariate marked Hawkes LOB simulator** | Interpretable event-intensity baseline with excitation, order sizes, cancellations, and state-dependent execution | Must implement matching engine, queue evolution, and endogenous response to injected orders; Hawkes event generator alone is insufficient |

**MarS starting point:** https://github.com/microsoft/MarS  
**ABIDES starting point:** https://github.com/jpmorganchase/abides-jpmc-public  
**LOB-Bench:** https://github.com/peernagy/lob_bench

Treat checkpoint availability and compatibility as *claims to verify in the current repository*, not assumptions. Prefer a small official MarS checkpoint and reproduce the official market-impact/TWAP example first. MarS pretrained market domain may differ from the US equity data used for ABIDES and Hawkes; this is a potentially disqualifying comparability problem unless addressed.

### Model-selection gates

- **Gate A — Runnable:** pinned commit, environment lock, reproducible seeded episode.
- **Gate B — Interactive:** external parent-order schedule can submit child orders; fills and subsequent market state react.
- **Gate C — Comparable:** same market conventions, tick size, lot size, fees, time horizon, order types, session conditions, and approximately matched liquidity regime.
- **Gate D — Stable:** no unexplained spread blow-ups, broken queues, non-fills, or pathological drift across seeds.
- **Gate E — Reproducible:** available checkpoint/data rights and computational budget.

If MarS fails, report the failure and consider a documented interactive generative fallback (e.g., a correctly reproduced TRADES configuration). **Do not silently substitute a noninteractive sequence generator.**

## 3. Data and calibration

1. Select one instrument/venue/session regime that can be represented across all models; do **not** automatically assume MarS's pretrained weights generalize to INTC/NASDAQ.
2. Secure legal access to event-level LOB messages, trades, and order book snapshots. Record provenance, exchange, dates, and licensing.
3. Chronologically split **train / validation / held-out test** days; prevent look-ahead.
4. Standardize timestamp precision, tick size, order-size units, book depth, trading hours, and treatment of hidden liquidity.
5. Calibrate each model using the same training regime **where feasible**. If calibration domains differ, label results *domain-shift exploratory*, not a clean head-to-head comparison.
6. Keep real held-out test data separate from the synthetic reference-market experiment.

## 4. Execution tasks and policies

**Canonical task:** sell a parent order over a fixed horizon, e.g. 50,000 shares over 30 minutes; test multiple sizes as percentages of average daily volume or contemporaneous displayed depth. Do not start with 500,000 shares until liquidity and model stability support it.

Include at least these policies:

| ID | Policy | Fixed or adaptive? | Definition |
|---|---|---|---|
| P0 | **TWAP** | Fixed | Equal target volume per time bucket |
| P1 | **Immediate execution** | Fixed | Marketable orders at start, with controlled participation/impact limits |
| P2 | **Front-loaded schedule** | Fixed | Execute more early, then taper |
| P3 | **Back-loaded schedule** | Fixed | Execute less early, then accelerate |
| P4 | **POV / participation rate** | Adaptive | Execute a specified fraction of observed market volume, with terminal completion safeguard |
| P5 | **Liquidity-aware** | Adaptive | Trade more when spread is tight and depth is ample; less when adverse |
| P6 | **Short-horizon signal-aware** | Adaptive | Adjust child-order size based on *past-only* order imbalance/returns and remaining inventory |
| P7 | **Risk-aware feedback controller** | Adaptive | Balance expected impact, volatility risk, and inventory/time-to-deadline |
| P8 (optional) | **MFG-derived execution policy** | Adaptive/model-derived | Calibrate MFG assumptions independently; benchmark only after stability checks |

Implement every policy through one common interface:

`action = policy(observation_t, remaining_inventory, time_remaining, state)`

Observations must contain only information available at decision time. Define market versus limit-order choices and unfilled-order handling consistently. Every policy must respect inventory, risk, participation, and completion constraints. Any forced terminal liquidation cost must be included in the reported result.

**Distinguish fixed policy evaluation from model-specific optimization.** Fixed TWAP is the *same rule* in all simulators; adaptive policies can take different actions because observed state differs.

## 5. Experiment I — Observational realism (held-out historical data)

Use LOB-Bench metrics where compatible; report both unconditional and conditional statistics:

- Spread, depth, volume, order-size distribution, event-type mix, inter-arrival times.
- Returns, volatility, imbalance, order-flow autocorrelation.
- Conditional spread/depth by regime, time of day, volatility, and order imbalance.
- Event-conditioned mid-price response and replenishment after orders, when supported.

Use bootstrap confidence intervals across days and episodes. Do not conflate a realism metric with a causal estimate of intervention response. Document metrics that are not directly comparable due to market domain.

## 6. Experiment II — Interventions and cross-simulator transfer

For each simulator, seed, initial state, parent-order size, and policy:

1. Run **no-intervention baseline**.
2. Run **with-policy intervention**, using matched initial conditions and random seeds where possible.
3. Estimate changes in mid-price, spread, depth, replenishment, realized fills, volatility, and execution cost.
4. Repeat across seeds and regimes; compare distributions rather than individual paths.

For model-specific policy optimization, optimize/tune on simulator-specific training episodes, then **freeze the policy** before testing in every other simulator.

Define a cross-simulator cost matrix:

`C[i, j] = mean cost of policy optimized in simulator i, evaluated in simulator j`

For a candidate policy set P, define within-world regret:

`Regret(i -> j) = C[i,j] - min_{p in P} mean_cost(p evaluated in j)`

Report mean, confidence interval, policy rank correlation, worst-case regret, and sensitivity to simulator and market regime. This is **relative simulated regret**, not real-market regret.

## 7. Experiment III — Synthetic reference-market decision validity

### Why
Historical order-book data generally cannot tell us what would have happened if a different large execution strategy had been used. Therefore simulator disagreement alone does not identify the correct policy.

### Reference-market protocol
1. Implement or select a **separate, independently specified market-generating process** with a transparent matching engine and adjustable price impact, replenishment, order clustering, and liquidity regimes.
2. Create **multiple** reference-world mechanisms (e.g., queue-reactive, regime-switching agent mix, and nonlinear impact); avoid simply declaring Hawkes or ABIDES to be universal truth.
3. Generate training/validation observations from each reference world; fit or adapt the three candidate simulators *without exposing the hidden reference rules*.
4. Enumerate or optimize a clearly bounded candidate policy class **in the reference world** with many seeds to estimate the best achievable policy within that class.
5. Independently obtain each fitted simulator's recommended policy.
6. Execute each recommendation **back in the reference world**, using held-out seeds and matched conditions.

Define:

`Reference regret(model) = E_ref[cost(recommended_policy_model)] - min_{p in P} E_ref[cost(p)]`

Evaluate whether observational realism predicts lower reference regret **across reference worlds, regimes, seeds, and models**, with uncertainty intervals. Three models alone do not justify broad correlation claims; many settings and careful dependence-aware inference are needed.

**Important:** The reference-world optimum is only optimal among tested policies (or within the specified optimization class), not globally optimal. Synthetic reference results do **not** establish real-world superiority.

## 8. Cost and evaluation definitions

For a sell order, report signed implementation shortfall with a consistent convention, e.g.:

`cost_bps = 10000 * (arrival_midprice - realized_volume_weighted_sell_price) / arrival_midprice`

Include fees, spread crossing, and terminal inventory penalties consistently. Clearly distinguish implementation shortfall from market-impact estimate, VWAP deviation, fill rate, and execution completion time.

Report:
- Mean / median / p90 / p95 execution cost in bps.
- Fill rate, remaining inventory, terminal liquidation cost.
- Intervention-induced mid-price response versus baseline.
- Policy rank stability and cross-simulator regret.
- Reference-world decision regret.
- Runtime, memory, GPU hours, and failed episode rate.
- Bootstrap confidence intervals, seed sensitivity, and calibration sensitivity.

## 9. Confounders and safeguards

- **Market-domain mismatch:** Chinese-equity-pretrained MarS vs US-equity ABIDES/Hawkes can invalidate a direct comparison.
- **Model instability:** prior pilot found ABIDES configurations with instability and a retrained TRADES setup with spread widening. Treat as engineering failures/limitations, not proof of structural model differences.
- **Different matching engines:** order execution accounting must be harmonized or reported as a confounder.
- **Noninteractive generators:** observationally realistic samples do not imply causal response to external orders.
- **Calibration fairness:** identical budgets and data access where possible; report unavoidable asymmetries.
- **Policy overfitting:** optimize on training seeds, evaluate on unseen seeds/regimes.
- **Synthetic reference bias:** test multiple unrelated reference mechanisms and disclose their assumptions.
- **Statistical inference:** paired comparisons where feasible, uncertainty intervals, multiple testing caution.
- **No real-world oracle:** never label synthetic regret as actual market optimality.

## 10. Claude Code execution sequence (do this in order)

### Phase 0 — Repository and feasibility audit (do not skip)
- Inspect official repositories, latest commits, licenses, pretrained model/checkpoint links, Docker/CUDA requirements, data requirements, and intervention APIs.
- Produce `feasibility_matrix.md` with verified links and pass/fail status.
- Estimate GPU-hours, storage, and runtime for each model.
- **Stop and report blockers** if MarS checkpoint, data rights, or interactive market impact cannot be demonstrated.

### Phase 1 — Minimal runnable pilots
- ABIDES: stable, seeded 10–30 minute episode; inspect spread and fills.
- MarS: run official smallest-checkpoint example, then its market-impact/TWAP demonstration.
- Hawkes: implement or select marked multivariate event process + order-book matching engine, test stability and queue evolution.
- Validate a tiny sell order changes fills and state in each simulator.
- Save pinned commits, commands, logs, plots, and seed configs.

### Phase 2 — Common evaluation harness
Create:
- `configs/` (instrument, regimes, execution horizons, order sizes, seeds)
- `adapters/` (ABIDES, MarS, Hawkes)
- `policies/` (fixed + adaptive policies)
- `metrics/` (cost, realism, impact, regret)
- `reference_worlds/` (multiple controlled markets)
- `experiments/` (realism, transfer, controlled validity)
- `reports/` (tables, charts, limitations)

Use a shared event/action schema, inventory ledger, and cost accounting. Unit-test policy causality and all sign conventions.

### Phase 3 — Pilot experiment
- Start with TWAP, immediate, POV, liquidity-aware, and risk-aware controller.
- At least 2 order sizes, 2 market regimes, and enough seeds to estimate variance (not a fixed token seed count).
- Produce intervention/no-intervention plots and cross-simulator cost matrix.
- Audit failures before scaling.

### Phase 4 — Controlled reference experiment
- Build at least 2 genuinely different reference mechanisms.
- Determine best-in-class policy via held-out reference evaluations.
- Evaluate each model's frozen recommended policy in the reference worlds.
- Report reference regret with confidence intervals.

### Phase 5 — Scale and write
- Expand policies, regimes, seeds, and instruments only after pilots pass.
- Generate reproducible tables and figures with scripts.
- Write paper with explicit separation of observational realism, robustness, and controlled validity.
- Maintain a `limitations.md` that documents all unverified claims and domain mismatches.

## 11. Acceptance criteria / stop rules

**Proceed with the three-way study only if:**
- All three models support comparable external order injection and execution accounting.
- The same or credibly harmonized market regime is available.
- Episodes remain stable across seeds.
- The pilot fits compute budget.

**Otherwise:** report a scoped ABIDES-vs-Hawkes study and treat MarS as a separate-domain case study, or replace it with a verified interactive generative model. Do not present an invalid apples-to-oranges comparison.

## 12. Expected research outputs

1. `feasibility_matrix.md` with evidence-backed status.
2. Reproducible environments, pinned commits, and dataset manifest.
3. Common execution-policy interface and policy tests.
4. Realism scorecard and conditional response plots.
5. Cross-simulator cost and regret matrices.
6. Reference-market policy regret results.
7. Ablations: order size, regime, calibration, checkpoint, and seed.
8. Draft paper with claims limited to demonstrated evidence.

## 13. Immediate instruction to Claude Code

> **Start with Phase 0 and Phase 1 only.** Verify MarS checkpoints and reproduce its TWAP/market-impact example before writing a large framework. Audit whether ABIDES, MarS, and Hawkes can share a comparable instrument and market regime. Show exact runnable commands, logs, and a feasibility matrix. If a model cannot be run interactively or domains cannot be matched, stop and propose an evidence-backed alternative. Once feasibility is confirmed, implement the common harness and execute the pilot, then the controlled reference-market experiment. Do not fabricate results, checkpoints, data availability, or model comparability.
