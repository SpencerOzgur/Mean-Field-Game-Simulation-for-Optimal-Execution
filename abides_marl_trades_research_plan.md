# Research proposal: Does a realistic market simulator lead to reliable execution decisions?
**Discussion draft — 8 October 2026**  
**Candidate systems:** ABIDES, ABIDES-MARL, TRADES/DeepMarket  
**Status:** Experimental design; no results have been generated.

## Executive summary

**Core question:** If three simulated financial markets appear plausible, do they recommend the *same way to execute a trade*? If they disagree, how much money could a trader lose by trusting one simulated market rather than another?

We propose a **decision-focused, controlled comparison** of ABIDES, ABIDES-MARL, and TRADES. We will first measure conventional realism (spread, volatility, order flow, liquidity, impact). We will then measure **policy disagreement**, **cross-simulator policy transfer**, and **economic regret**. We will test whether ordinary realism metrics predict decision reliability.

**Crucial caveat:** The three systems are not three interchangeable pretrained models. ABIDES is an agent-based simulation platform; ABIDES-MARL adds strategically learning participants; TRADES is a learned order-flow generator deployed in an ABIDES-based environment. Their different agent compositions, calibration, training, and experimental interfaces can confound a naive comparison. The paper should compare clearly defined *configurations*, not make universal claims about entire frameworks.

**Proposed paper claim to test, not assume:** “Market simulators with similar observational realism can produce materially different optimal execution policies, and conventional realism metrics may fail to predict policy transfer.”

## 1. Accessible explanation for non-finance researchers

Imagine you must sell 10,000 shares over 30 minutes. Selling everything immediately is fast, but can push the price down. Selling slowly may reduce immediate price impact, but exposes you to adverse price movements and the risk of not finishing. This is called **optimal execution**.

A **limit order book (LOB)** lists outstanding offers to buy (bids) and sell (asks), with quantities at each price. The **spread** is the difference between the best ask and best bid. **Liquidity** is how easily orders can trade without moving the price. **Slippage** is the difference between an intended/reference price and realized execution price.

A **market simulator** invents the orders and reactions of other participants. We ask: if we train or choose a selling strategy in simulator A, will it still work in simulator B? Agreement is reassuring; disagreement reveals **model risk**, not automatically which simulator is correct.

### The three candidates

| System | What drives the market? | Important strength | Key limitation for fair comparison |
|---|---|---|---|
| ABIDES | Explicit programmed agents exchanging messages with a matching exchange | Interpretable agent behavior; flexible controlled interventions | Results depend strongly on chosen agents and parameters |
| ABIDES-MARL | Multiple strategic/adaptive agents learning in a market simulation | Can represent responses to strategic traders and execution agents | Learning dynamics and agent objectives introduce additional variables |
| TRADES/DeepMarket | Transformer-based diffusion generator of conditional order flow, interacting with an ABIDES-based exchange | Data-driven synthetic flow; released code and checkpoints | Trained on particular stocks/dates; checkpoint/data compatibility and feedback must be verified |

**Do not write:** “ABIDES is trained on NASDAQ.” ABIDES is not intrinsically a pretrained model; its message design follows NASDAQ ITCH/OUCH-style protocols. TRADES's public instructions mention TSLA and INTC checkpoints and LOBSTER-format historical data.

## 2. Literature and novelty assessment

### What existing papers already establish

| Work | Established contribution | Overlap with proposed study | Remaining question |
|---|---|---|---|
| ABIDES (2019) | High-fidelity agent-based interactive market simulation | Provides baseline and exchange environment | Do its execution policies transfer to other simulators? |
| TRADES (2025) | Generative LOB order flow; realism, responsiveness, predictive-score comparisons; ABIDES integration | **Already compares generative market simulation against alternatives**, and evaluates intervention responsiveness | Does apparent realism imply agreement on optimal execution actions? |
| ABIDES-MARL (2025) | Multi-agent learning, endogenous price formation and execution | **Already studies strategic execution and agent responses** | How does the learned strategic environment compare with a generative-flow environment under a matched task? |
| LOB-Bench (2025) | Evaluation methods for generative LOB models | Important realism benchmark | Do benchmark scores track execution-policy regret? |
| M3 (2026) | Learned market-microstructure model with interactive execution experiments | Closely related alternative learned world model | Could be external validation, but data/market mismatch increases scope |

**Novelty status, as of this targeted October 2026 search:** I did **not identify a publication explicitly benchmarking the ABIDES + ABIDES-MARL + TRADES trio under a common execution-policy transfer and regret protocol**. This is **not proof that none exists**. Existing TRADES and ABIDES-MARL papers already cover substantial components of the problem. Before submission, repeat a systematic search across Google Scholar, Semantic Scholar, arXiv, SSRN, OpenAlex, and citations to the three primary papers, including preprints and workshops. Record queries, dates, screening decisions, and overlapping methods. **Do not claim “first ever” until that audit is complete.**

### Search log and follow-up queries

| Query | Status | Notes / next action |
|---|---|---|
| `ABIDES MARL TRADES comparison execution simulator policy evaluation` | Initial targeted web search performed | No exact three-way policy-transfer benchmark surfaced |
| `TRADES DeepMarket ABIDES diffusion interactive` | Initial targeted web search performed | TRADES directly extends ABIDES; not an independent matching engine |
| `ABIDES-MARL execution price formation` | Primary paper reviewed at abstract level | Execution optimization already studied within ABIDES-MARL |
| `"policy transfer" "limit order book" simulator` | **Pending systematic search** | Look for decision-aware simulator evaluation |
| `"simulator ranking" execution "market impact"` | **Pending systematic search** | Look for rank reversal and sim-to-real research |
| Citation graph of ABIDES-MARL, TRADES, LOB-Bench | **Pending** | Screen citing and cited papers |

## 3. Research questions and hypotheses

**RQ1 — Observational realism:** Which configurations reproduce real-market features most faithfully?

**RQ2 — Decision disagreement:** Given the same starting state, inventory, time horizon, and objective, do the simulators favor different execution actions?

**RQ3 — Transfer and regret:** How much does a policy optimized in one simulator deteriorate when evaluated in another?

**RQ4 — Predictiveness of realism:** Do lower spread/volatility/order-flow errors correlate with better cross-simulator policy transfer?

**RQ5 — Robustness:** Does disagreement change under high volatility, low liquidity, large orders, or adaptive opponents?

Hypotheses to **test**, not assume:
- H1: At least one pair of simulator configurations exhibits meaningful policy ranking reversals.
- H2: Observational realism alone does not fully explain cross-simulator regret.
- H3: Agent adaptation changes measured execution impact, especially for larger participation rates.

## 4. Experimental principle: isolate the cause of differences

Two tracks are required.

**Track A — Native-system comparison (ecological validity).** Run each system as intended, with documented native agent population and calibration. This answers: *What happens if a researcher uses each off-the-shelf approach?* Differences cannot be attributed solely to world-model architecture.

**Track B — Harmonized controlled comparison (causal clarity).** Match exchange rules, stock, date/regime, tick size, starting LOB, observation interface, external execution agent, reward, fees, order size, latency, session length, and randomization as far as each system permits. This answers: *How much disagreement remains after controlling identifiable confounds?*

A strict swap of only “world model” may be impossible for ABIDES-MARL because endogenous learning changes the participant population. If so, describe Track B as **partially harmonized**, publish a compatibility matrix, and avoid a causal “architecture alone” conclusion.

### Compatibility / feasibility gate — fill before main experiments

| Requirement | ABIDES | ABIDES-MARL | TRADES | Decision / blocker |
|---|---|---|---|---|
| Reproducible environment + commit hash | TBD | TBD | TBD | |
| Public runnable code | Verify | Verify | Verified repo | |
| Pretrained policy / checkpoint accessible | N/A | TBD | README reports TSLA/INTC checkpoints; verify download | |
| Run without licensed LOBSTER data | Yes for synthetic configs | TBD | README provides fallback example; quality caveat | |
| Same stock, session, tick/lot units | TBD | TBD | TBD | |
| Inject external market order | TBD | TBD | TBD | |
| Inject/cancel limit order | TBD | TBD | TBD | |
| Reactions to agent's orders influence later states | TBD | TBD | Paper reports responsiveness; verify in code | |
| Expose bid/ask/volume and fills | TBD | TBD | TBD | |
| Deterministic/repeatable seed | TBD | TBD | TBD | |
| Snapshot or standardized warm start | TBD | TBD | TBD | |
| Cost per 1,000 decision steps | TBD | TBD | TBD | |

**Go/no-go:** Do not begin policy training until each system can run a **10-episode smoke test** with an identical simple TWAP agent, log fills, and respond to order injections. If TRADES's checkpoint link, historical input, or external-agent integration fails, first resolve the dependency or narrow claims; do not quietly substitute a noninteractive synthetic CSV replay.

## 5. What types of agents should we include?

Separate **market-background agents** (the other traders generating the environment) from **evaluation agents** (the execution strategies being compared). This prevents confusing the simulator's built-in participants with the policy being evaluated.

### Background-agent decision table

| Background type | Plain-English role | Include? | ABIDES implementation | ABIDES-MARL implementation | TRADES counterpart |
|---|---|---|---|---|---|
| Noise trader | Buys/sells without a strong signal | **Core** | TBD | TBD | Implicit in learned flow; not directly identifiable |
| Liquidity provider / market maker | Posts buy and sell quotes | **Core** | TBD | TBD | Implicit in generated order flow |
| Fundamental/value trader | Reacts to estimated fair value | Sensitivity | TBD | TBD | No guaranteed one-to-one counterpart |
| Momentum trader | Follows short-term trends | Sensitivity | TBD | TBD | No guaranteed one-to-one counterpart |
| Adaptive strategic market maker | Changes quotes in response to competitors | **Core for MARL track** | Optional engineered adaptation | TBD | Responsiveness test, not necessarily a distinct agent |
| Informed trader | Trades on a signal unavailable to others | Stress test | TBD | TBD | May not be controllable independently |
| Experimental execution trader | Our fixed test agent, selling inventory | **Required everywhere** | External agent | External agent | Injected experimental agent |

**Selection rule:** Begin with the smallest configuration capable of producing an active two-sided book and realistic fills. Add strategic adaptation as a controlled factor, not as an untracked difference. Do not claim the same named agent type exists inside TRADES: its learned order flow may encode behavior without explicit agent identities.

### Evaluation policies: keep identical across environments

| Policy | Definition | Why include? | Hyperparameters |
|---|---|---|---|
| Immediate market sell | Sell entire inventory now | Fast but potentially expensive extreme | None |
| TWAP | Split inventory evenly over time | Widely understood baseline | Number of slices |
| POV | Trade a fixed fraction of market volume | Adapts to market activity | Participation rate |
| Simple urgency rule | Sell faster as deadline approaches or liquidity improves | Transparent state-dependent baseline | Urgency thresholds |
| Trained RL policy | Learns from simulator observations | Tests decision fidelity and overfitting | Same architecture/training budget |
| Robust policy (optional) | Optimizes across multiple simulators | Tests practical mitigation | Robustness objective |

**Main recommendation:** Start with immediate, TWAP, POV, and urgency rule; add one common RL algorithm only after the environment interfaces work. The goal is to compare world models, **not** run an algorithm tournament.

## 6. Primary task, variables, and controls

**Task:** Sell a fixed initial inventory \(Q_0\) within \(T\) minutes, with no shorting. Start with a simple single-asset scenario.

**Pilot values (illustrative, not calibrated):**
- \(T = 30\) minutes.
- \(Q_0\) selected as 0.1%, 0.5%, and 1.0% of estimated interval market volume.
- Decision interval: 30 seconds or event-based equivalent, fixed across systems.
- Action: market-sell a fraction of remaining inventory; later add passive limit orders.
- Observation: best bid/ask, spread, top-level depth, imbalance, recent returns/volume, remaining inventory, time remaining.
- Reward: realized sale proceeds relative to an arrival-price benchmark, minus fees and unfilled-inventory penalty.
- Evaluation: many independently seeded episodes across multiple market regimes; confidence intervals at the episode/day level.

**Important:** Avoid comparing a 30-minute task with a simulator that only supports a shorter meaningful calibrated horizon. Align timestamps, market-hours boundaries, and order-size units. A synthetic “same initial LOB” does not ensure identical latent conditions.

### Configuration registry

| Variable | Proposed primary setting | Alternatives | Why it matters |
|---|---|---|---|
| Instrument | INTC if all three support credible configuration | TSLA, additional stock | TRADES published checkpoints include INTC/TSLA |
| Data/calibration window | Matched as far as permitted; disclose mismatches | Different regimes | Domain shift changes behavior |
| Exchange priority | Price-time priority | Other only in sensitivity | Determines fills |
| Tick size | Instrument-consistent | ± sensitivity | Changes spreads |
| Inventory | Small / medium / large fraction of volume | Wider range | Impact is size-dependent |
| Horizon | 30 minutes | 5 / 60 minutes | Alters urgency |
| Fees | Common simplified schedule | Zero-fee sensitivity | Alters policy ranking |
| Latency | Common fixed latency | Heterogeneous | Affects passive fills |
| Market regime | Normal / volatile / illiquid | Open vs midday | Tests robustness |
| Random seeds | Paired where possible | Independent repeated draws | Variance control |

## 7. Metrics: what exactly will we learn?

### 7.1 Observational realism (against held-out real data)

- Best-bid/ask spread distribution and mean absolute error.
- Top-of-book depth distribution.
- Return volatility and autocorrelation.
- Order-flow imbalance distribution and autocorrelation.
- Trade size and interarrival-time distributions.
- Market impact / price response after injected orders.
- Tail behavior and intraday seasonality.
- LOB-Bench metrics where applicable.

**Critical:** TRADES may be evaluated on different historical data from ABIDES calibration. Without a common held-out real dataset, report observational realism as *within-domain*, not a fair winner ranking.

### 7.2 Execution outcomes

For a sell program, define **implementation shortfall per share** as:

\[
IS = P_{\mathrm{arrival}} - \frac{\sum_j q_j p_j}{\sum_j q_j}
\]

for a **fully filled** order, plus separately reported fees. Positive IS means the average sale price was below arrival price. For incomplete fills, do **not** use the above formula alone: use total proceeds and a clearly specified terminal mark-to-market or penalty for remaining inventory. Report all prices in **basis points (bps)** relative to arrival price, with fill rate and time-to-completion separately.

Additional metrics: execution cost, fill fraction, tail cost (e.g. 95th percentile), realized impact, inventory left at deadline, and variance of proceeds.

### 7.3 Decision disagreement

For each matched observation \(s\), evaluate a **common** set of candidate actions \(a\). Let \(Q_m(s,a)\) denote estimated *negative future execution cost* in simulator \(m\). Then:

\[
a_m^*(s)=\arg\max_a Q_m(s,a).
\]

Measure:
- **Action disagreement rate:** fraction of matched states for which \(a_i^*(s)\ne a_j^*(s)\).
- **Action-distance:** difference in chosen order size/urgency, not just exact equality.
- **Rank correlation:** Spearman correlation of action or policy rankings.
- **Margin-weighted disagreement:** emphasize states where competing actions have meaningfully different estimated costs.

**Technical caution:** A state sampled from one simulator may be out-of-distribution in another. First evaluate matched *initial* states and externally constructed benchmark states; then report separate on-policy trajectory comparisons. Never assume internal latent states are identical.

### 7.4 Cross-simulator transfer and regret

Train policy \(\pi_i\) in simulator \(i\), then evaluate it in simulator \(j\).

\[
C_{ij}=\mathbb{E}[\mathrm{execution\ cost\ of}\ \pi_i\ \mathrm{in}\ j].
\]

Define transfer regret in target \(j\):

\[
R_{ij}=C_{ij}-C_{jj},
\]

where \(C_{jj}\) is the cost of a policy optimized in \(j\), **with matched training budget**. A positive number indicates a penalty from training elsewhere. Because stochastic training can yield noisy \(C_{jj}\), report uncertainty; occasional negative observed regret does not invalidate the framework.

**Interpretation:** This matrix measures *simulator-to-simulator transfer*, not real-world regret. To claim decision accuracy, add independent historical/offline impact validation or controlled live-market evidence, which may be difficult because historical replay lacks causal counterfactual reactions.

## 8. Blank result templates (populate only after experiments)

### Table A — Feasibility and calibration

| System | Code commit | Calibration source/date | Exchange mechanics | Agent configuration | Checkpoint | Reproducible? |
|---|---|---|---|---|---|---|
| ABIDES | — | — | — | — | N/A | — |
| ABIDES-MARL | — | — | — | — | — | — |
| TRADES | — | — | — | World agent | — | — |

### Table B — Realism (lower errors are better)

| Simulator | Spread error | Depth error | Volatility error | Order-flow error | Impact error | Overall standardized score |
|---|---:|---:|---:|---:|---:|---:|
| ABIDES | — | — | — | — | — | — |
| ABIDES-MARL | — | — | — | — | — | — |
| TRADES | — | — | — | — | — | — |

### Table C — Policy cost by simulator (bps; lower is better)

| Policy | ABIDES | ABIDES-MARL | TRADES |
|---|---:|---:|---:|
| Immediate sale | — | — | — |
| TWAP | — | — | — |
| POV | — | — | — |
| Urgency rule | — | — | — |
| RL trained in ABIDES | — | — | — |
| RL trained in ABIDES-MARL | — | — | — |
| RL trained in TRADES | — | — | — |

### Table D — Transfer matrix \(C_{ij}\), bps (lower is better)

| Train \ Evaluate | ABIDES | ABIDES-MARL | TRADES |
|---|---:|---:|---:|
| ABIDES | — | — | — |
| ABIDES-MARL | — | — | — |
| TRADES | — | — | — |

### Table E — Transfer regret \(R_{ij}\), bps (relative to target-trained policy)

| Train \ Evaluate | ABIDES | ABIDES-MARL | TRADES |
|---|---:|---:|---:|
| ABIDES | 0 by definition | — | — |
| ABIDES-MARL | — | 0 by definition | — |
| TRADES | — | — | 0 by definition |

### Table F — Pairwise decision disagreement

| Pair | Matched states (N) | Different best action (%) | Mean action-size difference | Policy rank correlation | 95% CI |
|---|---:|---:|---:|---:|---|
| ABIDES vs MARL | — | — | — | — | — |
| ABIDES vs TRADES | — | — | — | — | — |
| MARL vs TRADES | — | — | — | — | — |

### Table G — Stress scenarios

| Scenario | Order size | Volatility | Liquidity | Biggest policy ranking reversal | Worst transfer regret | Notes |
|---|---|---|---|---|---|---|
| Normal | — | — | — | — | — | |
| High volatility | — | — | — | — | — | |
| Thin book | — | — | — | — | — | |
| Large execution | — | — | — | — | — | |
| Adaptive opponents | — | — | — | — | — | |

### Table H — Ablations

| Experiment | Held fixed | Changed | Intended inference | Outcome |
|---|---|---|---|---|
| Same external TWAP | Task + order sizes | Simulator | Basic execution-response differences | — |
| Fixed vs adaptive makers | Task + other agents | Adaptation | Contribution of strategic response | — |
| Small vs large orders | Everything else | Participation | Nonlinear impact | — |
| Matched vs native config | Task | Calibration / background | Confounding magnitude | — |
| No fees vs fees | Policies + market | Fee schedule | Cost-ranking sensitivity | — |

## 9. Statistical methodology

1. **Pre-register** primary metrics: transfer regret and policy rank reversal; treat all other metrics as secondary.
2. Run **pilot experiments** to estimate variability, then set episode counts using a power or confidence-interval-width calculation. Do not invent a fixed sufficient N.
3. Report means, medians, quantiles, and **bootstrap confidence intervals**, resampling by independent day/episode rather than individual highly correlated ticks.
4. Use multiple RL seeds and equal training budgets; evaluate on held-out seeds and, if feasible, held-out dates.
5. Report effect sizes in bps and relative to baseline TWAP, not only p-values.
6. Correct for multiple comparisons or specify a small primary hypothesis family.
7. Track wall-clock runtime, hardware, failed runs, checkpoint hashes, and data provenance.

## 10. Practical implementation architecture

```text
                    COMMON EXPERIMENT CONTROLLER
         scenario, starting book, inventory, horizon, seed
                                 |
             +-------------------+------------------+
             |                   |                  |
       ABIDES adapter      MARL adapter       TRADES adapter
             |                   |                  |
             +-------------------+------------------+
                                 |
                 STANDARDIZED OBSERVATION
        bid, ask, depth, imbalance, time, remaining shares
                                 |
                 COMMON EXECUTION POLICY
                market/limit/cancel actions
                                 |
                   STANDARDIZED LOGGING
          orders, fills, costs, states, seed, metadata
                                 |
               REALISM + DISAGREEMENT + TRANSFER
```

Suggested adapter contract:

```python
class MarketAdapter:
    def reset(self, scenario: dict, seed: int) -> dict: ...
    def step(self, action: dict) -> tuple[dict, float, bool, dict]: ...
    def submit_order(self, side: str, quantity: int, order_type: str,
                     limit_price: float | None = None) -> str: ...
    def cancel_order(self, order_id: str) -> None: ...
    def get_book(self, levels: int = 10) -> dict: ...
    def get_fills(self) -> list[dict]: ...
    def metadata(self) -> dict: ...
```

Do **not** force this API into all systems before checking what each simulator exposes. Write thin adapters and explicitly document unsupported actions. Treat agent learning updates during evaluation as a separate experimental condition: freeze opponents for the controlled baseline, allow online adaptation in a designated MARL sensitivity study.

## 11. Timeline, owners, and milestones

| Phase | Duration estimate | Owner | Deliverable | Exit criterion |
|---|---|---|---|---|
| 0. Novelty audit | 3–5 days | All | Search log + related-work matrix | No unaddressed directly overlapping paper |
| 1. Feasibility | 1–2 weeks | One owner/system | Runnable configs + 10 TWAP episodes | All three produce fills and reactive market responses |
| 2. Standardization | 1–2 weeks | Integration lead | Adapters, common metrics, tests | Unit and cost definitions match |
| 3. Baselines | 1 week | Evaluation lead | TWAP/POV/urgency tables | Stable estimates and confidence intervals |
| 4. Policy transfer | 2–3 weeks | RL lead | 3×3 cost/regret matrices | Comparable training/evaluation budgets |
| 5. Stress tests | 1–2 weeks | All | Ablations and failure cases | Main claim survives confound checks |
| 6. Paper | 2 weeks | All | Draft + reproducibility package | Limitations and novelty claims defensible |

These are **planning estimates**, not commitments.

## 12. What could invalidate the project?

- **No common task interface:** If TRADES or MARL cannot accept the same interventions, a three-way execution comparison is not defensible.
- **Non-comparable calibration:** If each simulator uses a different stock/date/market regime, apparent differences may be dataset artifacts.
- **MARL not a distinct simulator:** ABIDES-MARL shares ABIDES foundations; its distinct contribution is adaptive strategic participants, not necessarily a different exchange.
- **TRADES shares ABIDES exchange:** TRADES is not a fully independent exchange engine; the paper must not portray the three as independent market infrastructures.
- **Synthetic-to-synthetic only:** Transfer regret is not ground-truth real-world regret.
- **Weak counterfactual realism:** Matching historical prices does not prove realistic reactions to a large new trade.
- **Training instability:** Different RL policies may arise from stochastic optimization, not different world models.
- **Policy rank stability:** If all three agree, this is still a publishable negative result only if the protocol is rigorous and the contribution is more than a small benchmark.
- **Prior-art overlap:** An earlier study may already investigate decision-aware simulation fidelity under different terminology.

## 13. Concrete next actions for the three researchers

**ABIDES owner:** Pin a repository commit; document the exact RMSC/agent configuration; implement external TWAP orders, fills, and LOB logging; list all tunable calibration parameters.

**ABIDES-MARL owner:** Pin implementation and dependencies; identify which agents learn, whether their policies can be frozen, and whether external execution orders can be injected; document the difference from ABIDES at the exchange, agent, and objective levels.

**TRADES owner:** Clone DeepMarket; validate the checkpoint download and INTC/TSLA sample run; verify that the experimental agent's orders actually influence subsequent generated flow; document whether licensed LOBSTER data are required for the intended comparison.

**Joint decision meeting:** Review the feasibility matrix and approve one common instrument, horizon, action space, fee model, and primary outcome. Then run a synchronized 10-episode TWAP pilot before committing to RL training.

## 14. Primary sources and reproducibility links

1. ABIDES paper: https://arxiv.org/abs/1904.12066  
   Repository: https://github.com/abides-sim/abides  
   RMSC03 configuration: https://github.com/abides-sim/abides/wiki/RMSC03
2. ABIDES-MARL paper: https://arxiv.org/abs/2511.02016
3. TRADES paper: https://arxiv.org/abs/2502.07071  
   Published article: https://journals.sagepub.com/doi/10.3233/FAIA251249  
   Official DeepMarket repository: https://github.com/LeonardoBerti00/DeepMarket
4. LOB-Bench: https://proceedings.mlr.press/v267/nagy25a.html  
   Repository: https://github.com/peernagy/lob_bench
5. M3 (possible later sensitivity study): https://arxiv.org/abs/2608.19227  
   Repository: https://github.com/ArthurZhang02/m3-market-microstructure

### Meeting decision checklist

- [ ] Is our contribution **decision-aware simulator validation**, not merely a three-model leaderboard?
- [ ] Can all three simulators execute the **same inventory liquidation task**?
- [ ] Can we identify and disclose every major difference in background agents and training data?
- [ ] Is the primary claim about **cross-simulator disagreement**, not unobserved real-world correctness?
- [ ] Can we obtain a common held-out market dataset for at least part of the realism assessment?
- [ ] Do we have enough compute to train/evaluate one common policy family with multiple seeds?
- [ ] Has a full citation-forward/backward novelty audit been assigned and completed?

---

# October 8, 2026 — Feasibility and novelty validation addendum

**Evidence level:** Targeted online literature and repository-documentation audit. **Not** a systematic exhaustive bibliographic review and **not** a locally executed installation or benchmark. Statements marked **UNVERIFIED** require team smoke tests. Do not represent this document as proof that the experiment works or that no prior paper exists.

## Updated recommendation

**Conditional GO for a 1–2-week feasibility pilot; NO-GO for full three-system RL training until gates pass.** The scientifically sharper research question is:

> **Does conventional market-simulator realism predict the reliability of execution decisions, as measured by policy ranking stability and cross-simulator transfer regret?**

A mere ABIDES-versus-ABIDES-MARL-versus-TRADES leaderboard is not sufficient novelty. Our contribution must be a controlled *decision-aware validation protocol*, with matched task definitions, pre-registered metrics, and analysis connecting realism to downstream decision quality. This is a **hypothesis**; if realism predicts reliability strongly, that is an informative falsification.

## Findings from direct source inspection

| System | Confirmed from public documentation | Still unverified | Feasibility assessment |
|---|---|---|---|
| ABIDES | Open repository, configurable exchange and agents; RMSC03 explicitly supports optional POV execution agent and market-impact experiments | Exact modern runtime compatibility with teammates' chosen branch, performance, common-state reset | **High for a pilot** |
| ABIDES-MARL | November 2025 paper describes ABIDES-Gym extension, synchronized multi-agent decisions, standard RL compatibility, and a liquidity-trader execution problem | **Official accessible implementation, pretrained agents, freeze-vs-retrain protocol, ability to inject common external policy** | **Medium/unknown: primary blocker** |
| TRADES/DeepMarket | Public MIT-licensed repository; model checkpoints for TSLA and INTC advertised; simulation command uses an ABIDES extension; published paper tests responsiveness to experimental agent | Checkpoint downloads, exact interactive action API, dependency installation, usable comparable stock/date calibration, speed | **Medium; runnable demo documented, fair evaluation not proven** |

### Specific new feasibility findings

1. **ABIDES is not trained on NASDAQ.** Its message-based simulator follows NASDAQ ITCH/OUCH-style protocols; its agent behavior is configured. The RMSC03 documentation specifies 5,000 noise agents, a POV market maker, and an optional POV execution agent. Source: https://github.com/abides-sim/abides and https://github.com/abides-sim/abides/wiki/RMSC03
2. **TRADES is not an entirely independent exchange.** DeepMarket embeds a learned world agent in an ABIDES-derived environment. Thus the comparison is *hand-programmed background flow vs strategically learned agents vs learned generative background flow*, not three independent exchanges. Source: https://github.com/LeonardoBerti00/DeepMarket
3. **TRADES data access is a substantive issue.** Its README advertises TSLA and INTC checkpoints and a demo without user-provided LOBSTER data, but warns that the demo uses data different from training. It says reproducing the paper requires specific TSLA/INTC historical data (29–30 January 2015), and training a new model requires LOBSTER-format data. Synthetic TRADES-LOB CSVs are useful for offline analysis but **do not themselves demonstrate interactive policy evaluation**. Source: https://github.com/LeonardoBerti00/DeepMarket
4. **ABIDES-MARL already includes execution research.** Its paper studies a liquidity trader in an endogenously evolving market with informed trader, noise traders, and competing market makers. Simply showing adaptation changes execution is prior art. Source: https://arxiv.org/abs/2511.02016
5. **TRADES already tests market realism, usefulness, responsiveness, and compares to prior methods.** The paper itself says additional work is needed for mature trading-strategy evaluation. Therefore our distinction must be *policy-transfer regret and whether realism metrics predict decision reliability*. Source: https://journals.sagepub.com/doi/10.3233/FAIA251249
6. **LOB-Bench already has market-impact/price-response statistics.** A new metric that merely measures impact is not novel. Source: https://proceedings.mlr.press/v267/nagy25a.html
7. **A further 2026 competitor exists:** *Bridging the Reality Gap in Limit Order Book Simulation* explicitly develops interactive simulation for execution costs, impact, and price reversion. Include it in related work and consider it as a later independent structural holdout. Source: https://arxiv.org/abs/2603.24137
8. **PulseCheck (2026) adds practical competition** in distributional, event-response and stylized-fact simulator validation; it is not the same as policy-transfer evaluation, but weakens any claim that a realism-validation framework alone is new. Source: https://research.simudyne.com/blog/pulse-check/

## Novelty verdict and exact claim boundaries

**What appears established:** realistic LOB simulation, generative market backgrounds, adaptive strategic participants, injected execution agents, market-impact evaluation, standardized LOB realism benchmarks, and synthetic-data usefulness tests.

**What was NOT located in this targeted search:** an exact published, controlled three-way ABIDES / ABIDES-MARL / TRADES evaluation that explicitly tests **whether realism scores predict cross-simulator execution-policy ranking stability and regret**.

**This is not proof of absence.** An exhaustive review would require full-text database searches, forward/backward citation screening, related workshops, working papers, and replication code. The team's literature owner must perform that before using words such as “first” or “novel” in a submission.

**Defensible provisional contribution:** (1) a reproducible common interactive execution interface, (2) a pre-registered decision reliability/transfer benchmark, (3) a test of the realism–reliability relationship across enough independent simulator *configurations*, and (4) controlled diagnostics for disagreements. **Not** merely testing which simulator has the best P&L.

## Essential design correction: three simulators are not three statistical observations

To test whether realism predicts decision reliability, three model-level points are insufficient. Create multiple pre-specified configurations per system (different liquidity, agent mix, generator checkpoint if available, order size, regime) and evaluate on held-out scenarios. Treat observations within each simulator family as correlated. If a common held-out real-data reference cannot be obtained, report *within-domain realism* and explicitly refrain from declaring a real-world winner.

### Primary analysis table (new)

| Simulator | Configuration ID | Calibration data | Realism error (pre-registered) | Policy rank reversal (%) | Cross-simulator regret (bps) | Confidence interval | Held-out? |
|---|---|---|---:|---:|---:|---|---|
| ABIDES | A1 | TBD | — | — | — | — | — |
| ABIDES | A2 | TBD | — | — | — | — | — |
| ABIDES-MARL | M1 | TBD | — | — | — | — | — |
| ABIDES-MARL | M2 | TBD | — | — | — | — | — |
| TRADES | T1 | TBD | — | — | — | — | — |
| TRADES | T2 | TBD | — | — | — | — | — |

**Primary figure:** Realism error (horizontal) versus transfer regret or rank reversal (vertical), labeled by simulator family, with uncertainty intervals. Analyze separately by regime/order size; do not treat every correlated time step as independent evidence.

## Pilot acceptance gates (must be demonstrated in code)

| Gate | Required evidence | Pass condition | Current status |
|---|---|---|---|
| G1: Runnable | Exact commit, environment, command, log | All 3 produce 10 completed reproducible episodes | **UNVERIFIED** |
| G2: Interactivity | Inject buy/sell and compare no-intervention replay | Injected action changes matching/fills and downstream simulation as documented | **UNVERIFIED** |
| G3: Common policy | One identical TWAP liquidation agent in each | Same shares, horizon, action semantics, fees, and accounting | **UNVERIFIED** |
| G4: Comparable calibration | Instrument/date/market regime provenance | Differences matched or explicitly controlled | **UNVERIFIED; TRADES data concern** |
| G5: MARL policy control | Frozen opponents and optional online adaptation | Clearly separate learning during training vs evaluation | **UNVERIFIED** |
| G6: Reliable statistics | Multi-seed pilot, independent episodes | Cost and completion metrics have stable definitions and estimated uncertainty | **UNVERIFIED** |
| G7: Novelty | Citation matrix and searches across arXiv/SSRN/Scholar/OpenAlex | No directly equivalent full study overlooked | **PARTIAL ONLY** |

**Stop rule:** If G2–G4 fail, do not train RL or claim a three-way controlled comparison. Either repair compatibility or narrow to a well-controlled two-system study with a different independent holdout.

## Adversarial policy testing — optional, not the primary claim

Once a common interface works, let an execution policy optimize in each simulator and check for unusual repeated behaviors (e.g. precisely timed orders exploiting unrealistic book replenishment). **Do not call a strategy an “exploit” merely because it earns more synthetic P&L**; it requires evidence that behavior fails under other simulators or independent empirical constraints. Adversarial diagnostics are a **secondary mechanism study**, not a replacement for the realism-to-reliability primary test.

## Work allocation for the three researchers

- **ABIDES colleague:** produce pinned environment, RMSC03 agent inventory, injected TWAP, and full fill/cost log.
- **ABIDES-MARL colleague:** locate and pin actual code, confirm freeze/retrain controls and external-agent support, produce equivalent TWAP logs; report whether learning changes during evaluation.
- **TRADES owner:** validate checkpoint access and a documented INTC/TSLA demo; inspect DeepMarket's ABIDES integration, show that a new order changes subsequent world-agent flow, and report what real historical data are required.
- **Joint:** approve an experiment specification and run 10 matched TWAP episodes before writing any RL training code.

## Final decision

**Recommendation: conditional feasibility GO, novelty not certified.** This is a reasonable **research investigation**, not yet a validated publication gap. Advance to full research only if the three-way controlled pilot works and a documented citation audit rules out direct prior art. If the main hypothesis is falsified (realism strongly predicts reliable execution), that is still an interesting result provided the design and sample size are credible.
