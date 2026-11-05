# Results of design addendum 6 (5 November 2026)

## 1. End-to-end oracle: FAILED the pre-registered positive control

The exact generator of every reference market, with its true parameters, run through the unchanged candidate pipeline
(goal2_oracle_ablation.py oracle; results/simrel/goal2/addendum6/oracle.csv):

| family | always TWAP | oracle, 30 seeds | oracle, 100 seeds | median rank correlation 30 / 100 | candidates (mean of 29) |
|---|---|---|---|---|---|
| queue-reactive | 0.41 | 0.61 | 0.54 | 0.28 / 0.28 | 0.67 |
| agent-mix | 3.58 | 2.91 | 1.90 | 0.81 / 0.87 | 8.44 |
| Hawkes-built | 4.11 | 3.03 | 1.22 | 0.61 / 0.87 | 4.34 |

Criterion (30 seeds: regret at most 0.5 bps and a quarter of TWAP's in every family; median rank correlation at least
0.8): failed.

Diagnosis (exploratory):
- The pipeline is unbiased. Oracle and truth cost estimates agree: z-scores of the per-strategy differences have mean
  -0.05 and standard deviation 0.87 over 624 strategy-cells, 3% beyond 2 (5% expected). One Hawkes-built market (W1)
  shows a consistent offset (mean z -1.6); the cause has not been traced.
- The decisions are under-powered. In the queue-reactive markets a median of 7 of 11 alternatives cannot be told
  apart from the true best even with the truth's 300 seeds (median gap 0.46 bps; standard error with 30 seeds about
  1 bp): the decision is essentially flat and no simulator can reliably beat TWAP. In the agent-mix and Hawkes-built
  markets the best strategy is distinct (median gap about 5.7 bps), but per-seed noise (19-30 bps) gives standard
  errors of 3.5-5.5 bps at 30 seeds, so sampling noise alone produces regret of the order of TWAP's.
- Share of candidate regret beyond the oracle's (misspecification): queue-reactive 8%, Hawkes-built 30%, agent-mix 66%.
- In the Hawkes-built markets the generating design refitted from data does about as well as the oracle: parameter
  estimation is not the limiting factor.

Consequence for the Goal 2 conclusions: "simulators lose to TWAP" mixes two causes. Where the decision is flat
(queue-reactive) or under-powered (30 seeds), sampling noise and the optimizer's curse on that noise explain most of
it; misspecification dominates only in the agent-mix markets. The conservative rules of addendum 5 mostly protect
against noise.

## 2. Mechanism ablation

Order size fixed in shares; 200 seeds per market (goal2_oracle_ablation.py ablation; ablation_levels.csv,
ablation_misspecification.csv). Largest regret from misjudging one mechanism by one level (about a factor of 2):

| knob | family | 5% size | 15% size | changes the decision (> 1 bp) |
|---|---|---|---|---|
| join (replenishment) | queue-reactive | 0.43 | 0.64 | no |
| theta (order-flow momentum) | queue-reactive | 0.24 | 0.76 | no |
| fund (anchoring: impact decay, mean reversion) | agent-mix | 3.99 | 0.00 | yes |
| trend (trend following) | agent-mix | 2.86 | 0.00 | yes |
| withdraw (liquidity withdrawal) | agent-mix | 0.00 | 2.88 | yes |

Mean regret by distance between believed and true level (0 to 4 levels): fund 0 / 0.49 / 0.64 / 1.12 / 2.62;
withdraw 0 / 0.24 / 0.43 / 0.62 / 1.69; trend 0 / 0.25 / 0.43 / 0.69 / 1.55; join 0 / 0.15 / 0.34 / 0.48 / 0.76;
theta 0 / 0.11 / 0.22 / 0.50 / 0.58.

Along the queue-reactive knobs the optimal strategy changes label often (front-loaded, back-loaded, immediate, TWAP)
but every strategy costs about the same (TWAP regret at most 1.3 bps). Along the agent-mix knobs signal reversal is
optimal almost everywhere and TWAP costs 1-7 bps; the decision flips (to back-loading at strong anchoring, to TWAP at
strong trend following or strong withdrawal) at the extremes. The decision-relevant mechanisms are fundamental
anchoring, trend following and liquidity withdrawal, the agent-mix mechanisms the Hawkes and ABIDES candidates do not
represent.

## 3. Out-of-family rules (goal2_out_of_family.py; rules/out_of_family.csv)

| train → test | gated minus TWAP | shrinkage minus TWAP |
|---|---|---|
| queue-reactive (k 0.25) → agent-mix | +7.28 [+3.02, +8.98] | +1.06 [+0.45, +1.63] |
| queue-reactive → Hawkes-built | -1.94 [-4.61, +0.69] | -1.52 [-2.40, -0.63] |
| agent-mix (k 3) → queue-reactive | 0.00 (always TWAP) | +0.12 [-0.14, +0.45] |
| agent-mix → Hawkes-built | 0.00 (always TWAP) | +0.91 [-1.23, +3.02] |
| Hawkes-built (k 0) → queue-reactive | +0.05 [-0.27, +0.65] | -0.19 [-0.33, -0.07] |
| Hawkes-built → agent-mix | +7.46 [+3.84, +10.02] | +2.20 [+0.66, +5.01] |

Selection by metaorder accuracy (no training) minus TWAP: queue-reactive +0.16 [-0.07, +0.89], agent-mix +1.16
[+0.01, +3.19], Hawkes-built +1.60 [-3.39, +4.90]. By the criterion of the addendum, only a rule that always picks TWAP
generalises; the right level of trust in the simulators depends on the market family.
