# Continuation check: do fitted simulators still lose to TWAP in markets with little order-flow continuation?

Status: exploratory. It uses markets whose results have already been read (Test B of addendum 10 and the development
markets). No candidate was fitted and no decision was run; the only new simulation is passive episodes. The rules were
fixed in simrel/reports/continuation_check_plan.md (commit c6acb0d) before continuation was joined with any regret.
Code: simrel/experiments/continuation_check.py. Outputs: results/simrel/goal2/continuation_check/.

**Result: the benchmark cannot answer the question.** Enough Test B markets have little continuation (27 of 48), but in
those markets always-TWAP is already almost optimal (mean full-truth regret 0.13 bp, below the 0.5 bp headroom
condition), so there is nothing for a simulator to win or lose. The low-continuation subset is also mostly
queue-reactive. By the plan this is a limitation, not a result, and the regret comparisons were not run.

## 1. Inventory

| Source | Markets | Arms | Regret | Used |
|---|---|---|---|---|
| test_b.csv (Test B) | 48: 16 queue-reactive, 16 agent-mix, 16 latent; 2 sizes | always TWAP, no-simulator, rule (H10.1), corrected, original h_c_f_m_k1, compound Hawkes | `regret_full` full-truth (300 seeds); `regret` cross-fitted | primary |
| dev_fourway.csv, 16 core markets | R1a to R1e, R2a to R2e, W1 to W6; 2 sizes | always TWAP, oracle, original and recalibrated h_k1 and h_c_f_m_k1, compound Hawkes, corrected, frozen rule, corrected + frozen rule | development truth only | development only; W1 to W6 dropped (below) |
| heldout and seeds100 selection tables | 32 and 26 markets | always TWAP, single designs chosen by realism, ensembles | one definition; identical where they overlap | ensemble for the development markets only |
| test_a.csv (Test A) | 16: R1s, R2s, R3s | rules only (agreement, gated, diagnostic, sensitivity), no-simulator, oracle | `regret` full-truth, `regret_cf` cross-fitted (names opposite to Test B) | not used |
| dev_fourway.csv, 26 ablation markets | Xqr_*, Xam_* | as above | development truth | S4 only, descriptive |

Not used, and why:
- Test A: results/simrel/raw/sampled_markets.json and latent_markets.json are not on this machine. Rebuilding the markets
  would require rerunning the viability screen. If the files are supplied, Test A is added under the same rules.
- W1 to W6 (Hawkes-built development markets): they are built from the fitted Hawkes pickles in
  results/simrel/goal2/raw, which are not on this machine; recreating them would be new fitting. The development source
  is therefore the 10 hand-built markets R1a to R2e.
- Ablation markets: perturbations of two base markets, not independent; S4 only (section 6).

## 2. Continuation measure

As in the reversal check, with simrel/metrics/reversal.py unchanged: 20 passive episodes per market (seeds 70000 to
70019, 600 s warm-up, one 3,600 s segment), parent quantity 15% of the mean passive volume per 600 s window. S4 is the
mean impact-free P6-minus-TWAP timing cost over all 600 s windows at 30 s decisions, divided by the tick in bp; negative
means P6 is cheaper, that is, continuation. S3 slope is in ticks per unit of signal. For R1a to R2e the reversal check's
stored passive summaries were reused; S4 recomputed from them equals sim_table.csv to 2e-16 bp.

Real stocks (reversal check, 8 days, S4 in ticks): INTC -0.018 [-0.058, +0.027], CSCO +0.015 [-0.027, +0.058],
MSFT -0.028 [-0.089, +0.032].

## 3. Counts and feasibility

A market is low-continuation if S4 >= -0.05 ticks (sensitivity: -0.10).

| Source | Family | Markets | Low at -0.05 | Low at -0.10 | S4 min | median | max |
|---|---|---|---|---|---|---|---|
| Test B | queue-reactive | 16 | 16 | 16 | -0.031 | -0.002 | 0.000 |
| Test B | latent | 16 | 11 | 14 | -0.112 | -0.034 | +0.029 |
| Test B | agent-mix | 16 | 0 | 2 | -0.528 | -0.174 | -0.068 |
| development | queue-reactive | 5 | 4 | 4 | -0.131 | -0.022 | 0.000 |
| development | agent-mix | 5 | 1 | 1 | -0.289 | -0.177 | +0.127 |

- Feasibility (at least 8 Test B low-continuation markets): met, 27 at -0.05 and 32 at -0.10.
- The development source has 5 low-continuation markets, below 8; by the plan it is joined with regret only if Test B
  goes on, which it does not.
- Composition: the low-continuation subset at -0.05 is 16 queue-reactive and 11 latent, with no agent-mix market, so
  more than half is queue-reactive. Low continuation and family are almost the same split: agent-mix is the rest.
- In the queue-reactive family, low continuation mostly means a price that rarely moves within a window, not a real-stock
  price process: on average P6 and TWAP cost exactly the same in 68% of windows, and in three Test B markets (TBq_d03,
  TBq_d06, TBq_d09) in every window (S4 = 0 exactly). In Test B the latent family is different from the latent pilot
  markets of the reversal check (-0.141 ticks): its median S4 is -0.034 and its median S3 slope -0.001 ticks per unit,
  both within the real stocks' range.

## 4. Headroom

The only regret read: always-TWAP's full-truth regret in the low-continuation subset (per market, mean of the two sizes).

| Threshold | Low markets | Families | TWAP regret, mean (bp) | median | Headroom (>= 0.5 bp) |
|---|---|---|---|---|---|
| -0.05 | 27 | 16 queue-reactive, 11 latent | 0.13 | 0.12 | no |
| -0.10 | 32 | 16 queue-reactive, 14 latent, 2 agent-mix | 0.32 | 0.13 | no |

By family at -0.05: queue-reactive mean 0.13 bp (maximum 0.44), latent 0.14 bp (maximum 0.29); no market reaches
0.5 bp. At -0.10 the only markets at or above 0.5 bp are the 2 agent-mix markets (1.07 and 3.59 bp) and 2 latent markets
(0.55 and 1.28 bp), whose S4 lies between -0.10 and -0.05.

Headroom fails at both thresholds, and the subset is mostly queue-reactive. By plan sections 5 and 7 the comparisons
were not run, and no arm other than always TWAP was read in either subset. The figure
(fig_continuation_check.png) shows S4 by market and the headroom quantity; it shows no simulator arm.

## 5. Reading

The third reading applies: **the benchmark cannot answer whether the main finding depends on continuation.** In the
Test B markets with little continuation, always TWAP is within 0.13 bp of the best strategy on average, so any choice
of strategy, with or without a simulator, can lose at most a fraction of a basis point to TWAP or gain essentially
nothing over it. The markets where a choice of strategy matters are those with continuation. This is the same kind of
limitation as the external-validity screen (0 of 400 draws with a Nasdaq-like price response): the reference families
do not contain markets that both look like the real stocks in this respect and leave something to decide.

What this does not say: it does not say that simulators lose or win in low-continuation markets, and it does not say
that the main finding is driven by continuation. Both remain open.

## 6. Ablation markets (descriptive, outside all comparisons)

S4 in ticks, 20 passive episodes per market. Agent-mix base (flow half-life 20): -0.229.

| Knob (agent-mix) | Levels: S4 |
|---|---|
| fund (fundamentalist strength) | 0.025: -0.330, 0.05: -0.355, 0.2: -0.092, 0.4: -0.026 |
| trend (trend-follower strength) | 0.0125: -0.341, 0.025: -0.281, 0.1: -0.114, 0.2: -0.040 |
| withdraw (depth withdrawal) | 0.5: -0.326, 1: -0.298, 4: -0.141, 6: -0.086 |
| flow_halflife | 5: -0.301, 10: -0.235, 40: -0.181, 80: -0.160 |

Queue-reactive (base theta 1.2: -0.005): join 0.3: +0.002, 0.6: -0.010, 2.4 and 4.8: 0.000 (price never moves within a
window); theta 0.4: -0.000, 0.8: -0.000, 2: -0.023, 2.8: -0.039.

In the agent-mix family, continuation falls as the fundamentalist strength rises (only at 0.4 does it reach the
low-continuation range) and also as the trend-follower strength rises and as depth withdrawal rises.
Continuation falls slowly with a longer flow half-life. In the queue-reactive family continuation is near zero
throughout and grows only with theta. One seed set per market; these are descriptions, not tests.

## 7. Limits

- Exploratory, on already-read test markets. The rules were written before the join, but the analyst had seen Test B's
  overall results.
- One market set (Test B). Test A and W1 to W6 could not be rebuilt on this machine.
- S4 is a point estimate from 20 episodes; 10 of the 27 low-continuation markets have S4 intervals entirely below zero.
  Classification by point estimate follows the plan.
- Low continuation and family are almost the same split, so even with headroom, a low-minus-rest difference would
  largely be a family contrast.
- The headroom condition (0.5 bp) and the threshold (-0.05 ticks) were fixed by the plan; the conclusion holds at the
  -0.10 sensitivity threshold.
