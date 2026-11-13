# Reactive family, round 2: a rational predator (addendum 11, development stage)

Written 13 November 2026. Development only, on the six pilot markets (stream default_rng(20261120)). No market was
drawn, stream 20261121 was not touched, no candidate simulator was fitted and no trust rule was applied.

## 1. Verdict

The stopping rule is simrel/reports/addendum11_round2_rule.md (sha256 2fdb07da...f6b3), committed as 8edb5a9 and pushed
before any round-2 code or run existed. Applied as written, at size 15% on the five markets other than a03:

| arm | (a) pooled predator P&L against TWAP, bp of order value | (a) met | (b) markets where TWAP cost rises by 1.0 bp or more | (b) met | result |
|---|---|---|---|---|---|
| rational_medium | -0.89 (s.e. 0.08) | no | 0 of 5 | no | fails |
| rational_strong | -2.09 (s.e. 0.24) | no | 1 of 5 | no | fails |

Neither arm passes. By the rule, the predatory family stops: no Test R is drawn. The finding is negative: in these
markets, at these sizes, with this detector and this exit, a predator that trades against a detected TWAP does not
make money, and when it is made careful enough to lose little it no longer raises the cost of TWAP.

Rise of TWAP cost over predator off, per market (bp):

| arm | a01 | a02 | l01 | l02 | l03 |
|---|---|---|---|---|---|
| rational_medium | 0.01 | 0.28 | -0.17 | 0.09 | -0.13 |
| rational_strong | 1.80 | 0.76 | 0.05 | 0.09 | 0.19 |

Arms the rule does not judge, for comparison: aggressive_strong loses 7.47 bp and meets (b) in 3 of 5 markets;
passive_ungated_strong loses 7.10 bp and meets (b) in 1 of 5; aggressive_medium loses 2.17 bp and meets (b) in none.
No arm meets (a).

Market a03, reported separately as in the pilot: rational_medium P&L -0.92 bp (s.e. 0.24), rational_strong -3.17 bp
(s.e. 0.62). The change in TWAP cost there is -5.5 bp (s.e. 4.7) and -2.0 bp (s.e. 3.5), which is noise.

Figure: results/simrel/goal2/addendum11_round2/fig_round2_rule.png. Tables: verdict.csv and the other CSV files in the
same folder.

## 2. What changed

All changes are in simrel/reference_worlds/reactive.py. The aggressive predator of the pilot is unchanged: the rerun
off, aggressive_medium and aggressive_strong arms reproduce the pilot's 15,600 costs and predator accounts per arm
with a largest difference of exactly 0 (check_against_pilot.csv).

The rational predator is switched on by `exit="passive"` and `gate_mult`.

- Entry. Same detector (noisy time stamps, size_cv_tol 0.15, 8 matches) and same market-order entry, subject to the
  gate.
- Profit gate. Before each entry the predator estimates, from the public trade rows and best quotes only:
  - gain per share = remaining children x mid move per child. Remaining children cannot be observed; the run is
    assumed to last as long again as it has lasted, capped by the maximum holding time. The mid move per child is the
    move of the mid in the run's direction since the run began, divided by the number of trades after the first, and
    not below zero.
  - cost per share = half the spread on entry, plus `exit_allowance` (0.5) times the spread for exits that have to
    cross.
  - It trades only if gain >= `gate_mult` x cost. Rational arms use 1.0; the ungated arm uses 0.
- Passive exit. The whole position rests as limit orders (owner tag "predator") at the best same-side quote, from the
  moment of entry, so the trader's next child can fill them. Orders are cancelled and re-posted if the target price is
  more than `reprice_ticks` (1) away. Before a new entry the predator cancels its resting orders so that it cannot
  trade with itself. What is left 60 s after the pattern stops is taken with market orders.
- Fills. Read from the book's own order-level record, which also separates a fill from a cancellation. The engine
  leaves that record off and never reads it; the wrapper switches it on for an active passive predator. No shared file
  was changed.
- Event log. The predator's limit orders and cancels are logged through the base world's own `record`, as its market
  orders already were.
- Attribution. A ledger on the wrapper counts passive fills taken by the trader (those that appear between the end of
  the base world's step and the `on_external` call) and by background flow. The Predator object has no reference to it.
- Strength 0 is still bit-identical to the base world.

Tests: 53 pass (42 before, 11 new): passive-exit accounting, the gate blocking and allowing a scripted entry,
attribution isolation (no reference, and a scrambled ledger leaves every market path identical), strength 0 with the
rational settings, the aggressive predator unchanged by the new fields, a guard on run_twin_set's deep copies, and
the custom AUC against scipy's Mann-Whitney U on the saved pilot E3 data.

## 3. Run

Arms: off, aggressive_medium, aggressive_strong, rational_medium, rational_strong, passive_ungated_strong. Six
markets, sizes 5% and 15%, 14 strategies, 100 seeds; 40 viability twins; 46 passive episodes; 20 three-hour samples per
market and arm for E3. Same seeds and reference volumes as the pilot. Timing check projected 0.7 hours on 7
processes. No run failed.

## 4. Results

Means over the five markets other than a03. P&L is against TWAP, in bp of order value.

### 4.1 Costs and headroom

| size | arm | TWAP cost minus off | always-TWAP regret | regret, cross-fitted | P0r minus TWAP |
|---|---|---|---|---|---|
| 5% | off | 0.00 | 1.04 | 0.63 | 0.10 |
| 5% | aggressive_medium | 0.17 | 1.34 | 1.00 | -0.22 |
| 5% | aggressive_strong | -0.03 | 1.02 | 0.65 | 0.03 |
| 5% | rational_medium | -0.09 | 1.07 | 0.71 | 0.06 |
| 5% | rational_strong | -0.05 | 1.04 | 0.63 | 0.08 |
| 5% | passive_ungated_strong | 0.00 | 1.07 | 0.64 | 0.12 |
| 15% | off | 0.00 | 0.77 | 0.57 | 0.33 |
| 15% | aggressive_medium | 0.37 | 1.15 | 0.86 | -0.09 |
| 15% | aggressive_strong | 1.28 | 1.89 | 1.58 | -0.95 |
| 15% | rational_medium | 0.02 | 0.75 | 0.65 | 0.27 |
| 15% | rational_strong | 0.58 | 1.26 | 0.90 | -0.14 |
| 15% | passive_ungated_strong | 1.14 | 1.72 | 1.38 | -0.65 |

The best strategy is the same in every arm in nine of the ten market and size cells, as in the pilot.

### 4.2 The predator's account against TWAP

Each component is measured against the mid when the order traded; "holding" is the move of the mid while the position
was held. The components add up to the P&L.

| size | arm | P&L | entry cost | passive-exit gain | aggressive-flatten cost | holding | entry volume / order |
|---|---|---|---|---|---|---|---|
| 5% | aggressive_medium | -1.29 | -0.66 | 0 | -0.64 | 0.01 | 0.12 |
| 5% | aggressive_strong | -2.72 | -1.35 | 0 | -1.32 | -0.05 | 0.25 |
| 5% | rational_medium | -0.31 | -0.24 | 0.17 | -0.09 | -0.15 | 0.05 |
| 5% | rational_strong | -0.54 | -0.50 | 0.36 | -0.16 | -0.24 | 0.09 |
| 5% | passive_ungated_strong | -1.74 | -1.35 | 0.82 | -0.57 | -0.64 | 0.25 |
| 15% | aggressive_medium | -2.17 | -1.05 | 0 | -1.01 | -0.11 | 0.17 |
| 15% | aggressive_strong | -7.47 | -3.10 | 0 | -3.18 | -1.20 | 0.45 |
| 15% | rational_medium | -0.89 | -1.15 | 1.08 | -0.11 | -0.71 | 0.19 |
| 15% | rational_strong | -2.09 | -2.92 | 2.37 | -0.40 | -1.14 | 0.41 |
| 15% | passive_ungated_strong | -7.10 | -6.98 | 5.13 | -1.28 | -3.98 | 0.93 |

For the aggressive arms the flatten column is the pilot's market-order unwinding.

Reading:

- The passive exit does what it was meant to: it turns most of the exit from a cost into a gain. Passive gain recovers
  about 80% of the entry cost.
- It is not enough. Two losses remain. The part that still has to be flattened with market orders costs 0.1 to 1.3 bp.
  The holding term is negative in every passive arm: the predator buys back, on average, at a higher mid than it sold
  at. A resting bid is filled quickly when the price falls, which caps the gain, and has to be re-posted higher when
  the price rises, which keeps the loss.
- The passive exit alone does not reduce the loss. passive_ungated_strong loses 7.10 bp against 7.47 bp for
  aggressive_strong, because the position is bought back within seconds, the inventory cap stops binding, and the
  predator enters twice as much (0.93 of the order against 0.45).
- The gate is what cuts the loss, from 7.10 to 2.09 bp, by cutting entries by more than half. It cuts the effect on
  TWAP by half as well (1.14 to 0.58 bp).
- No arm is profitable in any of the five markets at 15%. The best per-market P&L is -0.10 bp (rational_medium, l02,
  where the predator enters in 13% of runs).

### 4.3 Counterparty and exits

| size | arm | share of passive buyback filled by the trader | share of exit volume flattened with market orders | background cancels / passive fills | gate block rate | runs with an entry |
|---|---|---|---|---|---|---|
| 5% | rational_medium | 7% | 24% | 2.3 | 58% | 38% |
| 5% | rational_strong | 2% | 21% | 1.8 | 58% | 38% |
| 5% | passive_ungated_strong | 4% | 23% | 2.7 | - | 60% |
| 15% | rational_medium | 12% | 11% | 1.4 | 60% | 57% |
| 15% | rational_strong | 8% | 12% | 1.3 | 60% | 57% |
| 15% | passive_ungated_strong | 7% | 16% | 1.5 | - | 85% |

- The trader is a minor counterparty. About 90% of what the predator buys back passively is sold to it by background
  flow, not by the trader's children. The predator is not, in the main, selling ahead of the trader and buying from
  the trader. In a03 the trader's share is higher (27% to 32% at 15%).
- The base worlds cancel the predator's resting orders heavily: 1.3 to 2.7 shares cancelled for every share filled.
- The gate does not make the 5% size inactive. It blocks 58% of entries at 5% and 60% at 15%, and the share of runs
  with at least one entry falls from 60% to 38% at 5% and from 85% to 57% at 15%. The 5% size is inactive in its
  effect, as in the pilot: no arm changes the cost of TWAP there by more than 0.2 bp.

### 4.4 Viability, false triggers and observability

- E4. All 36 market and arm cells pass the viability rule. Volume and spread change by less than 1% outside a03.
- False triggers on 46 passive episodes. The detector is unchanged, so trigger rates are those of the pilot (a01 0.63
  per episode, a03 1.1 to 1.3, others 0.15 or less). The new arms trade in the same episodes: the predator trades in
  44% of passive episodes in a01 and 30% in a03. Its share of volume stays at or below 0.2%. In passive data the gate
  blocks 5% to 10% of entries in a01 and about 36% to 39% in a03 and l03.
- E3, pooled cross-validated AUC of the eight diagnostic features, predator on against off:

| arm | AUC | permutation 95th percentile | permutation p |
|---|---|---|---|
| off against off (null) | 0.53 | 0.58 | 0.23 |
| rational_medium | 0.51 | 0.58 | 0.42 |
| rational_strong | 0.57 | 0.57 | 0.06 |
| passive_ungated_strong | 0.56 | 0.58 | 0.13 |

  rational_strong is at the edge of the permutation band. Resting bids are a little more visible than market orders
  alone, but the difference from the null is not established with 120 samples per class.

## 5. Limits

1. One design of a rational predator was tested: orders rest at the best quote from the moment of entry. A predator
   that rests its exit one tick behind (`exit_offset_ticks` = 1 exists and was not run), waits before posting, or sizes
   its entry to the queue could do better. The rule fixed the arms in advance, so none of these was tried, and the
   negative finding is for this design.
2. The gate's estimate of remaining children (as many again as seen so far) is a guess. With the first trigger at
   about 210 s it happens to be close to the truth for a 20-child TWAP.
3. The tick is 10 bp of the price and the spread is about one tick. In such a market a predator cannot quote inside
   the spread and its passive fills depend on queue position. The result may not carry over to a small-tick market.
4. The base worlds cancel resting shares newest first, whoever posted them, so they cancel the predator's orders more
   than an old order would be. Protecting them would need a change to the base worlds, which was not allowed. This
   lowers the predator's passive fill rate and works against it.
5. Passive fills are seen by the predator at its next wake-up (at most 5 s later). The mid used for the P&L
   components of a passive fill is the last quote before that wake-up, so the split between "passive gain" and
   "holding" is approximate. The total P&L is exact.
6. The result is for the pilot's six markets, 100 seeds, the noisy detector and sizes 5% and 15%. a03 remains too
   noisy to read.
7. Residual inventory at the end of the post window (marked at the final mid) occurs in about 2% of passive
   episodes in a01, as in the pilot.

## 6. What this means for addendum 11

The predatory family is stopped. The draft addendum is marked as stopped and is otherwise unchanged. What the two
rounds show together:

- A predator that hurts TWAP by about 1.3 bp (aggressive, strong, 15%) loses 7.5 bp doing it.
- A predator built to avoid losses (gate and passive exit) still loses 0.9 to 2.1 bp and hurts TWAP by 0.0 to 0.6 bp.
- TWAP's regular schedule is detectable at 15%, but in these markets the detection is not worth money: the child
  orders move the price by much less than the spread.
