# Oracle positive control, mechanism ablation and out-of-family rules (design addendum 6)

Written 5 November 2026, after all results of addenda 1-5 and the exploratory checks, before any of the computations
below. Prompted by review: the study asks three questions (can simulators reproduce markets; can they rank strategies;
can observable validation tell which to trust), and the last has a negative answer. The reframed question is: under
what conditions does a simulator's predictive error become amplified into poor execution decisions, and can
observable validation detect those conditions before optimisation? Three experiments address it.

## 1. End-to-end oracle (positive control)

The exact generator of each of the 26 reference markets, with its true parameters, is used as the "simulator" and run
through the unchanged candidate pipeline (simrel/experiments/goal2_run.py policy stage, candidate name "oracle"): the
same task, strategies, twin pairing and recommendation rule, on policy seeds 62000+ (the truth used seeds 0-299).
100 seeds per market; results are reported at 30 seeds (as for the candidates; the first 30) and at 100.

Reported per market group: mean regret of the oracle's recommendation; the share of cells where it recommends the
true best or a strategy within 0.25 bps of it; rank correlation of predicted and true costs over the 12 strategies;
and, for the well-specified markets, the candidate with the generating design but refitted parameters (parameter
error alone).

Pass criterion, fixed now: at 30 seeds, mean oracle regret at most 0.5 bps and at most a quarter of always-TWAP regret
in every group, and median rank correlation at least 0.8. If the oracle fails, the decision pipeline itself (sampling
noise, cost estimation, twin pairing) and not misspecification limits the decisions, and the Goal 2 conclusions are
re-read in that light.

## 2. Controlled mechanism ablation

Two base markets (queue-reactive R1a defaults; agent-mix R2a defaults). One mechanism is varied at a time over five
levels, all other parameters at base values; the order size is fixed in shares at the base market's 5% and 15%
(2,100 and 6,400 shares for the queue-reactive base, 4,700 and 14,300 for the agent-mix base: 5% and 15% of the base
reference volume, rounded down to 100), so only the mechanism changes.

| knob | family | levels (base marked) | mechanism |
|---|---|---|---|
| join | queue-reactive | 0.3, 0.6, 1.2*, 2.4, 4.8 | liquidity replenishment at the best |
| theta | queue-reactive | 0.4, 0.8, 1.2*, 2.0, 2.8 | order-flow momentum (imbalance chasing) |
| fund | agent-mix | 0.025, 0.05, 0.1*, 0.2, 0.4 | fundamental anchoring: impact decay and price mean reversion |
| withdraw | agent-mix | 0.5, 1, 2*, 4, 6 | liquidity withdrawal after one-sided flow |
| trend | agent-mix | 0.0125, 0.025, 0.05*, 0.1, 0.2 | trend following (momentum) |

22 markets (2 bases + 20 variants). For each: the true cost of every strategy at both sizes (200 seeds). Reported:
the true optimal strategy and the cost of every strategy along each knob; and the decision-misspecification matrix,
regret in market j of the strategy that is optimal in market i of the same knob, i.e. the cost of misjudging that
mechanism by one, two, three or four levels. A mechanism "changes the decision" if misjudging it by a factor of 2
(one level; for theta, the nearest level) costs more than 1 bp of regret in at least one direction at either size.

## 3. Out-of-family validation of decision rules

Families: queue-reactive (R1*: 10 markets), agent-mix (R2*: 10), Hawkes-built (W*: 6). For each ordered pair (train,
test) of different families: the gated ensemble rule of addendum 5 (historical errors, biases and k chosen on the
training family by the same procedure, then frozen) and the Bayesian-shrinkage rule (prior from the training family)
are evaluated on the test family without retuning; selection by metaorder impact accuracy needs no training and is
reported per test family. Reported: mean regret and the paired difference against always TWAP, with 95% intervals
from the three-level bootstrap (markets, designs, seeds). A rule generalises if it is not worse than TWAP (interval of
the difference not above zero) on every test family.
