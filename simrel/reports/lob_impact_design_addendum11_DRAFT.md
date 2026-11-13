STOPPED, 13 November 2026: the predatory family is stopped by the rule in simrel/reports/addendum11_round2_rule.md; no Test R is drawn. See simrel/reports/addendum11_round2.md.

# Reactive markets: do the frozen rules survive a market that responds to the schedule (design addendum 11, DRAFT)

DRAFT. Not frozen, not checksummed. Fields marked TBD are open and need a team decision. Nothing in this draft has
been run except the development pilot (simrel/reports/addendum11_pilot.md). The held-out markets below are specified
and not drawn. The frozen rules of addendum 10 are not changed by this addendum.

## 1. Questions

Q11.1. In markets where a participant detects and trades against regular schedules, do the frozen addendum 10 rules
still do no worse than always-TWAP?
Q11.2. The reactive participant is close to invisible in passive data but changes the true cost of TWAP. Do simulators
fitted to passive data, and the rules that use them, miss this cost and keep recommending TWAP where it is no longer
safe?
Q11.3. Does a randomised TWAP (P0r_twap_random) remove the added cost, and does any rule find it?

## 2. Market sets and leakage control

| set | markets | role |
|---|---|---|
| Pilot (development) | XRP_pilot_a01-a03, XRP_pilot_l01-l03, stream default_rng(20261120) | all tuning of the predator and of this addendum; already used; never reused |
| Test R (held-out) | TBD markets, half agent-mix, half latent (section 2.1) | the evaluation |

Until this addendum is frozen and checksummed, Test R is not drawn. After it is frozen, ranges, sizes, seeds, the
viability rule, the screens and the evaluation criteria do not change. The analysis script is written and checksummed
before it is run.

### 2.1 Test R

Base markets from the existing ranges (sampled.AM_RANGES, latent.RANGES). Draw stream: numpy default_rng(20261121),
agent-mix first, then latent, in fixed blocks of TBD draws per family (as in addendum 10, note 1). The test markets are
the first TBD draws per family that pass section 4, in draw order; rejected draws are listed with their reason. Names
XRa01-NN, XRl01-NN.

Each market is run with the predator off and on. The on version uses predator parameters drawn for that market from
section 3 on the same stream, after the base parameters. Predator random numbers: default_rng([seed, 11, 0]).

Standard task (sell, 10 minutes, 20 decisions, 2 minutes post window). Sizes: 5% and 15% of the predator-off reference
volume (TBD: see section 3.2). Candidates: the 13 strategies of candidate_set plus P0r_twap_random. Truth: TBD seeds
(pilot: 100 seeds give standard errors of 0.1 to 0.5 bp on the TWAP cost difference in five of six markets),
cross-fitted. Observable data for candidates: six passive 30-minute episodes of the predator-on market.

### 2.2 Number of markets

The unit of replication is the market. Pilot values, strong predator, 15% size, five markets: TWAP cost rises by
1.28 bp (between-market s.d. 0.8 bp); headroom over always-TWAP rises from 0.8 to 1.9 bp. At the weak setting and at
5% the pilot shows no effect. Minimum meaningful effect and n: TBD after the team fixes the strength region of
section 3.

## 3. Predator parameters

### 3.1 Ranges

| parameter | meaning | pilot value | proposed range |
|---|---|---|---|
| aggression | front-run size as a multiple of the estimated child | 0.5, 1, 2 | 0.25 to 2.5, log-uniform |
| inventory_cap | maximum position, share of 10-minute reference volume | 1%, 3%, 8% | 0.5% to 10%, log-uniform |
| ts_noise_s | s.d. of time-stamp noise seen by the predator | 0.2 | 0.1 to 0.3, uniform |
| size_cv_tol | size regularity test | 0.15 | fixed 0.15 (TBD) |
| gap_cv_tol | timing regularity test | 0.03 | fixed 0.03 |
| min_matches | trades needed to trigger | 8 | fixed 8 (TBD: 7) |
| window_s | detection window | 240 | fixed |
| lead_s | lead before the expected child | 2 | fixed |
| max_hold_s | longest holding time | 900 | fixed |
| unwind_halflife_s | unwinding speed | 15 | fixed |

Weak region: aggression below 0.75 and cap below 2%. In the pilot this region has no measurable effect on any cost. It
is kept in the range so that the evaluation includes markets where the predator exists and does not matter. Share of
Test R markets in the weak region: TBD.

The detector is the same at every strength. The tight-timing detector (gap 0.0005, no noise) is not part of the family:
it relies on the engine's exact 30 s grid and catches nearly every strategy. It is kept in the code as a comparison.

### 3.2 Order size and lot rounding (TBD)

The pilot found that whether TWAP is detected depends on whether its child size is close to a whole number of lots
(trigger rate 0.06 to 1.00 across markets at the same strength). Choose one before freezing:

a. Leave sizes at 5% and 15% and report the detection rate per market as a covariate.
b. Round the order so that the TWAP child is a whole number of lots.
c. Replace the size test by one in lots (all matched sizes within one lot of the median).

## 4. Viability rule and screens

A draw is kept if all of the following hold. None of them uses a candidate simulator, a trust rule or a true cost.

1. Existing viability rule on 40 unperturbed twins, predator off and on: share of failed runs at most 0.5, share of
   twins moving more than 300 bp at most 0.25.
2. Passive leak screen, on 46 passive 30-minute episodes of the predator-on market: predator share of volume at most
   TBD (pilot: at most 0.2% in every market at strong; negative control 4% to 7%).
3. False triggers per passive episode: reported for every market. Whether to screen on it is TBD. The pilot has two
   markets (a01 at 0.61, a03 at 1.17 per episode) where the predator does trade in passive data. Screening them out
   would hide a real property of the family; keeping them means the "invisible" premise of Q11.2 fails in part of the
   set. Proposed: do not screen; report results with and without markets above TBD triggers per episode.
4. Cost noise screen (TBD): markets like a03 pass rule 1 but have TWAP cost standard errors of 3 to 8 bp at 100 seeds.
   Proposed: no screen, more seeds (TBD) for markets whose predator-off TWAP cost has a standard error above 1 bp at
   100 seeds. The seed count is decided from predator-off runs only.

Reported for the whole set, not used to screen: the pooled observability AUC of the eight addendum 10 diagnostics,
predator on against off, with the off against off null (pilot: 0.53 to 0.56 against a null of 0.52).

## 5. Evaluation against the frozen addendum 10 rules

The rules, their parameters and the design set are those of results/simrel/goal2/addendum10/rules10_frozen.json, used
unchanged. Candidates are fitted to the passive episodes of each predator-on market exactly as in addendum 10. No rule
or simulator is tuned on Test R. The no-simulator rule keeps its development-market choice.

Reported per rule, predator off and on: regret, share of the TWAP-to-oracle gap captured, harmful-decision rate,
abstention rate. Per-market paired differences are averaged over sizes; inference by market bootstrap (2,000
resamples, decision seeds resampled within markets).

Hypotheses, stated before any Test R data exists:

H11.1 (expected to hold). The true cost of TWAP is higher with the predator on than off, outside the weak region.
H11.2 (expected to hold). Always-TWAP regret is higher with the predator on than off, outside the weak region.
H11.3 (expected to fail for the rules). The frozen tier-1 rule has lower regret than always-TWAP with the predator on.
Our prior is that it does not: the candidates see passive data in which the predator is nearly absent, so their
predicted costs for TWAP should not change, and the rule's decisions should be the same on and off.
H11.4. The frozen rule's recommendation is the same on and off in at least TBD of market and size cells (the direct
test of the mechanism behind H11.3).
H11.5. P0r_twap_random has lower true cost than TWAP with the predator on, outside the weak region, and no different
cost with the predator off.

Family-wise level and adjustment: TBD (addendum 10 used Holm at 0.025 one-sided). Secondary, unadjusted: by family, by
size, weak region alone, tier-2 own-execution rule (which does see realised TWAP costs and could notice the predator),
results with and without leaking markets.

## 6. Stopping rules and outputs

- If fewer than TBD markets per family pass section 4 within the draw block, the set is reported as drawn and n is not
  topped up from another stream.
- If H11.1 fails, the family does not create the situation it was built for. This is reported and H11.3 to H11.5 are
  reported as uninformative.
- No rule, simulator or predator parameter is changed after Test R is drawn, except fixes of defects, listed with
  their effect.
- Outputs: results/simrel/goal2/addendum11/ (TBD).

## 7. Known limits carried from the pilot

- The predator loses money at every strength (it pays the spread on each round trip). It is a penalty on regular
  schedules, not a profit-seeking front-runner.
- The first trigger comes at about 210 s of a 600 s order, so at most 12 of 20 children are front-run.
- The predator does not change which strategy is best in the pilot; it raises the cost of TWAP. Headroom added at the
  strong setting and 15% is about 1.1 bp.
- At 5% the detector is not TWAP-specific (TWAP 0.54, P5_liquidity 0.40, tilted schedules 0.23).
