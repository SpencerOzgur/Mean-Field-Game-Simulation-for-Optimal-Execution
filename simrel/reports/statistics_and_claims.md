# Statistics, units of independence and the status of every claim

7 November 2026. Written for the manuscript's methods section and for reviewers.

## 1. Units of independence

| level | what varies | count | independent? |
|---|---|---|---|
| market family | queue-reactive, agent-mix, Hawkes-built, latent liquidity | 4 | yes, but too few to resample; results are reported per family |
| reference market | one set of mechanism parameters | 4 + 6 + 10 + 6 (+6 held-out) | yes: the primary unit of replication |
| order size | 5% and 15% of the market's volume | 2 per market | no: the same market; never counted as independent |
| candidate design | 29 designs, 28 of them Hawkes variants, refitted in every market | 29 | no: designs share families, fitting code and the market's data |
| decision seed | Monte Carlo replications of a candidate's predicted costs | 30 or 100 | no: repeated evaluations within one market and design |
| truth seed | Monte Carlo replications of the true costs | 300 (200 in ablations) | no: estimates one market's true costs |

Real-data analyses (Goal 1): the unit is the stock (INTC discovery; CSCO, MSFT confirmatory); within a stock, intervals
resample seeds within slices (ten-minute windows grouped by volatility, volume and trend).

## 2. What each interval resamples

- Goal 2 correlations, selection regret and paired differences: three-level bootstrap, 2,000 resamples (1,000 in the
  addendum 8 and 9 analyses): markets with replacement; designs with replacement, one draw shared across the markets
  of a resample; decision seeds within each market and design, from which predicted costs, recommendations and
  regret are recomputed. Realism and LOB-Impact scores are held fixed (their own sampling error is not propagated).
  The truth's own noise is not resampled; the cross-fitted decomposition (best strategy identified on truth seeds
  0-149, costs evaluated on 150-299) is the check for it.
- Oracle and four-decision-maker decomposition: markets and decision seeds (and designs for the candidates).
- Goal 1: seeds within slices; equivalence tests use 90% intervals (two one-sided tests at 5%).
- No interval treats candidate designs, order sizes or seeds as independent replications of a market.

## 3. Multiple comparisons

The confirmatory tests on the ten drawn markets (addendum 4, 30 seeds, primary), with one-sided bootstrap p-values
(share of resamples on the wrong side of zero):

| hypothesis | estimate [95%] | one-sided p | Holm, 2 directional tests |
|---|---|---|---|
| C1 metaorder accuracy predicts good advice | +0.15 [-0.03, 0.28] | 0.056 | not significant |
| C2 choosing by it beats choosing by LOB-Bench order book | -2.66 [-4.75, +0.03] | 0.029 | not significant (needs 0.025) |
| C3 no realism measure predicts good advice | all point estimates <= 0; largest share above 0: 0.37 | - | holds (a claim of absence) |
| C4 choosing by metaorder accuracy against always TWAP | +0.64 [-0.20, +1.94] | - | descriptive |

No directional confirmatory hypothesis survives a multiplicity adjustment. The Goal 1 replication (H1-H5) is reported
per stock with each test at its stated level; H1 holds on all three stocks, so it does not depend on an adjustment.
Exploratory analyses (the literature checks, the error diagnostic, the budget curves) carry no multiplicity adjustment
and are labelled exploratory wherever they appear.

## 4. What "structural" means

Structural contribution = regret of the candidates at 100 seeds minus regret of the true generator at 100 seeds
(C - A, addendum 8). It contains:

1. mechanism error: the candidate family lacks a mechanism of the market;
2. parameter-estimation error: a candidate with the right structure estimates its parameters from six 30-minute
   episodes;
3. model-selection effects: averaging over 29 designs, most of which are not the generating one;
4. finite-sample effects that 100 seeds do not remove (regret at 100 seeds still contains noise for both C and A).

These are separated where the design allows: in the well-specified markets the generating design refitted from data
isolates (2) against the oracle; in addendum 9 the true generator with one mechanism switched off isolates (1) with no
estimation or selection error; the budget curves (addendum 9) show which part falls with more seeds. In the agent-mix
and held-out markets no candidate shares the generating structure, so (1)-(3) are not separable there: the paper calls
the remainder "structural (model) error", not "mechanism error", except where addendum 9 isolates it.

## 5. Confirmatory and exploratory claims

| claim | evidence | status |
|---|---|---|
| Similarly realistic simulators recommend opposite strategies | H1 on CSCO and MSFT, pre-registered after INTC discovery | confirmatory |
| No observational realism score predicts good advice | C3 on ten drawn markets; held-out family | confirmatory; replicated out of family |
| Own-impact accuracy predicts good advice | C1 not met at 30 seeds; +0.07 at 100 seeds | not supported |
| Choosing by own-impact accuracy beats choosing by LOB-Bench | C2: p = 0.029, not significant after adjustment; supported at 100 seeds and in the held-out family (secondary) | suggestive |
| Simulator-based selection beats always TWAP | not in any market set | not supported |
| Optimisation over a misspecified simulator is worse than TWAP | misspecified 20 markets (exploratory comparison), held-out family (pre-specified in addendum 8) | supported; the held-out result is the confirmatory one |
| Excess regret is structural, not sampling, where mechanisms are missing | addendum 8 decomposition (agent-mix, held-out) with its pre-stated criterion; addendum 9 | supported |
| Conservative gating prevents losses | A1, A3 on the drawn markets and the held-out family | supported, but it works by never leaving TWAP |
| A decision-aware oracle metric predicts regret better than impact measures | addendum 8, step 4 | not supported |
| Removing one mechanism from the true generator causes regret that more seeds do not repair | addendum 9, H9b, four agent-mix markets (fundamental anchoring) | supported (pre-specified); H9a (oracle regret falls to zero) holds in 3 of 4 markets, R2d's truth is too noisy to identify the best |
| Fixed designs refitted per market follow the true optimum as a mechanism changes | addendum 8, step 3, 30 ablation markets | not supported in agent-mix (2-10% of levels; decision-boundary test met in 88-100% of cells); moot in queue-reactive (all regrets < 0.5 bp) |
| The findings hold for a 30-minute, 60-decision task | addendum 8, step 5 variant, six held-out markets | supported: no fitted design beats always TWAP on average; the oracle does |

The 100-seed reruns of the same markets (addendum 7) are not independent confirmation of hypotheses developed on those
markets; they are reported as secondary. Independent tests are the drawn markets (addendum 4) and the held-out family
(addendum 8).
