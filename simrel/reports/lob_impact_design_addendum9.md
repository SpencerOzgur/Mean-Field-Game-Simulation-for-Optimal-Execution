# One mechanism removed from the true generator, and regret against simulation budget (design addendum 9)

Written 7 November 2026, evening, after the final 100-seed and held-out results (addenda 7, 8) and before any of the
runs below. Prompted by review: make the causal explanation decisive, and show whether regret falls with more
simulation, plateaus because of model error, or stays low because the strategies are nearly equivalent.

## 1. The true generator with one mechanism removed

Markets: four agent-mix markets whose best strategy is clearly distinct (gap to the second best 1.3-8.5 bps):
R2a_am_anchored, R2b_am_trending, R2d_am_toxic, R2s01. "Simulators": the market's own generator with its true
parameters (the oracle), and the same generator with exactly one mechanism switched off, everything else unchanged:

| simulator | change | mechanism removed |
|---|---|---|
| oracle | none | — |
| oracle-no-withdraw | withdraw = 0 | liquidity providers stepping back from one-sided flow |
| oracle-no-fund | fund = 0 | fundamentalists trading towards the latent value (anchoring) |
| oracle-no-trend | trend = 0 | trend followers |

300 decision seeds each (62000-62299; the truth used 0-299 of a separate stream). No fitting is involved, so a
removed-mechanism simulator has no parameter-estimation or model-selection error: whatever regret it adds over the
oracle at the same budget is structural by construction.

Reported per market and simulator, at budgets n = 10, 30, 100, 300 seeds (n < 300: mean over disjoint blocks):
regret; whether its recommended strategy is the true best; rank correlation of predicted and true costs; and the error
in each strategy's predicted gain over TWAP. Hypotheses: (H9a) the oracle's regret falls towards zero as n grows;
(H9b) for at least one removed mechanism per market, regret at n = 300 stays above 1 bp and does not fall from n = 100
to n = 300 (more seeds make the wrong answer more precise, not right). A removed-mechanism simulator whose unperturbed
runs fail the stability gate of the study is reported as such.

## 2. Regret against simulation budget, all markets

From the existing runs (100 seeds per candidate and per oracle; exploratory, since 30- and 100-seed results are already
known): mean regret of the oracle, the single candidates, the ensemble's cheapest strategy and always TWAP at n = 5, 10,
20, 30, 50, 100 seeds (mean over disjoint blocks), by family (queue-reactive, agent-mix, well-specified, held-out
latent). Reading: regret falling towards zero means estimation error; a plateau above the oracle means model error; all
curves near TWAP's means nearly equivalent strategies.
