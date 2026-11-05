# Limitations and unverified claims

This file lists what the study does not establish. It is maintained alongside
the code; numbers quoted here come from `simrel/reports/`.

## What was not done

| Planned | Status | Why |
|---|---|---|
| Experiment I: realism against held-out historical data | **Not done** | No licensed order-level data for any market. The only real data on hand is one public LOBSTER sample day (INTC, 2012-06-21) |
| Train / validation / test split of real trading days | Not done | Same |
| LOB-Bench scoring | Not done | It scores generated data against real LOBSTER-format data, which we do not have. A smaller realism score is computed from fields all adapters expose |
| Calibrating all three simulators to one real instrument | Not possible | No data; and MarS ships no training code, so it cannot be refitted to anything |
| MarS as a like-for-like third candidate | Not achieved | See "MarS" below. It is reported as a separate-domain case study and an exploratory candidate |
| P8, the MFG-derived policy | Not included | Optional in the plan. The earlier pilot found its calibration unstable (a noisy impact estimate divided by a small one), and the MFG solver in `src/equilibrium.py` has a sign error |
| Limit-order policies | Not included | All policies send market orders. Passive execution would need a fill model in each simulator |
| Fees, latency, hidden liquidity | Not modelled | Zero fees; ABIDES default latency; none elsewhere |
| Time-of-day and conditional realism statistics | Not done | Episodes are 10-minute windows at one clock time |
| Pre-registration, power analysis | Not done | Seed counts were set by the compute available on one laptop |

## What the results can and cannot support

- **No real-world oracle.** Every regret in this study is measured inside a simulated market. Reference-world regret is ground truth only for that reference world, and "best" means best among the twelve candidate policies, not globally optimal.
- **Reference worlds are ours.** Two mechanisms (queue-reactive, behavioural agent mix) with two regimes each. They are not claimed to resemble any real market, and four worlds are few. A different set could rank the simulators differently.
- **The Hawkes candidate has a home advantage.** It shares its matching engine, lot size and event vocabulary with the reference worlds, and is fitted by maximum likelihood on their event logs. ABIDES is tuned by a coarse grid search over three parameters and uses its own exchange. The two fits did not have the same budget or the same access to structure.
- **Three matching engines.** ABIDES-Markets, MarS's `mlib`, and `simrel/lob.py` all use price-time priority, but order handling differs in detail (for example, how a marketable order that exceeds the visible book is treated). This is a confounder, not something we harmonised away.
- **Realism score.** Six statistics from a 5-second quote grid. It ignores order sizes, event mix, depth and inter-arrival times, which the adapters do not expose uniformly. A simulator could score well on it and still be wrong about everything that matters for impact.
- **Few points for "does realism predict regret".** At most (simulators) x (reference worlds) x (sizes) settings, and settings within a world are not independent. Any correlation reported is descriptive.
- **Statistical power.** Policy differences of a few basis points are inside the noise in the more volatile worlds. Tables report intervals; cells whose intervals include zero are not evidence of equality.
- **Selection noise.** A simulator's "recommended policy" is the candidate with the lowest mean cost over its seeds. When several candidates are statistically tied, the recommendation is partly luck, and its reference regret inherits that.

## MarS

- Trained on Chinese equities; initialised here either from MarS's own noise agent or from replayed reference-world orders. Neither is a real market state, and the authors state that production use requires real order-level history.
- Run outside its supported environment (CPU, no Docker, no Ray). The model's outputs were validated against the official validation samples, but the surrounding simulation was not compared with a run in the supported environment.
- Only the 2M and 10M models were used. The larger models the paper reports on (up to 1.02B parameters) are not released, and conclusions here do not transfer to them.
- Unstable after hand-over from the initial state, and some rollouts crash or run away; these are counted as failures and excluded, which conditions the surviving sample on stability. A pre-task screen (mid move in the five minutes before the task) removes further seeds using pre-treatment information only.
- Child orders are marketable limit orders capped at the ten visible levels, because the engine has no market-order type.
- In the reference experiment MarS is not fitted. It is the pretrained model conditioned on a replay of reference-world order flow. Its regret there measures zero-shot transfer of a model trained on other markets, not the quality of the MarS approach.

## ABIDES

- Only configurations that were stable in the earlier pilot are used in Phase 3. Two of six configurations in that pilot were unstable and one seed never terminated.
- ABIDES cannot snapshot a market, so its twins are same-seed reruns that are identical only until the first child order. Paired comparisons are noisier than in the simulators that branch from one state.
- At a mean price of 1000 cents the RMSC04 parameters were rescaled linearly. No published calibration exists at that price level.

## Earlier pilot (context)

`results/simbench/` and `paper/` hold an earlier, narrower study (six ABIDES
configurations and a retrained TRADES market). Its instabilities (ABIDES
configurations with large unperturbed price moves; a retrained TRADES model
whose spread widened with time) are engineering limitations of those set-ups,
not evidence about the model families.
