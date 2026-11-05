# simrel: decision reliability of market simulators

Implements `market_simulator_decision_reliability_research_plan.md`: do
simulators that look realistic also give reliable execution-cost predictions
and policy recommendations?

Three claims are kept apart throughout:

1. **Observational realism**: how closely a simulator's unperturbed market resembles a target market.
2. **Cross-simulator robustness**: whether a policy chosen in one simulator stays competitive in another.
3. **Controlled decision validity**: whether a simulator fitted to a market whose rules we know recommends a policy that does well in that market.

Agreement between simulators is never treated as correctness, and nothing
here is validated against a real market.

## Layout

| Path | Contents |
|---|---|
| `core.py` | Task, Observation, EpisodeRecord, and the constraints applied to every policy's request |
| `lob.py` | Price-time-priority matching engine |
| `eventsim.py` | Episode loop for simulators built on `lob.py`; all policies branch from one market state |
| `adapters/abides.py` | ABIDES (via `simbench/abides_world.py`) |
| `adapters/mars.py`, `mars_local.py` | Microsoft MarS, run in-process on CPU, with snapshot branching and optional replay initialisation |
| `adapters/hawkes.py`, `hawkes_fit.py` | Marked multivariate Hawkes LOB simulator and its maximum-likelihood fit |
| `adapters/abides_fit.py` | ABIDES calibration grid |
| `policies/library.py` | Policies P0 to P7 and the candidate set |
| `metrics/cost.py`, `realism.py` | Cost accounting and the realism score |
| `reference_worlds/` | Two market mechanisms with known rules (queue-reactive; behavioural agent mix), two regimes each |
| `configs/` | Pilot and reference-experiment settings |
| `experiments/` | Phase 1 pilots, Phase 3 pilot, Phase 4 reference experiment, analyses |
| `reports/` | Feasibility matrix, limitations, tables and figures |
| `tests/` | Sign conventions, harness constraints, policy causality, matching engine |

## Policy interface

```python
qty = policy(observation, remaining_inventory, time_remaining, state)
```

`observation` is a frozen dataclass holding only what is known at the decision
time. The harness clips the request to `[0, remaining]`, rounds down to the
lot, and sends the remainder in the last slice, so every policy completes the
same parent order and any forced liquidation is in its cost.

## Running

See `reports/feasibility_matrix.md` for the environment and the Phase 1
commands. Then, with `MARS` a MarS checkout and `DATA` its asset directory:

```bash
python -c "import sys; sys.path.insert(0,'.'); from simrel.adapters.mars_local import setup_mars, download_assets; \
           setup_mars('$MARS','$DATA'); download_assets('$DATA')"
python -m pytest simrel/tests
python simrel/experiments/run_pilot.py $MARS $DATA 6        # Phase 3
python simrel/experiments/analyze_pilot.py
python simrel/experiments/run_reference.py $MARS $DATA 6    # Phase 4 (append: all hawkes,abides  to skip MarS)
python simrel/experiments/analyze_reference.py
```

Both runners are resumable. On a laptop, keep the machine awake
(`caffeinate -ims` on macOS).
