# Feasibility matrix (Phase 0 and Phase 1)

Audit date: 2026-10-11. Machine: one Apple M1 laptop, 16 GB, no NVIDIA GPU.
Everything below was checked by running it; the script or file that holds the
evidence is named in each row. "Not verified" means exactly that.

## Verdict

| Simulator | A Runnable | B Interactive | C Comparable | D Stable | E Reproducible | Role in the study |
|---|---|---|---|---|---|---|
| ABIDES (abides-jpmc-public) | Pass | Pass | Pass, after setting the price level | Pass for the configurations used | Pass | Candidate simulator |
| MarS (microsoft/MarS, 10M and 2M order models) | Pass, on CPU, outside its supported set-up | Pass | **Fail** for real markets; workable only inside a synthetic reference market | **Conditional**: unstable for about 15 minutes after hand-over; some rollouts crash | Pass | Separate-domain case study; exploratory candidate |
| Marked Hawkes LOB (written for this study) | Pass | Pass | Pass | Pass (subcritical by construction) | Pass | Candidate simulator |

Consequence, following the plan's stop rule (section 11): the three-way
comparison on real, harmonised market data is **not** possible. There is no
licensed order-level data for any market, and MarS cannot be refitted (no
training code is released), so MarS and the other two cannot be calibrated to
a common real instrument. What is possible, and what was built:

1. a pilot on each simulator's own domain, with MarS labelled as a separate-domain case study (Experiment II);
2. the controlled reference-market experiment (Experiment III), in which the common domain is a market we generate, so ABIDES and Hawkes can be fitted to it and MarS can be conditioned on it;
3. no Experiment I against held-out historical data. Realism is measured against held-out data from the reference worlds instead.

## ABIDES

| Item | Finding | Evidence |
|---|---|---|
| Source | `jpmorganchase/abides-jpmc-public`, commit `f9cbe51` (last push 2024-07-22), BSD-3-Clause | local clone |
| Environment | Runs on Python 3.12 and NumPy 2.x. Its `pomegranate` dependency no longer installs; a 40-line stand-in samples the same order-size mixture. Results are bit-identical between Python 3.9 and 3.12 | `simbench/pomegranate_shim.py` |
| Seeded episode | Same seed gives an identical mid path, also across processes | `results/simbench/gates.json` |
| Intervention | An execution agent submits market orders; fills are reported; later prices differ from the same-seed twin | `simrel/adapters/abides.py` |
| Impact mechanism | Mechanical: orders consume the book; background agents (value, momentum, market makers) react to the resulting prices and volumes. No explicit impact parameter | code inspection |
| Snapshot / restart | Not available. Twin and intervention are separate runs with the same seed, identical until the first child order | adapter |
| Stability | Two of six configurations in the earlier pilot were unstable (price moves above 1% in 30 minutes without intervention) and one seed never terminated. Only `A1_rmsc04` and `A5_deep_mm` are used here | `results/simbench/table_observables.csv` |
| Price level | Configurable. At a mean price of 1000 cents (one tick = 10 bps, matching the reference worlds) it stays stable: 10-minute mid move s.d. 43 bps, spread 12.7 bps over 8 seeds | this audit |
| Cost | 1 to 2 s per 12-minute episode on one core | measured |

## MarS

| Item | Finding | Evidence |
|---|---|---|
| Source | `microsoft/MarS`, commit `f04dd87` (2026-09-24), MIT | local clone |
| Checkpoints | Order models of 2M, 5M and 10M parameters are public on Hugging Face (`Don-Don/mars-order-{2m,5m,10m}`, MIT, 10 to 40 MB). 19M to 1.02B are not released ("approval being requested") | Hugging Face API, 2026-10-11 |
| Assets | `Don-Don/mars-order-assets` (MIT, 72 MB): feature converters, validation samples, stylised-fact sample data. These are Python pickles | Hugging Face API |
| Supported set-up | Linux, Docker, CUDA, Python 3.11/3.12, Ray Serve. "Direct installation without Docker is not supported" | README |
| What was run | The released model loaded in-process on CPU, replacing the Ray HTTP client; nothing in the MarS repository was modified. 14 ms (2M) to 24 ms (10M) per generated order on one core | `simrel/adapters/mars_local.py` |
| Inference check | On 256 official validation samples the next-order cross-entropy is 8.17 (2M), 7.93 (5M), 7.75 (10M) against 10.80 for a uniform guess, falling with model size as in the published scaling curve | `simrel/reports/pilot_mars_validation.json` |
| Official market-impact example | Reproduced (30 min noise-agent initialisation, 5 min MarS, TWAP buy of 30% of volume, 2 rollouts each). It runs and the TWAP agent moves the market | `simrel/reports/pilot_mars_official.json` |
| Intervention API | Any agent can submit limit orders to MarS's own exchange engine (`mlib`); executions are reported back. No market-order type, so a child order is sent as a marketable limit order sized to the visible depth | `simrel/adapters/mars.py` |
| Snapshot / restart | The environment can be deep-copied at a decision point, so the twin and every policy branch from one identical state. Verified: same policy twice gives the same path | adapter smoke test |
| Stability | **Problem.** In the official example the unperturbed mid moved +336 and +590 bps in 5 minutes, and +1,815 and +3,091 bps with the TWAP buy. Over 30 unperturbed minutes (4 seeds per model): in the first 15 minutes after hand-over, 5-minute moves above 200 bps occur in 25% (10M) and 17% (2M) of windows, up to 1,964 bps, with bursts of about 6,000 orders in 5 minutes. After 15 minutes the 10M model's largest 5-minute move was 70 bps (median 23). A 20-minute burn-in is therefore used | `pilot_mars_official.json`, `pilot_mars_stability.json` |
| Failure rates measured later | Pilot volume pass: 7 of 12 (10M) and 4 of 12 (2M) unperturbed rollouts failed. Pilot main pass: 5 of 10 and 1 of 11 seeds lost to failure or the pre-task screen, plus 15 of 120 and 13 of 240 policy rollouts. Conditioned on replayed reference-world orders: 17% to 33% of unperturbed runs failed and 40% to 50% of the rest moved more than 3% in ten minutes, in all four reference worlds | `pilot_stability_audit.csv`, `reference_simulator_status.csv` |
| Crashes | A rollout can empty its own book, after which MarS raises `ValueError: unknown mid price`; replaying another market's orders into it can trip an internal assertion. The official example wraps rollouts in an error handler for this reason. Failures are counted, not hidden | pilot log |
| Domain | Trained on Chinese equities. Lot size 100, tick 0.01 on a price near 10 (10 bps). The demo's starting state is a noise agent, not market data ("production-grade applications require complete order-level historical data") | README, code |
| Refitting | **Not possible.** No training or fine-tuning code is in the repository. The only adaptation available is to change what the model is conditioned on | repository search |
| Cost | About 2 minutes per seed for a 50-minute prefix (10M); about 25 s per 12-minute policy branch | measured |

## Hawkes LOB simulator

| Item | Finding | Evidence |
|---|---|---|
| Source | Written for this study: six mutually exciting event types (limit, market, cancel, by side) with exponential kernels, size and price-level marks, state-dependent baselines, applied to a price-time-priority book | `simrel/adapters/hawkes.py`, `simrel/lob.py` |
| Why not an event generator alone | Prices, spread and queues are outcomes of matching, not inputs. Cancels scale with resting depth and market orders thin out when the spread is wide | code |
| Intervention | Injected market orders hit the same book and enter the intensities as market-order events weighted by size, so later flow responds | `HawkesLOB.on_external` |
| Twin precision | Events are obtained by thinning state-independent candidate streams, so twin and intervention share candidates and marks wherever their states agree. Impact-cost s.d. 15 bps against a 10-minute mid s.d. of 53 bps (liquid regime, 30 seeds) | this audit |
| Stability | Branching ratio 0.45 for the hand-set regimes; fitted versions are rescaled if the estimate reaches 0.98 | `simrel/tests/test_core.py` |
| Calibration | EM maximum likelihood with the simulator's own state-dependent baselines. On its own output it recovers the decay exactly, the branching ratio 0.47 against 0.445, and the excitation matrix within 0.04 | `simrel/adapters/hawkes_fit.py` |
| Cost | About 0.5 s per seed for a twin plus three policies | measured |

## Shared conventions (Gate C)

| Convention | ABIDES | MarS | Hawkes and reference worlds |
|---|---|---|---|
| Matching | Price-time priority (ABIDES-Markets) | Price-time priority (`mlib`) | Price-time priority (`simrel/lob.py`) |
| Tick relative to price | Set by the price level; 10 bps at 1000 cents, 0.1 bps at the default 100,000 | 10 bps | 10 bps |
| Lot | 1 share | 100 shares | 100 shares |
| Child orders | Market orders | Marketable limit, capped at ten visible levels | Market orders |
| Fees, latency | None; ABIDES default latency | None | None |
| Task | Sell over 10 minutes in 20 decisions; size as a fraction of the world's own mean background volume | same | same |
| Cost | Implementation shortfall in bps of the arrival mid, unfilled shares charged at the worst slice price | same | same |

Three different matching engines are in use. They implement the same priority
rule, but this is a reported confounder, not a harmonised one.

## Other audit items

| Item | Finding |
|---|---|
| LOB-Bench (`peernagy/lob_bench`) | Public, last push 2026-01-27, no licence detected by GitHub. Expects LOBSTER-format message and book files for both real and generated data. Not used: there is no real data to score against, and the realism score here is computed from the fields every adapter exposes (`simrel/metrics/realism.py`) |
| Real order-level data | None licensed. The only real data on hand is one public LOBSTER sample day (INTC, 2012-06-21), which is not a train/validation/test set |
| TRADES fallback | Not used. Its checkpoints are no longer downloadable, and the version retrained in the earlier pilot had a spread that widened with time |
| Compute | No GPU-hours needed. The pilot and the reference experiment fit in a few hours on six cores |

## Commands

```bash
# environment (Python 3.12, native arm64)
uv venv --python 3.12 ~/.venvs/mars-312
uv pip install --python ~/.venvs/mars-312/bin/python pandas==2.2.3 scipy==1.15.2 pydantic==2.10.3 pydantic-settings==2.8.1 \
    python-dotenv==1.0.1 rich==13.9.4 zstandard==0.23.0 torch==2.6.0 transformers==4.50.0 huggingface_hub matplotlib seaborn \
    tqdm ruamel.yaml==0.18.6 requests coloredlogs psutil pytest
git clone https://github.com/microsoft/MarS.git                      # commit f04dd87
git clone https://github.com/jpmorganchase/abides-jpmc-public.git    # commit f9cbe51
uv pip install --python ~/.venvs/mars-312/bin/python --no-deps -e abides-jpmc-public/abides-core -e abides-jpmc-public/abides-markets

# Phase 1 pilots (MARS = MarS checkout, DATA = an empty directory for its assets)
python simrel/experiments/pilot_mars_validation.py $MARS $DATA
python simrel/experiments/pilot_mars_official.py   $MARS $DATA 2 0.3
python simrel/experiments/pilot_mars_stability.py  $MARS $DATA 4 30
python -m pytest simrel/tests

# Phase 3 pilot and Phase 4 reference experiment
python simrel/experiments/run_pilot.py     $MARS $DATA 6
python simrel/experiments/run_reference.py $MARS $DATA 6
```
