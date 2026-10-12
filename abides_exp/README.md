# ABIDES experiments

Tools for testing trading strategies and market events in
[ABIDES](https://github.com/jpmorganchase/abides-jpmc-public), JPMorgan's agent-based
limit order book simulator, and for adding permanent price impact to it.

Everything here measures effects the same way: each experiment is run twice with the
same random seed, once **treated** (strategy or event switched on) and once as a
**baseline** (same agents, switched off). The difference between the two runs is the
effect of the treatment. Results are averaged over many seeds with 95% confidence
intervals.

---

## Contents

1. [Setup](#1-setup)
2. [Quick start](#2-quick-start)
3. [How the experiments work](#3-how-the-experiments-work)
4. [The market](#4-the-market)
5. [Scenarios](#5-scenarios)
6. [Permanent impact (lambda)](#6-permanent-impact-lambda)
7. [Commands](#7-commands)
8. [Outputs and metrics](#8-outputs-and-metrics)
9. [Extending](#9-extending)
10. [Known issues and caveats](#10-known-issues-and-caveats)
11. [Files](#11-files)

---

## 1. Setup

ABIDES needs **Python 3.11** with **numpy < 2** and **pandas 1.5.3**. Use a separate
virtual environment; the main MFG code in `src/` runs fine in it too.

From the repository root:

```bash
# 1. Python 3.11 environment
python3.11 -m venv .venv-abides          # macOS: brew install python@3.11 if missing
source .venv-abides/bin/activate
python --version                         # must print 3.11.x
python -m pip install --upgrade pip

# 2. ABIDES source (gitignored, lives in external/)
mkdir -p external
git clone --depth 1 https://github.com/jpmorganchase/abides-jpmc-public.git external/abides-jpmc

# 3. Replace the pomegranate dependency (pomegranate 0.14 does not build on modern Python).
#    Same mixture distribution, sampled with numpy.
python - <<'EOF'
p = "external/abides-jpmc/abides-markets/abides_markets/models/order_size_model.py"
s = open(p).read()
s = s.replace("from pomegranate import GeneralMixtureModel\n", "")
s = s[: s.index("class OrderSizeModel")] + '''class OrderSizeModel:
    """Numpy replacement for the pomegranate GeneralMixtureModel (same mixture)."""

    def __init__(self) -> None:
        w = np.array(order_size["weights"], dtype=float)
        self.weights = w / w.sum()
        self.dists = [(d["name"], d["parameters"]) for d in order_size["distributions"]]

    def sample(self, random_state: np.random.RandomState) -> float:
        k = random_state.choice(len(self.dists), p=self.weights)
        name, (a, b) = self.dists[k]
        if name == "LogNormalDistribution":
            return round(random_state.lognormal(a, b))
        return round(random_state.normal(a, b))
'''
open(p, "w").write(s)
print("patched", p)
EOF

# 4. Dependencies (the pins matter: numpy 2 and pandas 2+ break ABIDES)
pip install -r requirements-abides.txt
pip install --no-deps -e external/abides-jpmc/abides-core -e external/abides-jpmc/abides-markets

# 5. Check
python abides_exp/smoke_test.py 2>&1 | tail -2
```

A working install prints `agents: 1117, L1 snapshots: 5060, median bid: 999.31` and
`ABIDES OK`. Results reproduce exactly across machines for a given seed.

**PyCharm:** open the repository root as its own project and set the interpreter to
`.venv-abides/bin/python`. Mark `src/` as Sources Root and `log/` as Excluded.

---

## 2. Quick start

All commands run from the repository root inside `.venv-abides`.

```bash
# One TWAP sale vs its baseline, printed to the terminal (about 15 s)
python abides_exp/example_twap.py

# Impact of a TWAP sale at several sizes, 30 seeds
python -m abides_exp.harness --seeds 30 --pct 0.01 0.05 0.10 0.20 --workers 8 --out results/impact_sweep
python -m abides_exp.analyze_impact results/impact_sweep

# All 10 scenarios, 50 seeds each
python -m abides_exp.battery --seeds 50 --workers 8 --out results/battery_50

# Permanent-impact sweep over lambda
python -m abides_exp.battery --lambdas 0 1 5 20 --seeds 50 --workers 8 \
  --only exec_twap exec_ac exec_pov exec_passive spoofing momentum_ignition predatory flash_crash \
  --out results/lambda_sweep
```

Rough run times on an 8-core laptop: about 1 second per treated/baseline pair, so
500 pairs take about 9 minutes.

---

## 3. How the experiments work

**Same-seed pairs.** A seed fixes one simulated market day: the same traders, the same
random orders, the same price path. Each experiment plays that day twice:

| Run | What happens |
|---|---|
| Baseline | The day as it would have happened anyway |
| Treated | The same day with one change (a seller, a manipulator, news, ...) |

Before the change starts the two runs are identical tick for tick. After it, any
difference is caused by the change. Because both runs share the same random path,
most of the market's own noise cancels: this is about 5 times more precise than
comparing a sale price with the arrival price.

**The baseline always contains the same agents, inactive.** Adding an agent to ABIDES
rebuilds its latency model and consumes random numbers, so a baseline without the
agent would be a different market day. Execution agents in a baseline trade 0
shares; scripted agents are created with `active=False`.

**Built-in checks, on every pair:**

- `pre_start_max_abs_bps` must be exactly 0: the runs agree before the window starts.
- `pnl_sum_check` must be about 0: money only moves between agents.

The tools print a warning if either check fails.

**Many seeds.** After the treatment starts, the two runs drift apart partly by chance
(the market is chaotic). Single runs mean little; use 50 or more seeds per result.

---

## 4. The market

The stock `rmsc04` configuration from ABIDES, simulated 09:30 to 11:00 by default.

| Count | Agent | Behaviour |
|---|---|---|
| 1 | Exchange | Order book and matching, price-time priority, prices in integer cents |
| 1,000 | Noise traders | Each trades once at a random time (09:00 to 16:00), random side, at the best price |
| 102 | Value traders | About every 3 minutes, get a noisy look at the true value and trade toward it; 90% of the time with a resting limit order |
| 2 | Market makers | About every minute, post 10 price levels on each side, sized to about 2.5% of recent volume |
| 12 | Momentum traders | About every 37 s, buy if the 20-period average price is above the 50-period one, else sell |

The true value (the "fundamental") follows a mean-reverting random process and, in
default ABIDES, ignores trading. Value traders anchor the price to it and supply most
of the liquidity. All of these settings can be changed with `--market` (section 7).

---

## 5. Scenarios

Times are relative to the 09:30 open. Defined in `scenarios.py`; list them with
`python -m abides_exp.battery --list`.

### Execution strategies: sell 12,000 shares (about 20% of window volume), 09:40 to 10:10

Baseline for each: the same strategy with 0 shares (it wakes and reads market data on
the same schedule but never trades).

| Scenario | Treated run |
|---|---|
| `exec_twap` | 30 equal market orders, one per minute |
| `exec_ac` | Almgren-Chriss trajectory with kappa*T = 3: front-loaded, about 64% sold in the first 10 minutes |
| `exec_pov` | Each minute, sells 20% of the previous minute's market volume (does not force completion) |
| `exec_passive` | Each minute, rests a limit order at the best ask for the shares due so far; market-sells any remainder at the last minute |

### Adversaries

Baseline: the same scripted agent with `active=False`.

| Scenario | Treated run |
|---|---|
| `spoofing` | 09:40 to 10:10, every 30 s: a 5,000-share fake order one tick behind the touch, then a real 300-share market order the other way 15 s later and a cancel. Sides alternate so inventory stays flat |
| `momentum_ignition` | From 09:40: buys 3,000 shares in 30 s, waits 60 s, sells them back over 10 min |
| `predatory` | A TWAP seller of 12,000 shares is present in **both** runs (the victim). The predator, treated only, sells 3,000 shares in the victim's first 2 minutes and buys them back over its last 6 |

### Market events (from 10:00)

| Scenario | Baseline | Treated run |
|---|---|---|
| `fundamental_jump` | No change | True value drops 50 bps at 10:00 |
| `mm_withdrawal` | Market makers quote normally | Market makers stop quoting and cancel for 10 minutes |
| `flash_crash` | No fire seller, market makers normal | A fire seller dumps 18,000 shares over 2 minutes (resending unfilled shares) while market makers withdraw for 5 minutes |

---

## 6. Permanent impact (lambda)

Default ABIDES has no permanent price impact: the fundamental ignores trading, so every
price move fades. The impact oracle adds it:

> Every fill of one of **our** agents (execution strategies and scripted traders) moves
> the fundamental by **lambda bps per 10,000 shares** of net flow, permanently.
> Value traders then carry the price to the new level.

This is the S = F + lambda * (cumulative flow) relation of the MFG model.

- `lambda = 0` reproduces the default market exactly.
- Example: `lambda = 5` and a 12,000-share sale lower the fundamental by 6 bps for good.
- Background agents' trades do not move the fundamental.
- It is applied to observations of the fundamental, so no random draws change and the
  same-seed design still holds.

Use it in any tool through the market options:

```bash
python -m abides_exp.harness --seeds 30 --pct 0.2 --market impact_lambda=5 --out results/twap_l5
python -m abides_exp.battery --seeds 50 --market impact_lambda=5 --out results/battery_l5
python -m abides_exp.battery --lambdas 0 1 5 20 --seeds 50 --out results/lambda_sweep
```

Under linear permanent impact, a 12,000-share sale should cost about lambda * 0.6 bps
more than at lambda = 0 and leave the price lambda * 1.2 bps lower. The sweep figure
draws these theory lines for comparison.

`fundamental_jump` and `mm_withdrawal` contain none of our agents, so lambda does not
affect them; leave them out of sweeps.

---

## 7. Commands

### `harness.py`: one execution strategy, many sizes and seeds

```bash
python -m abides_exp.harness --strategy twap --seeds 100 \
  --pct 0.01 0.025 0.05 0.10 0.20 0.40 --workers 8 --out results/impact_sweep
```

| Option | Meaning | Default |
|---|---|---|
| `--strategy` | `twap`, `ac`, `pov`, `passive` | `twap` |
| `--seeds`, `--seed-offset` | Seeds `offset .. offset+N-1` | 30, 0 |
| `--pct` or `--qty` | Sizes as a fraction of the market's volume in the window, or in shares | required |
| `--side` | `sell` or `buy` | `sell` |
| `--start`, `--duration`, `--slices` | Delay after the open, window length, number of slices | `00:10:00`, `00:30:00`, 30 |
| `--end-time` | Simulation end | `11:00:00` |
| `--market KEY=VALUE ...` | Market overrides (below) | none |
| `--workers`, `--out` | Parallel processes, output folder | CPUs - 1, `results/sweep` |

Phase 1 runs one baseline per seed and measures the window volume, which turns `--pct`
into shares. Phase 2 runs every (seed, size) pair. `--pct 0.2` means 20% of baseline
window volume, not a participation rate.

### `battery.py`: the scenario battery

```bash
python -m abides_exp.battery --seeds 50 --workers 8 --out results/battery_50
python -m abides_exp.battery --only spoofing flash_crash --seeds 100 --out results/battery_deep
python -m abides_exp.battery --lambdas 0 1 5 20 --seeds 50 --out results/lambda_sweep
python -m abides_exp.battery --list
```

| Option | Meaning |
|---|---|
| `--only NAME ...` | Run a subset of scenarios |
| `--seeds`, `--seed-offset` | As above (default 20 seeds) |
| `--market KEY=VALUE ...` | Market overrides |
| `--lambdas L ...` | Run the whole battery once per lambda, into `<out>/lambda_<L>/`, plus a combined sweep summary and figure |
| `--workers`, `--out` | Parallel processes, output folder (default `results/battery`) |

**Always give each run its own `--out`.** A second run into the same folder overwrites
its `summary.csv`, `battery.png` and `meta.json`.

### Market overrides (`--market`)

Any argument of `abides_markets.configs.rmsc04.build_config`, plus `impact_lambda`.
Examples:

```bash
--market num_value_agents=50          # fewer value traders
--market num_momentum_agents=40       # more trend followers
--market mm_pov=0.01                  # smaller market makers
--market mm_skew_beta=0.5             # market makers react to book imbalance
--market impact_lambda=5              # permanent impact
```

Note: removing value traders breaks the book (partial fills, huge price jumps) rather
than making the market gradually thinner, because they supply most of the liquidity.

### Analysis helpers

```bash
python -m abides_exp.analyze_impact results/impact_sweep     # impact_curve.png, impact_summary.csv
python -m abides_exp.rebuild_summary results/battery         # rebuild summary.csv and battery.png from scenario folders
python abides_exp/example_twap.py                            # walkthrough of a custom agent
python abides_exp/smoke_test.py                              # install check
```

---

## 8. Outputs and metrics

### Harness (`--out` folder)

| File | Contents |
|---|---|
| `runs.csv` | One row per (seed, size): fills, VWAP, shortfall, impact summary, check column |
| `paths.csv` | Impact path per run on a 30 s grid: `seed, qty, t_rel_s, impact_bps` |
| `baselines.csv` | One row per seed: window volume, arrival price |
| `meta.json` | All settings and the calibrated volume |

`analyze_impact.py` adds `impact_curve.png` and `impact_summary.csv`.

### Battery (`--out` folder)

| File | Contents |
|---|---|
| `<scenario>/runs.csv` | One row per seed with every metric below |
| `<scenario>/paths.csv` | Paths on a 10 s grid: `dmid_bps`, spreads, mispricing (treated and baseline) |
| `summary.csv` | Mean, 95% CI and n per scenario and metric |
| `battery.png` | Mean price effect vs baseline per scenario, with 95% bands |
| `meta.json` | Seeds, market settings, scenario parameters |
| `sweep_summary.csv`, `lambda_sweep.png` | Lambda sweeps only: summaries across lambda |

### Metrics

Sign convention: impact and shortfall are **positive when the price moves against the
trader**. `dmid` is treated minus baseline mid price, in bps of the baseline mid at the
window start, so negative means the treatment pushed the price down.

| Metric | Meaning |
|---|---|
| `dmid_mean_bps`, `dmid_min_bps`, `dmid_max_bps` | Price effect during the window |
| `dmid_end_bps`, `dmid_post30_bps` | Price effect at the window end, and averaged over the 30 min after (lasting effect) |
| `spread_ratio`, `vol_ratio` | Spread and 10 s return volatility, treated / baseline, in the window |
| `one_sided_t`, `one_sided_b` | Share of the window with an empty bid or ask side |
| `mispricing_t_bps`, `mispricing_b_bps` | Mean distance of price from the true fundamental |
| `pnl_<AgentType>` | Change in profit and loss of each agent type, in dollars, marked at the final fundamental; sums to about 0 |
| `subject_shortfall_bps`, `subject_fill_rate` | Execution strategies: cost against the arrival mid, share filled |
| `subject_pnl_usd`, `subject_market_fill_rate` | Scripted agents: trading profit, share of market orders filled |
| `victim_extra_cost_bps` | Predatory: victim's shortfall in the treated run minus the baseline |
| `spoof_push_bps` | Spoofing: price move toward the fake order before the real trade |
| `time_to_50pct_s`, `time_to_90pct_s`, `settled_gap_bps` | News jump: speed of price discovery, and final level vs the new value |
| `trough_bps`, `half_recovery_s` | Flash crash: lowest point and recovery time |
| `pre_start_max_abs_bps`, `pnl_sum_check` | Checks (section 3) |

Prefer paired metrics (`dmid_*`, `victim_extra_cost_bps`, or shortfall differences
between strategies on the same seed) over shortfall against the arrival price, which
mostly measures the market's own drift. `impact_peak_bps` (harness) and the minimum- or
first-passage metrics are biased upward by noise; use means where possible.

---

## 9. Extending

### A new execution strategy (for example the MFG agent)

Subclass `ExecutionAgent` in `agents/execution.py`. Override `slice_qty` to choose the
market-order size each slice, or `act` to manage limit orders. Set `needs_volume = True`
to get last-interval market volume in `self.transacted_volume`.

Sketch:

```python
class MFGAgent(ExecutionAgent):
    def __init__(self, *args, a=1.0, phi=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.a, self.phi = a, phi

    def slice_qty(self, current_time, bid, ask) -> int:
        frac = self.slices_sent / self.n_slices   # time elapsed in the window
        ...                                       # your control, in shares this slice
        return shares
```

Register it in `agents/__init__.py` (`STRATEGIES["mfg"] = MFGAgent`). It then works with
`harness.py --strategy mfg`. For the battery, add `"exec_mfg": make_exec("mfg", a=..., phi=...)`
to `SCENARIOS` in `scenarios.py`. Fills are reported to the impact oracle automatically.

### A new adversary or order-flow event

Write a script of timed actions for `ScriptedTrader` (`agents/scripted.py`). Actions are
`market`, `limit` (with `ticks` from the touch), `cancel_all`, and `market_to`
(cumulative target, resends unfilled shares). Then add a scenario builder in
`scenarios.py` following `spoofing` or `momentum_ignition`, with `active=treated`.

### A new market event

Add it to `events.py` and apply it in a scenario only when `treated` is true. Keep it
from consuming random numbers (see `shift_fundamental` and `withdraw_market_makers`),
otherwise the baseline stops matching.

---

## 10. Known issues and caveats

- **Impact lag (lambda > 0).** The fundamental moves the moment our agent trades, but the
  price only follows as value traders notice, over a few minutes. Round-trip strategies
  can therefore sell before the price drops and buy back after, earning back part of
  their own impact. This explains most of the improvement of the spoofer and igniter at
  high lambda; treat those numbers as upper bounds until own-trade impact is applied
  immediately.
- **Market orders on an empty side are dropped.** ABIDES discards the unfilled part of a
  market order when the opposite side of the book empties; it does not push the price
  lower. A large fire sale therefore empties the book instead of crashing the price.
- **Market maker withdrawal lags** by up to about a minute (it takes effect at each
  market maker's next action).
- **Single market, morning only, sells only.** All results so far use rmsc04 defaults,
  09:30 to 11:00, and sell orders.
- **Log folders.** ABIDES writes a small summary log per simulation. The tools give each
  run its own folder under `log/runs/` to avoid a race between parallel workers. Clean
  up with `rm -rf log/*`; `log/` is gitignored.
- **Python version.** Must be 3.11 (numpy < 2 has no wheels for 3.13). If `pip` tries to
  build pandas from source, you are in the wrong environment.

---

## 11. Files

| File | Purpose |
|---|---|
| `agents/execution.py` | `ExecutionAgent` base class; `TWAPAgent`, `AlmgrenChrissAgent`, `POVAgent`, `PassiveThenCrossAgent` |
| `agents/scripted.py` | `ScriptedTrader` and script builders for spoofing, ignition, predatory trading, fire sales |
| `agents/__init__.py` | `STRATEGIES` registry |
| `events.py` | `ShiftedOracle` (news jumps and permanent impact), market maker withdrawal |
| `scenarios.py` | The 10 scenarios, each with its treated and baseline version |
| `harness.py` | Size sweeps for one execution strategy |
| `battery.py` | Scenario battery and lambda sweeps |
| `analyze_impact.py` | Impact curve figure and summary from a harness run |
| `rebuild_summary.py` | Rebuild a battery's summary and figure from its scenario folders |
| `example_twap.py` | Minimal walkthrough of a custom agent |
| `smoke_test.py` | Install check |
