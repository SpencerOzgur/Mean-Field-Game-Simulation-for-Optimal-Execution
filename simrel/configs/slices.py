"""Configuration of the scenario-slice experiment on real Nasdaq order data.

Everything in the "decision rule" block was fixed before any policy was run in any simulator.
"""

import os

from simrel.core import Task

# The symbol, the data folder and the order size can be set from the environment, so one code base runs the INTC study,
# the confirmatory CSCO and MSFT replications and other order sizes. INTC at 10% is the default.
DATA_DIR = os.environ.get("SIMREL_DATA_DIR", "~/simdata/itch")
SYMBOL = os.environ.get("SIMREL_SYMBOL", "INTC")
FIT_DAYS = ["2019-01-30", "2019-03-27", "2019-07-30", "2019-08-30", "2019-10-18"]  # simulators are fitted to these
TEST_DAYS = ["2019-10-30", "2019-12-30", "2020-01-30"]  # realism is scored against these
WINDOW_S = 600

# slices: every combination of high / low on three window descriptors, split at the median over all windows
SLICE_FEATURES = {"volatility_bps": "vol", "volume_visible": "volu", "trend_bps": "trend"}
MIN_WINDOWS = 4  # a slice needs at least this many fit windows and this many test windows

TASK = Task(side="sell", horizon_s=WINDOW_S, n_slices=20, post_s=60)
SIZE_FRAC = float(os.environ.get("SIMREL_SIZE_FRAC", "0.10"))  # parent order as a fraction of the slice's mean executed volume per window
SIZE_TAG = "" if abs(SIZE_FRAC - 0.10) < 1e-9 else f"_s{int(round(SIZE_FRAC * 100))}"  # appended to policy result files
POLICY_SUBSET_ONLY = os.environ.get("SIMREL_POLICY_SUBSET") == "1"  # run only VERSION_POLICIES, for every version
FAST_HAWKES = os.environ.get("SIMREL_FAST_HAWKES") == "1"  # every Hawkes version on the compiled simulator
N_POLICY_SEEDS = 60  # ABIDES was run with the first 20; Hawkes with all 60 (its costs are noisier)
N_GEN = 12  # unperturbed simulated windows per slice and simulator for the realism score
MAX_SCORE_WINDOWS = 12  # test windows used per slice for the realism score (a fixed random subset when there are more)
HAWKES_WARMUP_S, ABIDES_WARMUP_S = 600, 1200

# ---- decision rule (fixed in advance)
# Realism is LOB-Bench's mean L1 distance over its 21 default statistics: 0 = identical distributions, 1 = disjoint.
REALISM_CEILING = 0.25  # a simulator is "realistic enough" in a slice at or below this
SIMILAR_BAND = 0.05  # two simulators have "similar realism" in a slice if their scores differ by at most this
# A simulator's best policy is the one with the lowest mean implementation shortfall. Two simulators "disagree" in a
# slice when each one's best policy is significantly worse than the other's best inside the other simulator
# (paired difference over seeds, 95% bootstrap interval above zero).

HAWKES_BETAS = (2.0, 10.0)  # decay rates tried, per second (faster decays fit better but make simulation too slow)
HAWKES_WEIGHT = {"weight_exponent": 0.5, "weight_cap": 3.0}
HAWKES_MAX_FIT_WINDOWS = 12

ABIDES_GRID = {
    "num_value_agents": [100, 300, 1000],
    "mm": [(0, 0.025, "60s"), (2, 0.025, "60s"), (2, 0.1, "10s"), (4, 0.25, "10s"), (8, 0.5, "5s")],  # (market makers, participation, wake-up interval)
    "fund_vol": [1e-5, 5e-5, 5e-4, 2e-3],
}
ABIDES_CALIB_SEEDS = 4
CALIB_MOMENTS = ["volatility_bps", "spread_bps", "volume_visible", "events", "depth10", "trend_bps"]

# ---- simulator versions (each realism fix is kept as its own version, so its effect on decisions can be measured)
# A version name is "<family>" or "<family>_<suffix>"; the base versions "hawkes" and "abides" ran all twelve policies.
# same_day  start from the book at the start of a test-day window of the slice, and anchor depth to it
# queue     cancels at the best quote scale with the best queue, cancels behind it with the depth behind it
# close     while the spread is two ticks or more, quote-improving limit orders arrive at the rate seen in the data
HAWKES_VERSIONS = {
    "hawkes": {},
    "hawkes_sd": {"same_day": True},
    "hawkes_q": {"queue": True},
    "hawkes_c": {"close": True},
    "hawkes_sd_q": {"same_day": True, "queue": True},
    "hawkes_sd_c": {"same_day": True, "close": True},
    "hawkes_q_c": {"queue": True, "close": True},
    "hawkes_sd_q_c": {"same_day": True, "queue": True, "close": True},
    # Added after the price-dynamics check. size: market orders sized relative to the best queue they hit (tried on one
    # slice; it doubled volume and did not reduce volatility, so it was not run further). fill: gaps between the best
    # and second price level are filled at FILL_RATE (the INTC book never shows such a gap).
    "hawkes_s": {"size": True},
    "hawkes_f": {"fill": True},
    "hawkes_sd_c_f": {"same_day": True, "close": True, "fill": True},
    "hawkes_q_f": {"queue": True, "fill": True},
    "hawkes_sd_q_c_f": {"same_day": True, "queue": True, "close": True, "fill": True},
    # Runs made on 2026-10-27 as hawkes_q_f and hawkes_sd_q_c_f had the queue-relative sizes switched on by mistake
    # (a refit stored the ratio table, and the runner passed it to every queue version). They are kept under these names.
    "hawkes_q_s_f": {"queue": True, "size": True, "fill": True},
    "hawkes_sd_q_c_s_f": {"same_day": True, "queue": True, "close": True, "size": True, "fill": True},
}
FILL_RATE = 200.0  # per second and side; effectively immediate on the 5-second scale of the analysis

# Multiple timescales (sum of exponential kernels with these decays, per second), simulated with the compiled loop
# (simrel/adapters/hawkes_fast.py). Impact calibration: the excitation caused by market orders is multiplied by a
# per-slice factor chosen on the FIT days to match LOB-Bench's impact-response curves; it is never chosen on test days.
MULTI_BETAS = (1.0, 10.0, 100.0)
IMPACT_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
HAWKES_VERSIONS.update({
    "hawkes_m": {"multi": True},
    "hawkes_sd_c_f_m": {"same_day": True, "close": True, "fill": True, "multi": True},
    "hawkes_sd_q_c_f_m": {"same_day": True, "queue": True, "close": True, "fill": True, "multi": True},
    "hawkes_sd_c_f_k": {"same_day": True, "close": True, "fill": True, "impact": True},
    "hawkes_sd_q_c_f_k": {"same_day": True, "queue": True, "close": True, "fill": True, "impact": True},
    "hawkes_sd_c_f_m_k": {"same_day": True, "close": True, "fill": True, "multi": True, "impact": True},
})
ABIDES_MOMENTUM_GRID = (12, 50, 150)  # momentum traders tried for ABIDES-calibrated, chosen on fit-day impact
VERSION_POLICIES = ["P0_twap", "P1_immediate", "P5_liquidity", "P6_signal_momentum"]  # the subset run for non-base versions
N_VERSION_SEEDS = 20

# ---- full realism evaluation (fixed on 2026-10-27, before any result of the gap-filling versions existed)
# Realism is reported as a profile of four groups, each scored against the same held-out real windows:
#   order_book   LOB-Bench's 21 default unconditional statistics (DEFAULT_SCORING_CONFIG), mean L1
#   conditional  LOB-Bench's conditional statistics (DEFAULT_SCORING_CONFIG_COND), mean L1, except "spread | time":
#                simulated windows carry an arbitrary clock time, so time of day is not comparable
#   price        five return and volatility statistics built from LOB-Bench's own functions, mean Wasserstein
#                distance (its automatic binning puts nearly all 5-second returns in one bin, which makes L1 uninformative)
#   impact       LOB-Bench's impact-response evaluation (impact.impact_compare), its Mean_Diffs in ticks
# No group is averaged into another; a simulator is "realistic" on a group, not overall.
REALISM_GROUPS = ["order_book", "conditional", "price", "impact"]
EXCLUDED_CONDITIONAL = ["spread | time"]

# ABIDES-calibrated ("abides_quoting"): the slice's ABIDES configuration plus quoting agents calibrated to the fit days
ABIDES_QUOTING_AGENTS = 50
# versions whose impact is calibrated on fit days (slice_08_impact_calibration.py); the calibrated version adds "_k"
IMPACT_CALIBRATE = ["hawkes_sd_c_f", "hawkes_sd_q_c_f", "hawkes_sd_c_f_m"]
N_CALIB_WINDOWS = 6
MAX_CALIB_BRANCHING = 0.97  # impact scales that push the branching ratio to this or above are excluded before simulating

# ---- robustness to fitting and tuning choices (part C of the validation)
# Hawkes refitted with one fit day left out ("fit_tag" selects those fits), for the original and the best version;
# ABIDES with the next-best calibration configurations (rank 1 = second best by fit-day moment error).
LODO_DAYS = list(range(len(FIT_DAYS)))
for _j in LODO_DAYS:
    HAWKES_VERSIONS[f"hawkes__lodo{_j}"] = {"fit_tag": f"_lodo{_j}"}
    HAWKES_VERSIONS[f"hawkes_sd_c_f_m__lodo{_j}"] = {"same_day": True, "close": True, "fill": True, "multi": True, "fit_tag": f"_lodo{_j}"}
HAWKES_VERSIONS["hawkes__fast"] = {"fast": True}  # control: the original fit on the compiled simulator, like the refits
ABIDES_ALTERNATIVES = ["abides__alt1", "abides__alt2", "abides_quoting__alt1"]

# ---- last robustness checks (exploratory, added 31 October 2026 after the CSCO confirmatory analysis)
# Hawkes with the branching-ratio cap at 0.8 or 0.95 instead of 0.9 (refits on the same windows, "fit_tag" selects them)
for _cap in ("0.8", "0.95"):
    HAWKES_VERSIONS[f"hawkes__cap{_cap}"] = {"fit_tag": f"_cap{_cap}"}
    HAWKES_VERSIONS[f"hawkes_sd_c_f_m__cap{_cap}"] = {"same_day": True, "close": True, "fill": True, "multi": True, "fit_tag": f"_cap{_cap}"}
# ABIDES-calibrated with its fundamental volatility and momentum traders recalibrated so that its 5-second volatility
# matches the fit days ("abides_quoting_vol"); the grid below is searched per slice.
ABIDES_VOL_GRID = {"fund_vol": [5e-4, 2e-3, 5e-3], "momentum": [12, 150]}
ABIDES_VOL_SEEDS = 3
