"""Tests of the reversal-check statistics (simrel/reports/reversal_check_plan.md)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simrel.core import Task
from simrel.eventsim import run_twin_set
from simrel.metrics import realism
from simrel.metrics import reversal as RV
from simrel.policies.library import candidate_set
from simrel.reference_worlds import agent_mix

TASK = Task("sell", 9000, 600, 20, 120)


class Recording:
    """Wraps a policy and records the observations it was shown."""

    def __init__(self, policy):
        self.policy, self.obs = policy, []

    def __call__(self, obs, remaining, time_remaining, state):
        self.obs.append(obs)
        return self.policy(obs, remaining, time_remaining, state)


def test_signal_sign_as_coded():
    # sell parent: after a fall with sellers dominating the signal is negative and P6_signal_reversal sells faster
    assert RV.sell_signal(np.array([1000.0]), np.array([-800.0]), np.array([-1.0]))[0] == -0.9
    assert RV.sell_signal(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0] == 0.0
    mid = np.full(20, 1000.0)
    mid[5:] = 999.0  # the price falls during the slice before decision 5
    vol, signed = np.zeros(20), np.zeros(20)
    vol[5], signed[5] = 1000.0, -800.0
    r = RV.backtest(mid, vol, signed, 20_000)
    assert r["twap_sched"] == [1000] * 20
    assert r["p6_sched"][5] == 1900 and r["p6_sched"][4] == 1000  # 1.9 times the even pace after the fall
    assert r["p6"] == r["twap"] == 0.75 * 10.0  # the price stays where it fell: timing makes no difference
    back = mid.copy()
    back[6:] = 1000.0  # the fall reverts: having sold more at the low costs more than TWAP
    r = RV.backtest(back, vol, signed, 20_000)
    assert r["p6"] > r["twap"]
    on = mid.copy()
    on[6:] = 998.0  # the fall continues: having sold more before it is cheaper than TWAP
    r = RV.backtest(on, vol, signed, 20_000)
    assert r["p6"] < r["twap"]


def test_backtest_is_zero_without_signal_and_symmetric():
    mid = 1000.0 + np.cumsum(np.r_[0.0, np.ones(19)])
    z = np.zeros(20)
    r = RV.backtest(np.full(20, 1000.0), z, z, 10_000)
    assert r["p6"] == r["twap"] == 0.0
    a, b = RV.backtest(mid, z + 500.0, z, 10_000, "sell"), RV.backtest(mid, z + 500.0, z, 10_000, "buy")
    assert a["twap"] == -b["twap"] and a["twap"] < 0  # a rising price favours the seller's later children


def test_backtest_reproduces_the_engine_schedules():
    """Replaying the observations the engine showed to P6 gives exactly the child sizes the engine requested."""
    pol = candidate_set(0.15)
    rec6, rec0 = Recording(pol["P6_signal_reversal"]), Recording(pol["P0_twap"])
    recs = run_twin_set(lambda s: agent_mix.AgentMix(agent_mix.AMParams(), s), "t", 3, TASK, {"p6": rec6, "p0": rec0})
    for tag, name, rec in (("p6", "p6", rec6), ("twap", "p0", rec0)):
        o = rec.obs
        assert len(o) == 19
        mid = np.array([x.mid for x in o] + [recs[name].slice_mid[-1]])
        vol = np.array([x.market_volume_last for x in o] + [0.0])
        sgn = np.array([x.signed_volume_last for x in o] + [0.0])
        r = RV.backtest(mid, vol, sgn, TASK.total_qty)  # the recorded flows already include the trader's own children
        assert r[tag + "_sched"] == [int(x) for x in recs[name].slice_req], tag
    assert len(set(int(x) for x in recs["p6"].slice_req)) > 3  # P6 did tilt


def test_s1_is_the_realism_formula_and_pools_by_sums():
    rng = np.random.default_rng(0)
    mids = [1000.0 + np.cumsum(rng.normal(size=721)) for _ in range(3)]
    segs = [RV.segment_summary(m - 0.5, m + 0.5, np.zeros((0, 2)), 1.0) for m in mids]
    assert abs(RV.s1(segs[:1]) - realism._acf1(np.diff(mids[0]))) < 1e-12
    num = sum(((np.diff(m) - np.diff(m).mean())[1:] * (np.diff(m) - np.diff(m).mean())[:-1]).sum() for m in mids)
    den = sum(((np.diff(m) - np.diff(m).mean()) ** 2).sum() for m in mids)
    assert abs(RV.s1(segs) - num / den) < 1e-12
    d30 = np.diff(mids[0][::6])
    assert abs(RV.s2(segs[:1]) - realism._acf1(d30)) < 1e-12


def test_bins_windows_and_s3_on_a_constructed_segment():
    """Trades are signed by the sign of the coming 30 s move, so the signal predicts the next return exactly."""
    rng = np.random.default_rng(1)
    step = rng.choice([-1.0, 1.0], size=120)  # mid change of each 30 s bin
    m30 = 1000.0 + np.r_[0.0, np.cumsum(step)]
    mid = np.repeat(m30[:-1], 6).tolist() + [m30[-1]]
    mid = np.array(mid)  # the move of bin j happens at its end
    trades = np.array([(30.0 * j + 10.0, 300.0 * step[j + 1]) for j in range(119)])  # flow of bin j has the sign of the move of bin j + 1
    s = RV.segment_summary(mid - 0.5, mid + 0.5, trades, 1.0)
    assert s["n_bins"] == 120 and len(s["windows"]) == 6 and s["volume"] == 119 * 300.0
    w = s["windows"][1]
    assert np.array_equal(w["mid"], m30[20:40]) and w["vol"][0] == 0.0 and w["vol"][1] == 300.0 and w["signed"][1] == 300.0 * step[21]
    slope, t = RV.s3([s, s])
    assert slope > 0 and abs(RV.s3_slope([s]) - slope) < 1e-9
    ctx = RV.context([s])
    assert ctx["spread_ticks"] == 1.0 and ctx["share_one_tick"] == 1.0 and abs(ctx["tick_bp"] - 1e4 / s["mean_mid"]) < 1e-9
    assert RV.parent_qty([s]) == int(0.15 * np.mean([x["window_volume"] for x in s["windows"]])) // 100 * 100


def test_missing_quotes_drop_the_window_and_the_returns_that_touch_them():
    mid = 1000.0 + np.zeros(721)
    bid, ask = mid - 0.5, mid + 0.5
    bid[100] = np.nan  # inside the first window, not at a decision time: only 5 s returns are affected
    bid[126] = np.nan  # a decision time of the second window
    s = RV.segment_summary(bid, ask, np.zeros((0, 2)), 1.0)
    assert s["n_missing"] == 2 and len(s["windows"]) == 5
