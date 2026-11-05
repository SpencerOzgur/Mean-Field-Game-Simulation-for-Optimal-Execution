"""Sign conventions, harness constraints, matching engine, and policy causality."""

import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simrel.adapters.hawkes import HawkesLOB, branching_ratio, regimes
from simrel.core import Observation, Task, child_quantity
from simrel.eventsim import run_twin_set
from simrel.lob import Book
from simrel.metrics.cost import episode_metrics, shortfall_bps
from simrel.policies.library import ExpSchedule, candidate_set


def make_obs(k, n=20, side="sell", **kw):
    base = dict(k=k, n=n, t=30.0 * k, bid=999.0, ask=1001.0, bid_depth1=500.0, ask_depth1=500.0, bid_depth5=3000.0, ask_depth5=3000.0,
                market_volume_last=4000.0, signed_volume_last=-800.0, mid_change_last=-1.0, arrival_mid=1000.0, total_qty=10000, side=side)
    base.update(kw)
    return Observation(**base)


# ---- sign conventions
def test_sell_below_arrival_is_a_positive_cost():
    cost, unfilled = shortfall_bps("sell", 1000.0, np.array([100.0]), np.array([99_900.0]), 100)
    assert unfilled == 0.0 and abs(cost - 10.0) < 1e-9


def test_buy_above_arrival_is_a_positive_cost():
    cost, _ = shortfall_bps("buy", 1000.0, np.array([100.0]), np.array([100_100.0]), 100)
    assert abs(cost - 10.0) < 1e-9


def test_selling_above_arrival_is_a_negative_cost():
    cost, _ = shortfall_bps("sell", 1000.0, np.array([100.0]), np.array([100_100.0]), 100)
    assert abs(cost + 10.0) < 1e-9


def test_unfilled_shares_are_charged_at_the_worst_slice():
    # half filled: 50 @ 999 (10 bps) and 0 afterwards; unfilled half is charged the worst slice cost (10 bps)
    cost, unfilled = shortfall_bps("sell", 1000.0, np.array([25.0, 25.0]), np.array([25 * 999.5, 25 * 999.0]), 100)
    assert abs(unfilled - 0.5) < 1e-12
    assert abs(cost - (0.5 * 7.5 + 0.5 * 10.0)) < 1e-9


# ---- harness constraints
def test_child_order_is_clipped_rounded_and_completed():
    greedy = lambda o, rem, tr, st: 1e9
    negative = lambda o, rem, tr, st: -50
    assert child_quantity(greedy, make_obs(3), 1234, {}, lot=100) == 1200
    assert child_quantity(negative, make_obs(3), 1234, {}, lot=100) == 0
    assert child_quantity(negative, make_obs(19), 1234, {}, lot=100) == 1234  # last slice sends the remainder
    assert child_quantity(lambda *a: float("nan"), make_obs(3), 1234, {}, lot=100) == 0


def test_every_policy_completes_the_parent_order():
    for name, pol in candidate_set(0.1).items():
        remaining, state = 10_000, {}
        for k in range(20):
            q = child_quantity(pol, make_obs(k), remaining, state, lot=100)
            assert 0 <= q <= remaining, name
            remaining -= q
        assert remaining == 0, name


def test_twap_is_equal_slices():
    remaining, state, sizes = 10_000, {}, []
    for k in range(20):
        q = child_quantity(ExpSchedule(0.0), make_obs(k), remaining, state, lot=100)
        sizes.append(q)
        remaining -= q
    assert sizes == [500] * 20


# ---- policy causality: decisions up to time k do not depend on anything after k
def test_policies_do_not_depend_on_later_observations():
    rng = np.random.default_rng(0)

    def path(perturb_after):
        seq = []
        for k in range(20):
            kw = dict(bid=999.0 - k, ask=1001.0 - k, market_volume_last=4000.0 + 10 * k, signed_volume_last=-500.0, mid_change_last=-1.0)
            if k > perturb_after:
                kw = dict(bid=900.0 + rng.normal(), ask=905.0 + rng.normal(), market_volume_last=9e4, signed_volume_last=5e4, mid_change_last=7.0,
                          bid_depth5=10.0, ask_depth5=10.0)
            seq.append(make_obs(k, **kw))
        return seq

    for name, pol in candidate_set(0.1).items():
        a, b = path(perturb_after=99), path(perturb_after=9)
        out = []
        for seq in (a, b):
            remaining, state, qs = 10_000, {}, []
            for k in range(10):  # only the shared prefix
                q = child_quantity(pol, seq[k], remaining, state, lot=100)
                qs.append(q)
                remaining -= q
            out.append(qs)
        assert out[0] == out[1], name


def test_observation_is_immutable():
    o = make_obs(0)
    try:
        o.bid = 1.0
        raise AssertionError("observation should be frozen")
    except dataclasses.FrozenInstanceError:
        pass


# ---- matching engine
def test_book_price_time_priority_and_conservation():
    b = Book()
    b.limit(True, 99, 100, owner="first")
    b.limit(True, 99, 100, owner="second")
    b.limit(True, 98, 300)
    fills = b.market(False, 150)
    assert fills == [(99, 100, "first"), (99, 50, "second")]
    assert b.level_qty("bid", 99) == 50 and b.depth("bid", 5) == 350
    fills, oid = b.limit(False, 98, 500)  # marketable sell limit: fills at resting prices, remainder rests at 98
    assert [f[:2] for f in fills] == [(99, 50), (98, 300)] and oid is not None
    assert b.best_bid() is None and b.best_ask() == 98 and b.level_qty("ask", 98) == 150


def test_market_order_fills_at_or_better_than_touch_for_the_passive_side():
    b = Book()
    for k in range(5):
        b.limit(True, 100 - k, 100)
    fills = b.market(False, 250)
    assert [f[0] for f in fills] == [100, 99, 98]


# ---- Hawkes simulator
def test_hawkes_regimes_are_subcritical():
    for p in regimes().values():
        assert branching_ratio(p.alpha) < 1.0


def test_hawkes_twin_is_identical_before_the_task_and_reacts_after():
    P = regimes()["H1_liquid"]
    out = run_twin_set(lambda s: HawkesLOB(P, s), "H1", 0, Task(total_qty=8000), {"twap": ExpSchedule(0.0), "again": ExpSchedule(0.0)}, warmup_s=300)
    twin, twap = out["none"], out["twap"]
    pre = twin.grid_t < 0
    assert np.array_equal(twin.grid_mid[pre], twap.grid_mid[pre])
    assert twap.filled == 8000 and twin.filled == 0
    assert not np.array_equal(twin.grid_mid[~pre], twap.grid_mid[~pre])
    assert np.array_equal(twap.grid_mid, out["again"].grid_mid)  # same seed, same policy, same path
    m = episode_metrics(twap, twin)
    assert m["impact_bps"] > 0  # a market sell pays at least the half spread against the twin's mid
