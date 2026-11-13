"""Tests of the permanent-impact (lambda) agent-mix family (development pilot)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simrel.core import Task
from simrel.eventsim import GRID_DT, run_twin_set
from simrel.lob import Book
from simrel.policies.library import candidate_set
from simrel.reference_worlds import lambda_world as LW
from simrel.reference_worlds import sampled
from simrel.reference_worlds.agent_mix import AgentMix

TASK = Task("sell", 6000, 600, 20, 120)
VALS = LW.pilot_draws()["XLP_pilot_a01"][1]
VOLUME = 68000.0
RECORD_FIELDS = ["slice_mid", "slice_spread", "slice_req", "fill_qty", "fill_notional", "grid_t", "grid_mid", "grid_spread"]


def same(a, b) -> bool:
    return a.arrival_mid == b.arrival_mid and a.background_volume == b.background_volume and all(
        np.array_equal(getattr(a, f), getattr(b, f)) for f in RECORD_FIELDS)


def base(seed):
    return AgentMix(sampled.params("am", VALS), seed)


# ---- 1. lambda = 0 is AgentMix, bit for bit
@pytest.mark.parametrize("mode", LW.MODES)
def test_lambda_zero_is_bit_identical_to_agent_mix(mode):
    pol = candidate_set(0.1)
    ref = run_twin_set(base, "t", 3, TASK, pol)
    got = run_twin_set(LW.make_lambda(VALS, 0.0, VOLUME, mode), "t", 3, TASK, pol)
    assert set(ref) == set(got) and "none" in got and len(got) == 13  # twin and the 12 strategies of candidate_set
    for name in ref:
        assert same(ref[name], got[name]), name


def test_lambda_zero_event_log_and_state_are_identical():
    out = []
    for m in (base(5), LW.make_lambda(VALS)(5)):
        m.log = []
        book, tape = Book(), []
        m.init_book(book)
        m.advance(book, 0.0, 300.0, tape)
        out.append((m.log, tape, m.F, m.flow, m.ret))
    assert out[0] == out[1]


def test_lambda_changes_the_market_but_not_the_candidate_streams():
    """With lambda on, paths differ; the candidate event times and the exogenous draws are the same streams."""
    a, b = base(7), LW.make_lambda(VALS, 1.0, VOLUME)(7)
    books = []
    for m in (a, b):
        book, tape = Book(), []
        m.init_book(book)
        m.advance(book, 0.0, 600.0, tape)
        books.append(tape)
    assert books[0] != books[1]
    assert np.array_equal(a.next_t, b.next_t)  # thinning candidates: same streams, same number of draws
    assert a.exo.random() == b.exo.random()  # exogenous stream at the same position
    assert all(x.random() == y.random() for x, y in zip(a.rngs, b.rngs))


# ---- 2. every market-order path moves F exactly once per fill
def _count_markets(monkeypatch):
    calls = {"filled": 0, "limit_fills": 0}
    market, limit = Book.market, Book.limit

    def counted_market(self, is_buy, qty):
        fills = market(self, is_buy, qty)
        calls["filled"] += bool(sum(f[1] for f in fills))
        return fills

    def counted_limit(self, is_buy, price, qty, owner=None):
        fills, oid = limit(self, is_buy, price, qty, owner)
        calls["limit_fills"] += len(fills)
        return fills, oid

    monkeypatch.setattr(Book, "market", counted_market)
    monkeypatch.setattr(Book, "limit", counted_limit)
    return calls


def test_all_flow_moves_F_once_per_fill_manual_loop(monkeypatch):
    """fund_vol = 0, so F moves only through lambda: F - p0 must equal lam x (all signed executed shares)."""
    calls = _count_markets(monkeypatch)
    m = LW.make_lambda(VALS, 0.5, VOLUME, fund_vol=0.0)(11)
    book, tape, t, ext = Book(), [], 0.0, 0.0
    m.init_book(book)
    n_ext = 0
    while t < 900.0:
        if t >= 300.0 and int(t) % 30 == 0:  # an injected order every 30 s, alternating side, as the engine does it
            is_buy = (int(t) // 30) % 2 == 0
            got = sum(f[1] for f in book.market(is_buy, 700))
            m.on_external(is_buy, got, t, book)
            ext += got if is_buy else -got
            n_ext += bool(got)
        m.advance(book, t, t + GRID_DT, tape)
        t += GRID_DT
    signed = sum(x[1] for x in tape) + ext
    assert len(tape) > 50 and n_ext > 10
    assert m.n_moves == len(tape) + n_ext == calls["filled"]
    assert calls["limit_fills"] == 0  # no limit order of the world executes, so no executed order bypasses the hook
    assert m.moved_shares == signed
    assert m.F - m.p0 == pytest.approx(m.lam * signed, abs=1e-9)


@pytest.mark.parametrize("mode", LW.MODES)
def test_F_moves_once_per_fill_through_the_engine(monkeypatch, mode):
    calls = _count_markets(monkeypatch)
    reg = LW.Registry()
    pol = {k: candidate_set(0.1)[k] for k in ("P0_twap", "P1_immediate", "P6_signal_reversal")}
    recs = run_twin_set(LW.make_lambda(VALS, 0.5, VOLUME, mode, fund_vol=0.0, registry=reg), "t", 4, TASK, pol)
    snap, runs = reg.items[0], reg.items[1:]
    assert len(runs) == len(pol) + 1
    names = ["none"] + list(pol)
    n_children = {n: int((recs[n].fill_qty > 0).sum()) if n != "none" else 0 for n in names}
    sold = {n: float(recs[n].fill_qty.sum()) if n != "none" else 0.0 for n in names}
    if mode == "all":
        total = snap.n_moves + sum(r.n_moves - snap.n_moves for r in runs)
        assert total == calls["filled"]
        assert calls["limit_fills"] == 0
        for r in runs:
            assert r.F - r.p0 == pytest.approx(r.lam * r.moved_shares, abs=1e-9)
        assert runs[0].n_moves > snap.n_moves > 0
    else:  # trader-only: background trades leave F unchanged; each child moves it once
        assert snap.n_moves == 0 and snap.F == snap.p0
        for n, r in zip(names, runs):
            assert r.n_moves == n_children[n]
            assert r.moved_shares == -sold[n]
            assert r.F - r.p0 == pytest.approx(-r.lam * sold[n], abs=1e-9)
        assert runs[0].F == runs[0].p0 and calls["filled"] > 100


# ---- 3. trader-only arm: passive data are those of lambda = 0
def test_trader_only_passive_run_is_bit_identical_to_lambda_zero():
    out = []
    for f in (LW.make_lambda(VALS), LW.make_lambda(VALS, 2.0, VOLUME, "trader")):
        r = LW.passive_mid(f, 9, 900.0)
        out.append((r["mid"].tolist(), r["tape"], r["model"].F))
    assert out[0] == out[1]


def test_trader_only_twin_is_the_lambda_zero_twin_and_twap_differs():
    pol = {"P0_twap": candidate_set(0.1)["P0_twap"]}
    ref = run_twin_set(LW.make_lambda(VALS), "t", 6, TASK, pol)
    got = run_twin_set(LW.make_lambda(VALS, 2.0, VOLUME, "trader"), "t", 6, TASK, pol)
    assert same(ref["none"], got["none"])
    assert not same(ref["P0_twap"], got["P0_twap"])


# ---- 4. determinism
@pytest.mark.parametrize("mode", LW.MODES)
def test_determinism(mode):
    pol = {k: candidate_set(0.1)[k] for k in ("P0_twap", "P6_signal_reversal")}
    a = run_twin_set(LW.make_lambda(VALS, 0.5, VOLUME, mode, fund_vol=0.1), "t", 8, TASK, pol)
    b = run_twin_set(LW.make_lambda(VALS, 0.5, VOLUME, mode, fund_vol=0.1), "t", 8, TASK, pol)
    for name in a:
        assert same(a[name], b[name]), name
    c = run_twin_set(LW.make_lambda(VALS, 0.5, VOLUME, mode, fund_vol=0.1), "t", 9, TASK, pol)
    assert not same(a["none"], c["none"])


# ---- 5. units and volatility matching
def test_units():
    lam = LW.ticks_per_share(1.0, 50_000.0)  # 1 bp per 1% of 50,000 shares = 0.1 tick per 500 shares
    assert lam == pytest.approx(0.1 / 500.0)
    assert LW.ticks_per_share(0.0, 50_000.0) == 0.0


def test_pilot_draws_are_fixed_and_do_not_depend_on_the_block_used():
    a, b = LW.pilot_draws(), LW.pilot_draws()
    assert a == b and len(a) == 16
    assert list(a)[0] == "XLP_pilot_a01" and all(k.startswith(LW.PILOT_PREFIX) for k in a)
    assert LW.PILOT_SEED == 20261130


def test_match_fund_vol_on_a_known_function():
    res = LW.match_fund_vol(lambda fv: 2.0 + 100.0 * fv**2, 0.3, 6.0)  # exact answer 0.2
    assert res["feasible"] and res["matched"] and abs(res["ratio"] - 1.0) <= LW.VOL_TOL
    assert res["fund_vol"] == pytest.approx(0.2, abs=0.01)
    bad = LW.match_fund_vol(lambda fv: 8.0 + 100.0 * fv**2, 0.3, 6.0)  # above target even at fund_vol = 0
    assert not bad["feasible"] and bad["fund_vol"] == 0.0
    same_already = LW.match_fund_vol(lambda fv: 6.1, 0.3, 6.0)
    assert same_already["matched"] and same_already["fund_vol"] == pytest.approx(0.3) and len(same_already["evals"]) == 1


def test_volatility_matching_search_is_reproducible():
    def search():
        def var_of(fv):
            mids = [LW.passive_mid(LW.make_lambda(VALS, 1.0, VOLUME, fund_vol=fv), 20_000 + s, 1200.0)["mid"] for s in range(4)]
            return LW.return_var(mids)

        target = LW.return_var([LW.passive_mid(LW.make_lambda(VALS), 20_000 + s, 1200.0)["mid"] for s in range(4)])
        return LW.match_fund_vol(var_of, VALS["fund_vol"], target, max_iter=3)

    a, b = search(), search()
    assert a == b
    assert a["fund_vol"] < VALS["fund_vol"]  # all-flow lambda adds variance, so the matched fund_vol is lower
