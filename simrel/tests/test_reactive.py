"""Tests of the reactive (predator) reference family and of the randomised TWAP (design addendum 11, development)."""

import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simrel.core import Task
from simrel.eventsim import FlowModel, run_twin_set
from simrel.lob import Book
from simrel.policies.library import ExpSchedule, candidate_set
from simrel.policies.reactive import RandomTWAP, reactive_candidate_set
from simrel.reference_worlds import reactive as RX

TASK = Task("sell", 6000, 600, 20, 120)
MARKETS = RX.pilot_markets()
CASES = [MARKETS["XRP_pilot_a01"], MARKETS["XRP_pilot_l01"]]
ON = replace(RX.PredatorParams(), aggression=2.0, inventory_cap=5000)  # the family's detector: noisy time stamps, size-led
TIGHT_ON = replace(ON, **RX.DETECTORS["tight"])  # the comparison detector: exact time stamps, tight gap test
TIGHT = replace(RX.PredatorParams(), **RX.DETECTORS["tight"])
RECORD_FIELDS = ["slice_mid", "slice_spread", "slice_req", "fill_qty", "fill_notional", "grid_t", "grid_mid", "grid_spread"]


def same(a, b) -> bool:
    return a.arrival_mid == b.arrival_mid and a.background_volume == b.background_volume and all(
        np.array_equal(getattr(a, f), getattr(b, f)) for f in RECORD_FIELDS)


def run(case, pp, seed=3, policies=None, recorder=None):
    fam, vals = case
    policies = policies if policies is not None else reactive_candidate_set(0.1, "t", seed)
    return run_twin_set(RX.make_reactive(fam, vals, pp, recorder), "t", seed, TASK, policies), policies


# ---- 1. strength 0 is the base world, bit for bit
@pytest.mark.parametrize("case", CASES)
def test_strength_zero_is_bit_identical_to_the_base_world(case):
    fam, vals = case
    pol = reactive_candidate_set(0.1, "t", 3)
    base = run_twin_set(lambda s: RX.base_world(fam, vals, s), "t", 3, TASK, pol)
    for pp in (RX.PredatorParams(), replace(ON, aggression=0.0), replace(ON, inventory_cap=0), replace(TIGHT_ON, aggression=0.0)):
        off, _ = run(case, pp, policies=pol)
        assert set(off) == set(base) and "none" in off
        for name in base:
            assert same(base[name], off[name]), name


def test_strength_zero_event_log_is_identical():
    fam, vals = CASES[0]
    logs = []
    for wrap in (False, True):
        m = RX.make_reactive(fam, vals, RX.PredatorParams())(5) if wrap else RX.base_world(fam, vals, 5)
        m.log = []
        book, tape = Book(), []
        m.init_book(book)
        m.advance(book, 0.0, 300.0, tape)
        logs.append((m.log, tape))
    assert logs[0] == logs[1]


# ---- 2. identity-blindness
class ScriptedBase(FlowModel):
    """A base world that prints a given list of trades on the tape and does nothing else."""

    lot, p0, log = 100, 1000, None

    def __init__(self, trades):
        self.trades = sorted(trades)

    def init_book(self, book):
        for k in range(1, 30):
            book.limit(True, 1000 - k, 10**6)
            book.limit(False, 1000 + k, 10**6)

    def tick(self, book, t):
        pass

    def record(self, *a):
        pass

    def advance(self, book, t, t_until, tape):
        tape.extend(x for x in self.trades if t <= x[0] < t_until)


def test_detector_input_has_no_identity_field():
    assert list(inspect.signature(RX.Predator.observe).parameters) == ["self", "t", "side", "size"]
    assert list(inspect.signature(RX.TradeLog.add).parameters) == ["self", "t", "side", "size"]
    assert list(inspect.signature(RX.detect).parameters) == ["rows", "p"]
    log = RX.TradeLog(100.0)
    log.add(1.0, -1, 300)
    assert log.rows[0] == (1.0, -1, 300.0) and len(RX.TradeLog.FIELDS) == 3


@pytest.mark.parametrize("pp", [ON, TIGHT_ON])
def test_external_and_background_trades_are_indistinguishable_to_the_predator(pp):
    """The same trades, once printed by the base world and once injected through on_external, leave the predator in the same
    state (including the timestamp noise it drew for them)."""
    noise = [(7.3, 200), (41.9, -100), (66.6, 500), (93.2, -400), (128.4, 100), (171.0, -300)]
    children = [(30.0 * k, -300) for k in range(1, 17)]
    states = []
    for external in (False, True):
        world = RX.ReactiveWorld(ScriptedBase(noise if external else noise + children), RX.Predator(pp, 0))
        book, tape = Book(), []
        world.init_book(book)
        t = 0.0
        while t < 600.0:
            if external and any(abs(c[0] - t) < 1e-9 for c in children):
                book.market(False, 300)
                world.on_external(False, 300, t, book)
            world.advance(book, t, t + 5.0, tape)
            t += 5.0
        p = world.pred
        states.append((list(p.log.rows), p.triggers, p.fills, p.inv, p.cash, p.mode))
    assert states[0] == states[1]
    assert states[0][1] and states[0][2], "the scripted run should trigger and trade"


# ---- 3. determinism
@pytest.mark.parametrize("case", CASES)
def test_same_seed_gives_identical_output(case):
    pol = {"P0_twap": ExpSchedule(0.0), "P0r_twap_random": reactive_candidate_set(0.1, "t", 3)["P0r_twap_random"]}
    ra, rb = RX.Recorder(), RX.Recorder()
    a, _ = run(case, ON, policies=pol, recorder=ra)
    b, _ = run(case, ON, policies=pol, recorder=rb)
    for name in a:
        assert same(a[name], b[name])
    for x, y in zip(ra.branches, rb.branches):
        assert x.fills == y.fills and x.triggers == y.triggers


# ---- 4. flat at the end, consistent accounting, and the predator's orders are public
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pp", [ON, TIGHT_ON])
def test_predator_is_flat_and_its_accounts_add_up(case, pp):
    rec = RX.Recorder()
    pol = {"P0_twap": ExpSchedule(0.0)}
    out, _ = run(case, pp, policies=pol, recorder=rec)
    pred = rec.by_run(pol)["P0_twap"]
    assert pred.fills, "the predator should trade against a TWAP"
    assert pred.inv == 0  # flat at the end of the post window (a late false trigger can break this; its rate is in the pilot report)
    assert pred.inv == sum(f[1] for f in pred.fills)
    assert pred.cash == pytest.approx(-sum(np.sign(f[1]) * f[2] for f in pred.fills))
    assert max(abs(x) for x in np.cumsum([f[1] for f in pred.fills])) <= pp.inventory_cap
    in_task = [f for f in pred.fills if f[0] >= 1200.0]
    assert in_task and in_task[0][1] < 0, "it trades on the metaorder's side first"
    assert not same(out["P0_twap"], run(case, replace(pp, aggression=0.0), policies=pol)[0]["P0_twap"])


def test_flatten_deadline_bounds_the_unwind():
    """Once the pattern stops, the position is closed within patience x period + flatten_after_s (+ one step)."""
    children = [(30.0 * k, -300) for k in range(1, 17)]
    world = RX.ReactiveWorld(ScriptedBase(children), RX.Predator(TIGHT_ON, 0))
    book, tape = Book(), []
    world.init_book(book)
    for k in range(130):
        world.advance(book, 5.0 * k, 5.0 * k + 5.0, tape)
    p = world.pred
    assert p.fills and p.inv == 0 and p.mode == RX.IDLE
    assert max(f[0] for f in p.fills) <= 480.0 + p.p.patience * 30.0 + p.p.flatten_after_s + p.p.unwind_step_s


def test_predator_orders_reach_the_tape_the_log_and_the_base_world_signal():
    children = [(30.0 * k, -300) for k in range(1, 17)]
    world = RX.ReactiveWorld(ScriptedBase(children), RX.Predator(ON, 0))
    book, tape = Book(), []
    world.init_book(book)
    for k in range(110):
        world.advance(book, 5.0 * k, 5.0 * k + 5.0, tape)
    mine = [(f[0], f[1]) for f in world.pred.fills]
    assert mine and all(x in tape for x in mine)
    fam, vals = CASES[0]
    w = RX.make_reactive(fam, vals, ON)(1)
    w.log = []
    book, tape = Book(), []
    w.init_book(book)
    w.advance(book, 0.0, 50.0, tape)
    flow, n_log, n_tape = w.base.flow, len(w.log), len(tape)
    got, _ = w._send(book, tape, 50.0)(False, 400)
    assert got == 400 and len(tape) == n_tape + 1 and len(w.log) == n_log + 1
    assert w.log[-1][1:4] == (3, 400, 0)  # a market sell at level 0, like any background market order
    assert w.base.flow == pytest.approx(flow - 400)


# ---- 5. detection
def rows_of(times, sizes, side=-1):
    return [(t, side, z) for t, z in zip(times, sizes)]


@pytest.mark.parametrize("p", [RX.PredatorParams(), TIGHT])
def test_regular_sequence_triggers_and_jittered_does_not(p):
    n = p.min_matches + 1
    regular = rows_of([30.0 * k for k in range(n)], [300] * n)
    hit = RX.detect(regular, p)
    assert hit is not None and hit.side == -1 and hit.period == pytest.approx(30.0) and hit.size == pytest.approx(300)
    rng = np.random.default_rng(0)
    assert RX.detect(rows_of([30.0 * k + rng.uniform(-6, 6) for k in range(n)], [300] * n), p) is None  # time jitter
    assert RX.detect(rows_of([30.0 * k for k in range(n)], ([100, 900, 200, 1500, 100, 700] * 2)[:n]), p) is None  # size jitter
    assert RX.detect(rows_of([30.0 * k for k in range(p.min_matches - 1)], [300] * (p.min_matches - 1)), p) is None  # too few
    mixed = [(30.0 * k, -1 if k % 2 else 1, 300) for k in range(n)]
    assert RX.detect(mixed, p) is None  # not one-sided at this period


def test_family_detector_tolerates_timestamp_noise_and_the_tight_one_does_not():
    rng = np.random.default_rng(3)
    noisy = sorted(rows_of([30.0 * k + 0.2 * rng.standard_normal() for k in range(9)], [300] * 9))
    assert RX.detect(noisy, RX.PredatorParams()) is not None
    assert RX.detect(noisy, TIGHT) is None
    # lot rounding of a small TWAP (100 / 200 shares): too irregular in size for the family's detector, not for the tight one
    small = rows_of([30.0 * k for k in range(9)], [100, 100, 200] * 3)
    assert RX.detect(small, RX.PredatorParams()) is None and RX.detect(small, TIGHT) is not None


def test_timestamp_noise_is_applied_to_every_trade_from_the_predators_own_stream():
    a, b = RX.Predator(RX.PredatorParams(), 5), RX.Predator(RX.PredatorParams(), 5)
    for k in range(20):
        a.observe(10.0 * k, 1, 100)
        b.observe(10.0 * k, 1, 100)
    assert a.log.rows == b.log.rows
    err = np.array([r[0] for r in a.log.rows]) - 10.0 * np.arange(20)
    assert np.all(err != 0) and np.all(np.abs(err) <= 3 * 0.2 + 1e-12) and 0.05 < err.std() < 0.5


def test_regular_sequence_is_found_among_other_trades():
    p = TIGHT
    rng = np.random.default_rng(1)
    noise = [(float(t), int(rng.choice([-1, 1])), 100.0 * rng.geometric(0.35)) for t in rng.uniform(0, 150, 60)]
    rows = sorted(noise + rows_of([30.0 * k for k in range(6)], [300] * 6))
    rows = [r for r in rows if r[0] <= 150.0]
    assert rows[-1][0] == 150.0
    assert RX.detect(rows, p) is not None
    assert RX.detect(sorted(noise), p) is None


# ---- 6. the randomised TWAP
def test_p0r_completes_and_is_reproducible():
    for case in CASES:
        for seed in range(3):
            pol = {"P0r_twap_random": reactive_candidate_set(0.1, "w", seed)["P0r_twap_random"]}
            a, _ = run(case, RX.PredatorParams(), seed=seed, policies=pol)
            b, _ = run(case, RX.PredatorParams(), seed=seed, policies=pol)
            assert a["P0r_twap_random"].filled == TASK.total_qty
            assert np.array_equal(a["P0r_twap_random"].fill_qty, b["P0r_twap_random"].fill_qty)
            keep, _ = pol["P0r_twap_random"].plan(TASK.n_slices)
            assert np.all(a["P0r_twap_random"].slice_req[:-1][~keep[:-1]] == 0), "skipped slices send nothing"


def test_p0r_plan_depends_on_world_and_seed_only():
    a = reactive_candidate_set(0.05, "w", 1)["P0r_twap_random"]
    assert a == reactive_candidate_set(0.15, "w", 1)["P0r_twap_random"]
    ka, ca = a.plan(20)
    kb, cb = RandomTWAP(a.key).plan(20)
    assert np.array_equal(ka, kb) and np.array_equal(ca, cb) and ca[-1] == pytest.approx(1.0)
    assert not np.array_equal(ka, reactive_candidate_set(0.05, "w", 2)["P0r_twap_random"].plan(20)[0]) or \
        not np.array_equal(ca, reactive_candidate_set(0.05, "w", 2)["P0r_twap_random"].plan(20)[1])
    plans = [RandomTWAP((7, s)).plan(20)[0] for s in range(200)]
    assert 0.2 < 1 - np.mean([k[:-1].mean() for k in plans]) < 0.4  # about skip_prob of the slices are skipped
    # the same plan in two different markets: it does not use a market stream
    pol = {"P0r_twap_random": a}
    r1, _ = run(CASES[0], RX.PredatorParams(), seed=1, policies=pol)
    r2, _ = run(CASES[1], RX.PredatorParams(), seed=1, policies=pol)
    assert np.array_equal(r1["P0r_twap_random"].slice_req[:-1] == 0, r2["P0r_twap_random"].slice_req[:-1] == 0)


def test_reactive_candidate_set_extends_the_candidate_set():
    base, ext = candidate_set(0.05), reactive_candidate_set(0.05, "w", 0)
    assert list(ext)[:-1] == list(base) and list(ext)[-1] == "P0r_twap_random" and len(ext) == 13
    assert all(ext[k] == base[k] for k in base) and "P0r_twap_random" not in candidate_set(0.05)


# ---- round 2: the rational predator (passive exit, profit gate), attribution, Recorder guard, AUC check
PASSIVE = replace(ON, exit="passive", gate_mult=0.0)  # passive exit, no gate
RATIONAL = replace(ON, exit="passive", gate_mult=1.0)


def run2(case, pp, seed=3, policies=None, world=None):
    fam, vals = case
    policies = policies if policies is not None else {k: v for k, v in reactive_candidate_set(0.1, "t", seed).items() if k in ("P0_twap", "P2_front_b2", "P1_immediate")}
    rec, led = RX.Recorder(), RX.Recorder()
    make = RX.make_reactive(fam, vals, pp, rec, led)
    if world is not None:
        inner = make
        make = lambda s: world(inner(s))  # noqa: E731
    return run_twin_set(make, "t", seed, TASK, policies), rec.by_run(policies), led.by_run(policies)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pp", [PASSIVE, RATIONAL])
def test_passive_exit_accounts_add_up(case, pp):
    recs, preds, ledgers = run2(case, pp)
    traded = 0
    for name, pr in preds.items():
        mark = float(recs[name].grid_mid[-1])
        s = pr.summary(1200.0, mark)
        assert pr.inv == sum(f[1] for f in pr.fills) == sum(x[1] for x in pr.legs)
        assert pr.cash == pytest.approx(-sum(np.sign(f[1]) * f[2] for f in pr.fills))
        assert s["pred_pnl_ticks"] == pytest.approx(pr.cash + pr.inv * mark)  # a residual is reported and marked at the final mid
        assert s["pred_inv_end"] == pr.inv
        parts = sum(s[f"pred_{k}_edge_ticks"] for k in ("entry", "unwind", "passive", "flatten")) + s["pred_hold_pnl_ticks"]
        assert parts == pytest.approx(s["pred_pnl_ticks"])
        assert s["pred_unwind_volume"] == 0  # the passive predator never uses the aggressive unwinding schedule
        # every share bought back passively is attributed to exactly one counterparty
        assert s["pred_passive_volume"] == ledgers[name]["trader"] + ledgers[name]["background"]
        assert sum(o[2] for o in pr.orders.values()) <= abs(pr.inv)  # resting orders never exceed the position
        if pr.mode == RX.IDLE:
            assert pr.inv == 0 and not pr.orders
        traded += s["pred_passive_volume"]
    assert traded > 0 or pp.gate_mult > 0  # without the gate the exit path was exercised
    assert ledgers["none"]["trader"] == 0  # no trader in the twin


class FakeOps:
    def __init__(self):
        self.placed, self.sent = [], []

    def quote(self):
        return 999, 1001

    def place(self, is_buy, price, qty):
        self.placed.append((is_buy, price, qty))
        return len(self.placed)

    def cancel(self, oid):
        pass

    def send(self, is_buy, qty):
        self.sent.append((is_buy, qty))
        return qty, 999.0 * qty


def scripted_predator(drop_per_child: float) -> tuple:
    """Eight regular sells of 300 shares, 30 s apart, with the mid falling by drop_per_child ticks after each."""
    pp = replace(RX.PredatorParams(), aggression=1.0, inventory_cap=1000, exit="passive", gate_mult=1.0, ts_noise_s=0.0)
    pr = RX.Predator(pp, 0)
    for k in range(8):
        t = 100.0 + 30.0 * k
        pr.observe_quote(t - 1.0, 999 - k * drop_per_child, 1001 - k * drop_per_child)
        pr.observe(t, -1, 300)
    assert pr.mode == RX.TRACK and pr.front_t is not None
    return pr, FakeOps()


def test_gate_blocks_entry_when_expected_gain_is_below_cost_and_allows_it_when_above():
    pr, ops = scripted_predator(0.0)  # no price move per child: gain 0, cost one spread
    gain, cost = pr.gate_estimate(pr.front_t)
    assert gain == 0.0 and cost == pytest.approx(2.0)
    pr.act(pr.front_t, ops.send, ops)
    assert ops.sent == [] and ops.placed == [] and pr.inv == 0 and (pr.gate_checks, pr.gate_blocks) == (1, 1)

    pr, ops = scripted_predator(0.5)  # half a tick per child, eight children assumed left: gain 4 ticks against cost 2
    gain, cost = pr.gate_estimate(pr.front_t)
    assert gain == pytest.approx(4.0) and cost == pytest.approx(2.0)
    pr.act(pr.front_t, ops.send, ops)
    assert ops.sent == [(False, 300)] and pr.inv == -300 and (pr.gate_checks, pr.gate_blocks) == (1, 0)
    assert ops.placed == [(True, 999, 300)]  # the whole position rests at the best bid straight after entry


class PoisonLedger(dict):
    """Returns nonsense on every read. If any decision read the attribution, the market path would change."""

    def __getitem__(self, key):
        return 10 ** 9 + 7 * len(key)

    def get(self, key, default=None):
        return self[key]


def poison(world):
    world.ledger = PoisonLedger(world.ledger)
    return world


def test_attribution_never_reaches_the_predators_decisions():
    # 1. the Predator has no name for it and holds no reference to the world or its ledger
    src = inspect.getsource(RX.Predator)
    assert "ledger" not in src and "trader" not in src and "ReactiveWorld" not in src
    recs, preds, ledgers = run2(CASES[0], PASSIVE)
    for pr in preds.values():
        for v in vars(pr).values():
            assert not isinstance(v, (RX.ReactiveWorld, dict)) or v is pr.orders
            assert all(v is not led for led in ledgers.values())
    assert sum(led["trader"] + led["background"] for led in ledgers.values()) > 0
    # 2. scrambling the ledger changes nothing in the market or in the predator's account
    recs2, preds2, _ = run2(CASES[0], PASSIVE, world=poison)
    for name in recs:
        assert same(recs[name], recs2[name])
        assert preds[name].fills == preds2[name].fills and preds[name].triggers == preds2[name].triggers


@pytest.mark.parametrize("case", CASES)
def test_strength_zero_is_bit_identical_with_the_rational_settings(case):
    fam, vals = case
    pol = {k: v for k, v in reactive_candidate_set(0.1, "t", 3).items() if k in ("P0_twap", "P5_liquidity")}
    base = run_twin_set(lambda s: RX.base_world(fam, vals, s), "t", 3, TASK, pol)
    for pp in (replace(RX.PredatorParams(), exit="passive", gate_mult=1.0), replace(RATIONAL, aggression=0.0), replace(RATIONAL, inventory_cap=0)):
        recs, preds, ledgers = run2(case, pp, policies=pol)
        for name in base:
            assert same(base[name], recs[name])
            assert not preds[name].fills and not preds[name].orders and not any(ledgers[name].values())


def test_aggressive_predator_is_unchanged_by_the_round_2_fields():
    recs, _ = run(CASES[0], ON, policies={"P0_twap": candidate_set(0.1)["P0_twap"]})
    recs2, _ = run(CASES[0], replace(ON, gate_mult=5.0, exit_offset_ticks=1, reprice_ticks=0, passive_exit_s=1.0), policies={"P0_twap": candidate_set(0.1)["P0_twap"]})
    assert all(same(recs[k], recs2[k]) for k in recs)


def test_recorder_matches_run_twin_sets_copies():
    """Fails if run_twin_set stops making exactly one deep copy for the snapshot and one per run, in order."""
    fam, vals = CASES[1]
    full = reactive_candidate_set(0.1, "t", 3)
    pol = {"P1_immediate": full["P1_immediate"], "P0_twap": full["P0_twap"]}
    rec, led = RX.Recorder(), RX.Recorder()
    inner = RX.make_reactive(fam, vals, replace(RX.PredatorParams(), ts_noise_s=0.0), rec, led)

    def make(s):
        m = inner(s)
        m.pred.history = []
        return m

    recs = run_twin_set(make, "t", 3, TASK, pol)
    assert len(rec.branches) == len(led.branches) == len(pol) + 2
    assert len({id(b) for b in rec.branches}) == len(rec.branches)
    by = rec.by_run(pol)
    start = 1200.0
    own = {name: [(t, z) for t, s, z in pr.history if s == -1 and abs((t - start) / 30.0 - round((t - start) / 30.0)) < 1e-9 and 0 <= t - start < 600] for name, pr in by.items()}
    fills = {name: [(start + 30.0 * k, float(q)) for k, q in enumerate(recs[name].fill_qty) if q > 0] for name in pol}
    for name in pol:  # each run's predator saw exactly that run's children
        assert all(x in own[name] for x in fills[name]) and len(fills[name]) >= 1
    assert (start, 6000.0) in own["P1_immediate"] and (start, 6000.0) not in own["P0_twap"] and (start, 6000.0) not in own["none"]
    assert len(fills["P0_twap"]) == 20 and not all(x in own["none"] for x in fills["P0_twap"])
    assert rec.branches[0].history is not by["none"].history  # the snapshot copy is not one of the runs


def test_custom_auc_equals_mann_whitney_on_the_saved_e3_data():
    import pandas as pd
    from scipy.stats import mannwhitneyu

    from simrel.experiments.reactive_pilot_report import FEATURES, _fit_logistic, auc

    path = Path(__file__).resolve().parents[2] / "results" / "simrel" / "goal2" / "addendum11_pilot" / "e3_samples.csv"
    d = pd.read_csv(path)
    checked = 0
    for arm in ("strong", "strong_loose"):
        on = d[(d["arm"] == arm) & (d["tag"] == "A")]
        off = d[(d["arm"] == "off") & (d["tag"] == "B")]
        both = pd.concat([on, off])
        y = np.r_[np.ones(len(on)), np.zeros(len(off))]
        feats = [f for f in FEATURES if both[f].notna().all()]
        X = both[feats].to_numpy()
        X = (X - X.mean(0)) / X.std(0)
        w = _fit_logistic(X, y)
        for score in (w[0] + X @ w[1:], X[:, 0]):  # a classifier score and a single feature (with ties possible)
            u = mannwhitneyu(score[y == 1], score[y == 0]).statistic
            assert auc(score, y) == pytest.approx(u / (len(on) * len(off)), abs=1e-12)
            checked += 1
    assert checked == 4
