"""
LOB-Impact protocol on the Goal 1 simulators (design: simrel/reports/lob_impact_design.md).

LOB-Impact (Nigmatulin et al., ICAIF 2026) injects a metaorder of child market orders into a simulator on a message
clock and measures the response. Calibration per stock and day, as in the paper: the child is the day's median
market-order size, and the spacing is dN = 9 / phi generated messages (participation 10%), phi being the share of
market orders in the day's visible message stream. Arms, with the same seed (same start book, same background
random streams) in every arm:

    noins   free run of 101 dN messages                         phase 1
    beta    100 children, one every dN messages                 phase 2 (visible)
    blind   as beta, but the simulator's reaction is switched    phase 2 attribution (Hawkes only: the children fill
            off: no self-excitation from the children           in the book but do not excite the intensities)
    relax   10 children, then 100 cooling windows of dN          phase 3

Each rollout starts from the real book at the start of one ten-minute window of a usable slice (all eight days) and
is labelled with that window's day; ABIDES, which starts from its own book, takes the day label of the same start.
Rollouts are reduced to the numbers the estimators need (simrel.metrics.lob_impact.reduce); a few are also written
as LOBSTER CSVs in the reference tool's layout for a cross-check.

    SIMREL_SYMBOL=INTC python simrel/experiments/impact_01_run.py calib
    SIMREL_SYMBOL=INTC python simrel/experiments/impact_01_run.py run --sims hawkes,abides [--procs 4] [--shard i/n]
    SIMREL_SYMBOL=INTC python simrel/experiments/impact_01_run.py score
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel import slices as S  # noqa: E402
from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402
from simrel.metrics import lob_impact as LI  # noqa: E402

OUT = ROOT / "results" / "simrel" / "lob_impact" / C.SYMBOL
SIMS = ["hawkes", "hawkes_sd_q_c", "hawkes_sd_c_f_m", "abides", "abides_quoting"]
ARMS = {"noins": (0, 0), "beta": (100, 0), "blind": (100, 0), "relax": (10, 100)}  # arm -> (children, cooling windows)
BASE_ARMS = list(ARMS)
# diagnostic build-up arms (design addendum 2): _t spaces children by the real stock's time between them (ABIDES only;
# the event simulators already run near the real message rate), _x5 uses five times the median market order
ARMS.update({"beta_t": (100, 0), "beta_x5": (100, 0), "beta_t_x5": (100, 0)})
N_ROLLOUTS = 64  # per simulator, arm and side
N_EXPORT = 6  # rollouts per simulator, arm and side also written as LOBSTER CSVs (INTC, two simulators)
EXPORT_SIMS = ("hawkes", "abides")
SEED0 = 70_000
PARTICIPATION = 0.10
MAX_SIM_S = 8 * 3600  # a simulator that needs longer than this for one rollout has stalled
MAX_ABIDES_SPAN_S = 5.5 * 3600  # ABIDES rollouts start at 10:00 and must end before its agents stop at 16:00


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pickle.dump(obj, open(tmp, "wb"))
    tmp.replace(path)


def load(path: Path):
    return pickle.load(open(path, "rb"))


# ------------------------------------------------------------------ calibration
def market_orders(m: np.ndarray) -> np.ndarray:
    """Sizes of aggressive orders: visible executions with the same time stamp and resting side form one order."""
    ex = m[m[:, 1] == 4]
    if not len(ex):
        return np.zeros(0)
    key = np.r_[True, (np.diff(ex[:, 0]) != 0) | (np.diff(ex[:, 5]) != 0)]
    return np.bincount(np.cumsum(key) - 1, weights=ex[:, 3])


def stage_calib() -> None:
    """Per day: child size, spacing dN, daily volume V and log range (the paper's daily statistics), message rate."""
    rows, starts = [], []
    wins = pd.read_csv(R.OUT / "windows.csv")
    wins = wins[wins["slice"].isin(R.usable_slices())]
    for d in C.FIT_DAYS + C.TEST_DAYS:
        m, b = L.load_day(C.DATA_DIR, C.SYMBOL, d)
        # start books: the book just before every window of a usable slice on this day
        for _, w in wins[wins["date"] == d].iterrows():
            a = int(np.searchsorted(m[:, 0], int(w["start_s"]) * L.NS))
            starts.append({"slice": w["slice"], "day": d, "start_s": int(w["start_s"]), "orders": S.book_orders(b[a - 1])})
        t = m[:, 0] / L.NS
        k = (t >= 9.5 * 3600) & (t < 16 * 3600)
        m, b = m[k], b[k]
        m, b = L.visible(m, b, b[0])
        mo = market_orders(m)
        phi = len(mo) / len(m)
        ex = m[m[:, 1] == 4]
        rows.append({"day": d, "messages": len(m), "market_orders": len(mo), "phi": phi, "dN": int(round((1 / PARTICIPATION - 1) / phi)),
                     "child": int(np.median(mo)), "V": float(ex[:, 3].sum()), "high": float(ex[:, 4].max()), "low": float(ex[:, 4].min()),
                     "open": float(ex[0, 4]), "close": float(ex[-1, 4]), "rate_per_s": len(m) / 23400.0})
        print(f"  {d}: phi {phi:.4f}  dN {rows[-1]['dN']}  child {rows[-1]['child']}  V {rows[-1]['V']:,.0f}", flush=True)
    c = pd.DataFrame(rows)
    c["range"] = np.log(c["high"] / c["low"])
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(OUT / "calibration.csv", index=False)
    # the reference tool's daily statistics file (lob_impact.vol format)
    pd.DataFrame({"ticker": C.SYMBOL, "day": c["day"], "open_price": c["open"], "highest_price": c["high"], "lowest_price": c["low"],
                  "close_price": c["close"], "execution_sum": c["V"]}).to_csv(OUT / "daily_stats.csv", index=False)
    # shuffled once so that any prefix of the start books spreads over days and slices
    order = np.random.default_rng(0).permutation(len(starts))
    save(OUT / "starts.pkl", [starts[i] for i in order])
    print(f"  {len(starts)} start books over {len({s['day'] for s in starts})} days and {len(R.usable_slices())} slices", flush=True)
    # phase 1 target: the real stream from each start, 101 dN visible messages
    stage_real()
    for sim in SIMS:
        if sim.startswith("abides"):
            for name in R.usable_slices():
                abides_rate(sim, name)


def calibration() -> pd.DataFrame:
    return pd.read_csv(OUT / "calibration.csv").set_index("day")


def daily() -> dict:
    c = calibration()
    return {d: {"V": float(r["V"]), "range": float(r["range"])} for d, r in c.iterrows()}


def stage_real() -> None:
    """Records of the real stream (the paper's Historic arm without injection) from the first N_ROLLOUTS starts."""
    path = OUT / "rec" / "real" / "noins.pkl"
    if path.exists():
        return
    cal, starts = calibration(), load(OUT / "starts.pkl")
    recs, cache = [], {}
    for st in starts[:N_ROLLOUTS]:
        d = st["day"]
        if d not in cache:
            m, b = L.load_day(C.DATA_DIR, C.SYMBOL, d)
            mv, bv = L.visible(m, b, b[0])
            cache = {d: (mv, bv)}
        m, b = cache[d]
        a = int(np.searchsorted(m[:, 0], st["start_s"] * L.NS))
        n = 101 * int(cal.loc[d, "dN"])
        r = to_rollout(m[a:a + n], b[a:a + n], [], d, +1)
        recs.append(LI.reduce(r))
    save(path, recs)


# ------------------------------------------------------------------ rollouts
def to_rollout(msg: np.ndarray, book: np.ndarray, aggr, day: str, sign: int) -> dict:
    ask, bid = book[:, 0].astype(float), book[:, 2].astype(float)
    ask[np.abs(ask) >= L.EMPTY] = np.nan
    bid[np.abs(bid) >= L.EMPTY] = np.nan
    m = msg.astype(float)
    m[:, 0] = msg[:, 0] / L.NS
    return {"msg": m, "ask": ask, "bid": bid, "aggr": np.asarray(aggr, int), "day": day, "sign": sign}


def child_rows(msg: np.ndarray, times_ns) -> list:
    """Row of each child: the last execution row stamped with the child's time (a child that filled nothing is dropped)."""
    out = []
    for ts in times_ns:
        rows = np.flatnonzero((msg[:, 0] == ts) & (msg[:, 1] == 4))
        if len(rows):
            out.append(int(rows[-1]))
    return out


def eventsim_rollout(model, dN: int, child: int, n_ins: int, n_cool: int, is_buy: bool, react: bool, warm_s: float, clock0_s: int = 10 * 3600):
    """Run a simrel.eventsim flow model on a message clock: a child every dN tape rows after the warm-up."""
    from simrel.lob import Book

    book = Book()
    book.tape = []
    model.init_book(book)
    model.advance(book, 0.0, warm_s, [])
    n0, t = len(book.tape), float(warm_s)
    rate = max(n0 / warm_s, 1.0)
    end = (n_ins + n_cool + 1) * dN if n_ins else 101 * dN
    j, times = 0, []
    while True:
        done = len(book.tape) - n0
        target = (j + 1) * dN if j < n_ins else end
        if done >= target:
            if j < n_ins:
                book.now = t
                fills = book.market(is_buy, child)
                got = sum(f[1] for f in fills)
                if react:
                    model.on_external(is_buy, got, t, book)
                times.append(int(round((clock0_s + t) * L.NS)))
                j += 1
                continue
            break
        if t - warm_s > MAX_SIM_S:
            raise RuntimeError(f"stalled: {done} of {end} messages after {MAX_SIM_S} s")
        step = max(0.5 * (target - done) / rate, 2.0 / rate)
        model.advance(book, t, t + step, [])
        t += step
        rate = max((len(book.tape) - n0) / (t - warm_s), 0.1)
    w = S.visible_window(L.tape_to_lobster(book.tape, warm_s, t - warm_s + 1e-6, clock0_s=clock0_s))
    return w.message, w.book, child_rows(w.message, times)


def abides_rollout(spec, seed: int, dN: int, child: int, n_ins: int, n_cool: int, is_buy: bool, rate: float):
    """ABIDES on a time clock matched to the message clock: one child every dN / rate seconds (rate = the
    configuration's own visible message rate), placed by the existing execution agent."""
    from simbench import abides_world as AW

    AW.WORLDS[spec.name] = spec
    n = n_ins if n_ins else 100
    # a slow configuration would run past the close: the rollout is then fitted into MAX_ABIDES_SPAN_S, which makes
    # its children denser than dN messages apart (design addendum 1)
    dt = min(dN / rate, MAX_ABIDES_SPAN_S / (n + n_cool + 1))
    task = AW.Task(side="buy" if is_buy else "sell", total_qty=child * n_ins, horizon_s=int(round(n * dt)), n_slices=n,
                   post_s=int(round((n_cool + 1) * dt)))
    ep = AW.run_episode(spec.name, seed, task, (lambda s: child) if n_ins else (lambda s: 0), f"li{n_ins}", warmup_s=C.ABIDES_WARMUP_S, keep_history=True)
    pre = dt
    w = S.visible_window(L.abides_to_lobster(ep.history, ep.start_ns, -pre, pre + task.horizon_s + task.post_s))
    exec_id = max(int(h.get("oppos_agent_id", -1) or -1) for h in ep.history if h["type"] == "EXEC") if n_ins else -1
    day0 = ep.start_ns - ep.start_ns % (86400 * L.NS)
    times = sorted({int(h["time"]) - day0 for h in ep.history if h["type"] == "EXEC" and h.get("oppos_agent_id") == exec_id}) if n_ins else []
    return w.message, w.book, child_rows(w.message, times)


def abides_rate(sim: str, name: str) -> float:
    """Visible messages per second of an ABIDES configuration, from its unperturbed realism windows (cached in
    abides_rates.json by the calib stage, so the rollouts do not need the windows)."""
    path = OUT / "abides_rates.json"
    rates = json.load(open(path)) if path.exists() else {}
    key = f"{sim}|{name}"
    if key not in rates:
        ws = R.generated(sim, name) or [R.unperturbed_window(sim, name, R.SEED_GEN)]
        rates[key] = float(np.mean([len(w.message) / w.length_s for w in ws]))
        json.dump(rates, open(path, "w"), indent=1)
    return rates[key]


def rollout_job(args) -> str:
    sim, arm, side, k = args
    path = OUT / "rec" / sim / f"{arm}_{side}_{k:03d}.pkl"
    if path.exists():
        return "cached"
    R._memory_guard(path, 6.0)
    st = load(OUT / "starts.pkl")[k]
    cal = calibration().loc[st["day"]]
    dN, child = int(cal["dN"]), int(cal["child"]) * (5 if arm.endswith("_x5") else 1)
    n_ins, n_cool = ARMS[arm]
    is_buy, seed = side == "buy", SEED0 + k
    t0 = time.time()
    try:
        if sim.startswith("abides"):
            spec = R.abides_family_spec(sim, st["slice"])
            rate = float(cal["rate_per_s"]) if arm.startswith("beta_t") else abides_rate(sim, st["slice"])
            msg, book, aggr = abides_rollout(spec, seed, dN, child, n_ins, n_cool, is_buy, rate)
        else:
            model = R.hawkes_model(st["slice"], sim, starts=[st["orders"]])(seed)
            msg, book, aggr = eventsim_rollout(model, dN, child, n_ins, n_cool, is_buy, arm != "blind", C.HAWKES_WARMUP_S)
        r = to_rollout(msg, book, aggr, st["day"], 1 if is_buy else -1)
        rec = LI.reduce(r, n_ins, n_cool)
        rec.update(sim=sim, arm=arm, side=side, k=k, slice=st["slice"], dN=dN, child=child, n_children=len(aggr), wall_s=time.time() - t0)
        if C.SYMBOL == "INTC" and sim in EXPORT_SIMS and k < N_EXPORT and arm in BASE_ARMS:  # diagnostic arms are never exported
            export_csv(sim, arm, side, k, msg, book, aggr, st["day"])
    except Exception as err:
        rec = {"sim": sim, "arm": arm, "side": side, "k": k, "failed": repr(err)[:300]}
    save(path, rec)
    return f"{sim} {arm} {side} {k}: " + (rec["failed"][:80] if "failed" in rec else f"{rec['n_rows']} rows, {rec['n_children']} children, {rec['wall_s']:.0f}s")


def export_job(args) -> str:
    """Regenerate one exported rollout from its seed, write its CSVs, and compare with the stored record."""
    sim, arm, side, k = args
    st = load(OUT / "starts.pkl")[k]
    cal = calibration().loc[st["day"]]
    dN, child = int(cal["dN"]), int(cal["child"])
    n_ins, n_cool = ARMS[arm]
    is_buy, seed = side == "buy", SEED0 + k
    if sim.startswith("abides"):
        msg, book, aggr = abides_rollout(R.abides_family_spec(sim, st["slice"]), seed, dN, child, n_ins, n_cool, is_buy, abides_rate(sim, st["slice"]))
    else:
        model = R.hawkes_model(st["slice"], sim, starts=[st["orders"]])(seed)
        msg, book, aggr = eventsim_rollout(model, dN, child, n_ins, n_cool, is_buy, arm != "blind", C.HAWKES_WARMUP_S)
    export_csv(sim, arm, side, k, msg, book, aggr, st["day"])
    new = LI.reduce(to_rollout(msg, book, aggr, st["day"], 1 if is_buy else -1), n_ins, n_cool)
    old = load(OUT / "rec" / sim / f"{arm}_{side}_{k:03d}.pkl")
    same = all(np.array_equal(np.asarray(new.get(f), float), np.asarray(old.get(f), float), equal_nan=True)
               for f in ("eps", "tmid", "mid_child", "mid_pre", "kmid") if f in new or f in old) and new["n_rows"] == old["n_rows"]
    return f"{sim} {arm} {side} {k}: {'reproduces the stored record' if same else 'DIFFERS from the stored record'}"


def export_csv(sim, arm, side, k, msg, book, aggr, day) -> None:
    """The reference tool's layout: noins and blind under controls/, beta and relaxation under grid/."""
    model = sim.replace("_", "-")
    shape = "relaxation" if arm == "relax" else "beta"
    base = {"noins": OUT / "csv" / "controls" / "noins", "blind": OUT / "csv" / "controls" / "invisible"}.get(arm, OUT / "csv" / "grid")
    exp = base / f"{C.SYMBOL}-{model}-{shape}" / side / f"exp_{k:03d}"
    gen = exp / "data_gen"
    gen.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(msg[:, :6].copy())
    ts = pd.Series(msg[:, 0].astype(np.int64))
    m[0] = (ts // L.NS).astype(str) + "." + (ts % L.NS).astype(str).str.zfill(9)
    m.to_csv(gen / f"{C.SYMBOL}_{day}_message_real_id_{k}_gen_id_{k}.csv", header=False, index=False)
    ob = book[:, :4].copy()
    ob[np.abs(ob) >= L.EMPTY] = 2147483647
    pd.DataFrame(ob).to_csv(gen / f"{C.SYMBOL}_{day}_orderbook_real_id_{k}_gen_id_{k}.csv", header=False, index=False)
    if aggr:
        np.savetxt(exp / f"aggressive_indices_{day}.csv", np.asarray(aggr, int), fmt="%d")


def jobs(sims: list, arms: list = BASE_ARMS) -> list:
    out = []
    for sim in sims:
        for arm in arms:
            if arm == "blind" and sim.startswith("abides"):
                continue  # an agent-based market cannot be made blind to an order that sits in its book
            for side in (("buy",) if arm == "noins" else ("buy", "sell")):
                out += [(sim, arm, side, k) for k in range(N_ROLLOUTS)]
    return [j for j in out if not (OUT / "rec" / j[0] / f"{j[1]}_{j[2]}_{j[3]:03d}.pkl").exists()]


# ------------------------------------------------------------------ scoring
def records(sim: str, arm: str) -> list:
    # the side follows the arm name directly, so beta never picks up the diagnostic arms beta_t, beta_x5, beta_t_x5
    out = [load(p) for side in ("buy", "sell") for p in sorted((OUT / "rec" / sim).glob(f"{arm}_{side}_*.pkl"))]
    return [r for r in out if "failed" not in r]


def score_sim(sim: str) -> dict:
    dl = daily()
    row = {"sim": sim}
    noins = records(sim, "noins")
    row.update({f"p1_{k}": v for k, v in LI.phase1(noins).items()})
    beta = records(sim, "beta")
    row.update({f"p2_{k}": v for k, v in LI.build_up(LI.points(beta), dl).items()})
    K, _ = LI.k_clock(beta, 100, 0)
    row.update({f"p2_{k}": v for k, v in LI.peak(K, 100).items()})
    K, _ = LI.k_clock(records(sim, "relax"), 10, 100)
    row.update({f"p3_{k}": v for k, v in LI.relaxation(K, 10).items()})
    blind = records(sim, "blind")
    if blind:
        row.update({f"att_{k}": v for k, v in LI.attribution(beta, blind).items()})
    row["n_failed"] = sum("failed" in load(p) for p in (OUT / "rec" / sim).glob("*.pkl"))
    return row


def stage_score(sims: list) -> pd.DataFrame:
    real = LI.phase1(load(OUT / "rec" / "real" / "noins.pkl"))
    rows = [{"sim": "real", **{f"p1_{k}": v for k, v in real.items()}}] + [score_sim(s) for s in sims if (OUT / "rec" / s).exists()]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "lob_impact_scores.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["sim", "p1_gamma_flow", "p1_H", "p1_R100_over_R1", "p1_drift_ticks", "p2_identified", "p2_delta", "p2_delta_lo", "p2_delta_hi", "p2_Y",
            "p2_peak_bps", "p2_detected", "p3_ep", "p3_slope_per_duration", "att_vb", "n_failed"]
    print(t[[c for c in cols if c in t]].round(3).to_string(index=False))
    return t


if __name__ == "__main__":
    args = sys.argv[1:]
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 4
    sims = args[args.index("--sims") + 1].split(",") if "--sims" in args else SIMS
    shard = tuple(int(x) for x in args[args.index("--shard") + 1].split("/")) if "--shard" in args else (0, 1)
    stage = args[0]
    if stage == "calib":
        stage_calib()
    elif stage == "run":
        todo = jobs(sims, args[args.index("--arms") + 1].split(",") if "--arms" in args else BASE_ARMS)
        if "--n" in args:
            todo = [j for j in todo if j[3] < int(args[args.index("--n") + 1])]
        R.pool_map(rollout_job, todo[shard[0]::shard[1]], procs, "lob-impact rollouts")
    elif stage == "score":
        stage_score(sims)
    elif stage == "export":
        # rewrite the CSV copies of the exported rollouts from their seeds, and check each reproduces its stored record
        R.pool_map(export_job, [(sim, arm, side, k) for sim in EXPORT_SIMS for arm in BASE_ARMS if not (arm == "blind" and sim.startswith("abides"))
                                for side in (("buy",) if arm == "noins" else ("buy", "sell")) for k in range(N_EXPORT)], procs, "export")
