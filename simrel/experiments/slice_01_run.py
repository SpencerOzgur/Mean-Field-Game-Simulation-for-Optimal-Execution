"""
Scenario-slice experiment on real Nasdaq order data.

    python simrel/experiments/slice_01_run.py [stage ...] [--procs N]

Stages (all are cached under results/simrel/slices/raw and can be re-run):

    slices    cut the days into ten-minute windows, describe them, assign slices
    fit       fit the Hawkes simulator to each slice; simulate the ABIDES grid and pick a configuration per slice
    gen       unperturbed simulated windows per slice and simulator, in LOBSTER format
    score     LOB-Bench realism of every simulator in every slice, and the real-against-real floor
    policy    the candidate execution policies in every slice and simulator, with unperturbed twins

With no stage given, all are run in order.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel import slices as S  # noqa: E402
from simrel.configs import slices as C  # noqa: E402
from simrel.core import Task  # noqa: E402
from simrel.data import lobster as L  # noqa: E402

OUT = S.out_dir()
RAW = OUT / "raw"
SEED_CALIB, SEED_GEN, SEED_POLICY = 20_000, 30_000, 40_000
SIMS = ["hawkes", "abides"]


# ---------------------------------------------------------------- small utilities
def save(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_windows(path: Path, pairs: list) -> None:
    arrays = {}
    for i, w in enumerate(pairs):
        arrays[f"m{i}"], arrays[f"b{i}"], arrays[f"p{i}"], arrays[f"s{i}"] = w.message, w.book, w.book_before, np.array([w.start_s, w.length_s])
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, n=len(pairs), **arrays)
    tmp.replace(path)


def load_windows(path: Path) -> list:
    d = np.load(path)
    return [L.Window(C.SYMBOL, "", int(d[f"s{i}"][0]), int(d[f"s{i}"][1]), d[f"m{i}"], d[f"b{i}"], d[f"p{i}"]) for i in range(int(d["n"]))]


class _Safe:
    """Wraps a job so an interrupt from the memory guard becomes a result instead of a dead worker."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, job):
        try:
            return self.fn(job)
        except KeyboardInterrupt:
            return f"{job}: stopped by the memory guard"


def pool_map(fn, jobs: list, procs: int, label: str) -> list:
    out, t0 = [], time.time()
    if not jobs:
        return out
    with Pool(min(procs, len(jobs)), maxtasksperchild=20) as pool:  # fresh workers now and then also release memory
        for i, res in enumerate(pool.imap_unordered(_Safe(fn), jobs, chunksize=1), 1):
            out.append(res)
            print(f"  [{label}] {i}/{len(jobs)}  {time.time() - t0:.0f}s  {res if isinstance(res, str) else ''}", flush=True)
    return out


def usable_slices() -> list:
    info = json.load(open(OUT / "slices.json"))
    return [s for s, d in info["counts"].items() if d["fit"] >= C.MIN_WINDOWS and d["test"] >= C.MIN_WINDOWS]


def slice_info() -> dict:
    return json.load(open(OUT / "slices.json"))


# ---------------------------------------------------------------- stage: slices
def stage_slices() -> None:
    feats, by_slice = [], {}
    frames = []
    for date in C.FIT_DAYS + C.TEST_DAYS:
        ws = S.load_windows([date])
        frames.append((date, ws))
        feats.append(S.feature_table(ws))
        print(f"  {date}: {len(ws)} windows", flush=True)
    f, cuts = S.assign_slices(pd.concat(feats, ignore_index=True))
    f.to_csv(OUT / "windows.csv", index=False)
    for _, ws in frames:
        for key, w in ws.items():
            row = f.loc[f["key"] == key].iloc[0]
            by_slice.setdefault((row["slice"], row["role"]), []).append(w)
    counts = {}
    for name in S.slice_names():
        counts[name] = {}
        for role in ("fit", "test"):
            pairs = by_slice.get((name, role), [])
            counts[name][role] = len(pairs)
            save_windows(RAW / f"real_{name}_{role}.npz", pairs)
        d = f[f["slice"] == name]
        counts[name].update({k: float(d[k].mean()) for k in ["volatility_bps", "volume_visible", "trend_bps", "spread_bps", "events", "depth10", "mid"]})
    json.dump({"cuts": cuts, "counts": counts}, open(OUT / "slices.json", "w"), indent=1)
    print(pd.DataFrame(counts).T.round(2).to_string())


def real_windows(name: str, role: str) -> list:
    return load_windows(RAW / f"real_{name}_{role}.npz")


def real_features(name: str, role: str) -> list:
    f = pd.read_csv(OUT / "windows.csv")
    return f[(f["slice"] == name) & (f["role"] == role)].to_dict("records")


# ---------------------------------------------------------------- stage: fit
def fit_hawkes_job(name: str) -> str:
    from simrel.adapters import hawkes_fit

    path = RAW / f"hawkes_{name}.pkl"
    if path.exists():
        return f"{name} cached"
    ws = real_windows(name, "fit")
    pick = np.random.default_rng(0).permutation(len(ws))[: C.HAWKES_MAX_FIT_WINDOWS]
    obs = [L.to_observed(ws[i]) for i in sorted(pick)]
    params, info = hawkes_fit.fit(obs, betas=C.HAWKES_BETAS, lot=100, empirical=True, n_iter=40, **C.HAWKES_WEIGHT)
    starts = [S.book_orders(w.book_before) for w in ws]
    save(path, {"params": params, "info": info, "starts": starts})
    return f"{name} beta {info['beta']} branching {info['branching_ratio']:.2f} rate {info['events_per_s']:.1f}/s"


def fit_hawkes_queue_job(name: str) -> str:
    """The queue version's fit (same windows as the base fit), and the spread-closing rate, which both fits report."""
    from simrel.adapters import hawkes_fit

    path = RAW / f"hawkes_q_{name}.pkl"
    if path.exists():
        return f"{name} cached"
    ws = real_windows(name, "fit")
    pick = np.random.default_rng(0).permutation(len(ws))[: C.HAWKES_MAX_FIT_WINDOWS]
    obs = [L.to_observed(ws[i]) for i in sorted(pick)]
    params, info = hawkes_fit.fit(obs, betas=C.HAWKES_BETAS, lot=100, empirical=True, n_iter=40, queue=True, **C.HAWKES_WEIGHT)
    save(path, {"params": params, "info": info})
    return (f"{name} beta {info['beta']} share of cancels at best {info['queue_w']:.2f} closing rate {info['close_rate_per_side']:.0f}/s "
            f"wide-spread time {100 * info['wide_spread_time_share']:.2f}%")


def fit_hawkes_multi_job(args) -> str:
    """Multi-timescale fits, with and without the queue-dependent cancels, on the same windows as the base fit."""
    from simrel.adapters import hawkes_fit

    name, queue = args
    path = RAW / f"hawkes_m{'q' if queue else ''}_{name}.pkl"
    if path.exists():
        return f"{name} cached"
    ws = real_windows(name, "fit")
    pick = np.random.default_rng(0).permutation(len(ws))[: C.HAWKES_MAX_FIT_WINDOWS]
    obs = [L.to_observed(ws[i]) for i in sorted(pick)]
    params, info = hawkes_fit.fit(obs, betas=C.HAWKES_BETAS, lot=100, empirical=True, n_iter=40, queue=queue,
                                  multi_betas=C.MULTI_BETAS, **C.HAWKES_WEIGHT)
    save(path, {"params": params, "info": info})
    return f"{name} multi{' + queue' if queue else ''}: loglik {info['loglik']:.0f}, branching before rescaling {info.get('alpha_rescaled_from', info['branching_ratio']):.2f}"


def fit_hawkes_lodo_job(args) -> str:
    """Refits with all windows of one fit day left out: single kernel and multi-timescale."""
    from simrel.adapters import hawkes_fit

    name, j = args
    path_s, path_m = RAW / f"hawkes_lodo{j}_{name}.pkl", RAW / f"hawkes_m_lodo{j}_{name}.pkl"
    if path_s.exists() and path_m.exists():
        return f"{name} lodo{j} cached"
    f = pd.read_csv(OUT / "windows.csv")
    dates = f[(f["slice"] == name) & (f["role"] == "fit")]["date"].tolist()  # same order as the stored fit windows
    ws = [w for w, d in zip(real_windows(name, "fit"), dates) if d != C.FIT_DAYS[j]]
    pick = np.random.default_rng(0).permutation(len(ws))[: C.HAWKES_MAX_FIT_WINDOWS]
    obs = [L.to_observed(ws[i]) for i in sorted(pick)]
    params, info = hawkes_fit.fit(obs, betas=C.HAWKES_BETAS, lot=100, empirical=True, n_iter=40, **C.HAWKES_WEIGHT)
    save(path_s, {"params": params, "info": info, "starts": [S.book_orders(w.book_before) for w in ws], "left_out": C.FIT_DAYS[j],
                  "n_windows": len(ws)})
    pm, im = hawkes_fit.fit(obs, betas=C.HAWKES_BETAS, lot=100, empirical=True, n_iter=40, multi_betas=C.MULTI_BETAS, **C.HAWKES_WEIGHT)
    save(path_m, {"params": pm, "info": im})
    return f"{name} without {C.FIT_DAYS[j]}: {len(ws)} windows, branching before rescaling {info.get('alpha_rescaled_from', info['branching_ratio']):.2f}"


def fit_hawkes_cap_job(args) -> str:
    """Refits on the same windows as the base fits, with a different branching-ratio cap."""
    from simrel.adapters import hawkes_fit

    name, cap = args
    tag = f"_cap{cap}"
    path_s, path_m = RAW / f"hawkes{tag}_{name}.pkl", RAW / f"hawkes_m{tag}_{name}.pkl"
    if path_s.exists() and path_m.exists():
        return f"{name} cap {cap} cached"
    ws = real_windows(name, "fit")
    pick = np.random.default_rng(0).permutation(len(ws))[: C.HAWKES_MAX_FIT_WINDOWS]
    obs = [L.to_observed(ws[i]) for i in sorted(pick)]
    p, info = hawkes_fit.fit(obs, betas=C.HAWKES_BETAS, lot=100, empirical=True, n_iter=40, max_branching=float(cap), **C.HAWKES_WEIGHT)
    save(path_s, {"params": p, "info": info, "starts": [S.book_orders(w.book_before) for w in ws]})
    pm, im = hawkes_fit.fit(obs, betas=C.HAWKES_BETAS, lot=100, empirical=True, n_iter=40, max_branching=float(cap), multi_betas=C.MULTI_BETAS, **C.HAWKES_WEIGHT)
    save(path_m, {"params": pm, "info": im})
    return f"{name} cap {cap}: branching {info['branching_ratio']:.2f}"


def abides_vol_job(args) -> str:
    """One unperturbed ABIDES-calibrated window with a given fundamental volatility and momentum-trader count."""
    name, fv, mom, seed = args
    path = RAW / "abides_vol" / f"{name}_{fv:g}_{mom}_{seed}.pkl"
    if path.exists():
        return "cached"
    spec = abides_quoting_spec(name)
    spec = dataclasses.replace(spec, name=f"{spec.name}_fv{fv:g}_mom{mom}", fund_vol=fv, num_momentum_agents=mom)
    try:
        row = L.features(abides_window(spec, seed))
    except Exception as err:
        row = {"failed": repr(err)[:200]}
    save(path, row)
    return f"{name} fund_vol {fv:g} momentum {mom} seed {seed}: " + (f"volatility {row['volatility_bps']:.2f}" if "failed" not in row else "FAILED")


def stage_abides_vol(procs: int) -> None:
    (RAW / "abides_vol").mkdir(exist_ok=True)
    grid = list(itertools.product(C.ABIDES_VOL_GRID["fund_vol"], C.ABIDES_VOL_GRID["momentum"]))
    jobs = [(n, fv, m, SEED_CALIB + 500 + k) for n in usable_slices() for fv, m in grid for k in range(C.ABIDES_VOL_SEEDS)]
    pool_map(abides_vol_job, [j for j in jobs if not (RAW / "abides_vol" / f"{j[0]}_{j[1]:g}_{j[2]}_{j[3]}.pkl").exists()], procs, "abides volatility")
    chosen, rows = {}, []
    for n in usable_slices():
        target = float(np.mean([f["volatility_bps"] for f in real_features(n, "fit")]))
        best = None
        for fv, m in grid:
            r = [load(RAW / "abides_vol" / f"{n}_{fv:g}_{m}_{SEED_CALIB + 500 + k}.pkl") for k in range(C.ABIDES_VOL_SEEDS)]
            r = [x for x in r if "failed" not in x]
            if len(r) < C.ABIDES_VOL_SEEDS:
                continue
            v = float(np.mean([x["volatility_bps"] for x in r]))
            rows.append({"slice": n, "fund_vol": fv, "momentum": m, "volatility_bps": v, "target": target})
            if best is None or abs(v - target) < best[0]:
                best = (abs(v - target), fv, m)
        chosen[n] = {"fund_vol": best[1], "momentum": best[2]}
        print(f"  {n}: target volatility {target:.2f} bps -> fund_vol {best[1]:g}, momentum {best[2]}", flush=True)
    json.dump(chosen, open(OUT / "abides_vol_calibration.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv(OUT / "abides_vol_calibration.csv", index=False)


def abides_specs() -> list:
    from simbench.abides_world import WorldSpec

    r_bar = int(round(np.mean([d["mid"] for d in slice_info()["counts"].values()]) * 100))
    specs = []
    for nv, (nm, pov, wake), fv in itertools.product(*C.ABIDES_GRID.values()):
        specs.append(WorldSpec(name=f"S_v{nv}_m{nm}p{pov:g}w{wake}_f{fv:g}", description="slice calibration candidate", num_value_agents=nv,
                               num_mm=nm, mm_pov=pov, mm_wake_up_freq=wake, fund_vol=fv, r_bar=r_bar))
    return specs


def abides_window(spec, seed: int) -> L.Window:
    from simbench import abides_world as AW

    AW.WORLDS[spec.name] = spec
    t = C.TASK
    ep = AW.run_episode(spec.name, seed, AW.Task(side=t.side, total_qty=0, horizon_s=t.horizon_s, n_slices=t.n_slices, post_s=t.post_s),
                        lambda s: 0, "none", warmup_s=C.ABIDES_WARMUP_S, keep_history=True)
    return S.visible_window(L.abides_to_lobster(ep.history, ep.start_ns, 0, t.horizon_s))


def abides_grid_job(args) -> str:
    spec, seed = args
    path = RAW / "abides_grid" / f"{spec.name}_{seed}.pkl"
    if path.exists():
        return f"{spec.name} cached"
    try:
        row = L.features(abides_window(spec, seed))
    except Exception as err:  # a configuration that empties its book or never trades is recorded, not hidden
        row = {"failed": repr(err)[:200]}
    save(path, row)
    return f"{spec.name} {seed}" + (" FAILED" if "failed" in row else "")


def stage_fit(procs: int) -> None:
    names = usable_slices()
    pool_map(fit_hawkes_job, names, procs, "hawkes fit")
    pool_map(fit_hawkes_queue_job, names, procs, "hawkes queue fit")
    pool_map(fit_hawkes_multi_job, [(n, q) for n in names for q in (False, True)], procs, "hawkes multi-timescale fit")
    (RAW / "abides_grid").mkdir(exist_ok=True)
    specs = abides_specs()
    pool_map(abides_grid_job, [(s, SEED_CALIB + k) for s in specs for k in range(C.ABIDES_CALIB_SEEDS)], procs, "abides grid")
    grid = {}
    for s in specs:
        rows = [load(RAW / "abides_grid" / f"{s.name}_{SEED_CALIB + k}.pkl") for k in range(C.ABIDES_CALIB_SEEDS)]
        grid[s.name] = [r for r in rows if "failed" not in r]
    chosen, table = {}, []
    for name in names:
        target = S.moments(real_features(name, "fit"))
        errs = {}
        for s in specs:
            if len(grid[s.name]) < C.ABIDES_CALIB_SEEDS:
                continue
            errs[s.name], parts = S.moment_error(grid[s.name], target)
            table.append({"slice": name, "config": s.name, "error": errs[s.name], **parts})
        best = min(errs, key=errs.get)
        chosen[name] = dataclasses.asdict(next(s for s in specs if s.name == best))
        print(f"  ABIDES for {name}: {best}  moment error {errs[best]:.2f}", flush=True)
    json.dump(chosen, open(OUT / "abides_chosen.json", "w"), indent=1)
    pd.DataFrame(table).to_csv(OUT / "abides_calibration.csv", index=False)
    rows = []
    for name in names:
        h = load(RAW / f"hawkes_{name}.pkl")
        rows.append({"slice": name, **{k: v for k, v in h["info"].items() if not isinstance(v, (dict, list))}})
    pd.DataFrame(rows).to_csv(OUT / "hawkes_fits.csv", index=False)


# ---------------------------------------------------------------- simulators
def hawkes_model(name: str, version: str = "hawkes", impact_scale: float | None = None, fit_day_starts: bool = False, starts: list | None = None):
    """Returns make_model(seed). Each seed starts from the real book at the start of one of the slice's windows.

    The base version starts from fit-day windows. With same_day, it starts from test-day windows (the simulator is
    conditioned on the opening book of the period it is scored against, as LOB-Bench conditions a generator on a real
    prefix) and the depth at which cancellations balance is set to that book's depth, so depth does not drift back to
    the fit days' level. The fitted intensities and marks always come from the fit days only.

    starts, if given, replaces the start books (list of (orders, p0)); same_day versions then condition their
    cancellation depth on the given book (used by the LOB-Impact rollouts, which start from every day's windows).
    """
    from simrel.adapters.hawkes import MB, MS, HawkesLOB
    from simrel.adapters.hawkes_fast import FastHawkesLOB

    flags = C.HAWKES_VERSIONS[version]
    tag = flags.get("fit_tag", "")  # refits with one fit day left out
    h = load(RAW / f"hawkes{tag}_{name}.pkl")
    base = h["params"]
    if flags.get("multi"):  # the multi-timescale fit replaces the single-kernel one (queue variant if asked)
        base = load(RAW / f"hawkes_m{'q' if flags.get('queue') else ''}{tag}_{name}.pkl")["params"]
    if flags.get("queue") or flags.get("close") or flags.get("size"):
        hq = load(RAW / f"hawkes_q_{name}.pkl")
        if flags.get("queue") and not flags.get("multi"):
            base = hq["params"]
        if flags.get("close"):
            rate = (h if tag else hq)["info"]["close_rate_per_side"]
            lam_max = np.array(base.lam_max, dtype=float)
            lam_max[:2] += rate
            base = dataclasses.replace(base, close_rate=rate, lam_max=lam_max)
        if flags.get("size"):  # the ratio table is estimated in the queue fit, which logs the best queues
            base = dataclasses.replace(base, market_ratio_q=hq["params"].market_ratio_q)
    if not flags.get("size"):  # fits made after 2026-10-27 also carry the ratio table; it is used only by "size" versions
        base = dataclasses.replace(base, market_ratio_q=None)
    if flags.get("fill"):
        lam_max = np.array(base.lam_max, dtype=float)
        lam_max[:2] += C.FILL_RATE
        base = dataclasses.replace(base, fill_rate=C.FILL_RATE, lam_max=lam_max)
    if flags.get("impact") or impact_scale is not None:  # scale the excitation caused by market orders
        s = impact_scale if impact_scale is not None else json.load(open(OUT / "impact_calibration.json"))[version.removesuffix("_k")][name]
        alpha = np.array(base.alpha, dtype=float)
        alpha[:, [MB, MS]] *= s
        stack = None
        if base.alpha_stack is not None:
            stack = np.array(base.alpha_stack, dtype=float)
            stack[:, :, [MB, MS]] *= s
        # Fits sit at the stability limit (branching ratio rescaled to 0.9), so a factor above 1 is a reallocation:
        # trades get a larger share of the excitation and the matrix is scaled back to the original branching ratio.
        b0 = float(np.max(np.abs(np.linalg.eigvals(np.asarray(base.alpha)))))
        b1 = float(np.max(np.abs(np.linalg.eigvals(alpha))))
        if b1 > b0:
            alpha *= b0 / b1
            if stack is not None:
                stack *= b0 / b1
        base = dataclasses.replace(base, alpha=alpha, alpha_stack=stack)
    cls = FastHawkesLOB if (C.FAST_HAWKES or flags.get("multi") or flags.get("impact") or flags.get("fast") or tag or impact_scale is not None) else HawkesLOB
    same_day = flags.get("same_day") and not fit_day_starts
    if starts is None:
        starts = [S.book_orders(w.book_before) for w in real_windows(name, "test")] if same_day else h["starts"]

    def make(seed: int):
        orders, p0 = starts[seed % len(starts)]
        extra = {}
        if same_day or (flags.get("same_day") and fit_day_starts):
            depth = max(0.5 * sum(q for _, _, q in orders), 1000.0)
            ratio = depth / base.ref_depth
            extra = {"ref_depth": depth, "ref_q1": base.ref_q1 * ratio, "ref_rest": base.ref_rest * ratio}
        return cls(dataclasses.replace(base, init_orders=orders, p0=p0, **extra), seed)

    return make


def abides_spec(name: str):
    from simbench.abides_world import WorldSpec

    return WorldSpec(**json.load(open(OUT / "abides_chosen.json"))[name])


def abides_quoting_spec(name: str, base_spec=None):
    """ABIDES-calibrated: the slice's ABIDES configuration plus a population of quoting agents whose order rate,
    price levels and sizes are those of the slice's fit windows, and whose quote lifetime makes the resting depth
    (top ten levels, both sides) equal to the fit windows' depth: depth = rate * lifetime * mean size."""
    spec = base_spec if base_spec is not None else abides_spec(name)
    feats = real_features(name, "fit")
    ws = real_windows(name, "fit")
    adds = np.concatenate([w.message[w.message[:, 1] == 1, 3] for w in ws])
    rate = len(adds) / (len(ws) * C.WINDOW_S)
    depth = 2.0 * float(np.mean([f["depth10"] for f in feats]))
    h = load(RAW / f"hawkes_{name}.pkl")["params"]
    pick = np.linspace(0, len(h.level_q) - 1, 500).astype(int)
    # quote lifetimes by placement level (0, 1, 2 or more ticks behind the best): time from placement to deletion of
    # every order placed and deleted inside a fit window
    life = {0: [], 1: [], 2: []}
    for w in ws:
        m = w.message
        prev = np.vstack([w.book_before[None, :], w.book[:-1]])
        add = np.nonzero(m[:, 1] == 1)[0]
        best = np.where(m[add, 5] == 1, prev[add, 2], prev[add, 0])
        lvl = np.clip(np.where(m[add, 5] == 1, best - m[add, 4], m[add, 4] - best) // L.TICK, 0, 2)
        placed = {int(o): (t, int(v)) for o, t, v in zip(m[add, 2], m[add, 0], lvl)}
        for o, t in zip(m[m[:, 1] == 3, 2], m[m[:, 1] == 3, 0]):
            if int(o) in placed:
                life[placed[int(o)][1]].append((t - placed[int(o)][0]) / 1e9)
    life = tuple(tuple(float(x) for x in np.quantile(np.asarray(life[k]), (np.arange(1000) + 0.5) / 1000)) for k in (0, 1, 2))
    n_agents = C.ABIDES_QUOTING_AGENTS
    return dataclasses.replace(spec, name=spec.name + "_quoting", num_quoting=n_agents, quoting_rate=rate / n_agents,
                               quoting_lifetime_s=depth / (rate * float(adds.mean())), quoting_lifetimes=life,
                               quoting_levels=tuple(int(x) for x in h.level_q[pick]), quoting_sizes=tuple(int(x) for x in h.limit_size_q[pick]))


def abides_alt_spec(name: str, rank: int):
    """The rank-th best ABIDES configuration for a slice by fit-day moment error (rank 0 is the chosen one)."""
    cal = pd.read_csv(OUT / "abides_calibration.csv")
    cal = cal[cal["slice"] == name].sort_values("error")
    target = cal["config"].iloc[rank]
    return next(s for s in abides_specs() if s.name == target)


def abides_family_spec(sim: str, name: str, momentum: int | None = None):
    if sim == "abides_quoting_vol":
        c = json.load(open(OUT / "abides_vol_calibration.json"))[name]
        spec = abides_quoting_spec(name)
        return dataclasses.replace(spec, name=f"{spec.name}_fv{c['fund_vol']:g}_mom{c['momentum']}", fund_vol=c["fund_vol"], num_momentum_agents=c["momentum"])
    if "__alt" in sim:
        rank = int(sim.split("__alt")[1])
        base = abides_alt_spec(name, rank)
        return abides_quoting_spec(name, base) if sim.startswith("abides_quoting") else base
    if sim == "abides_quoting_mom" or momentum is not None:  # ABIDES-calibrated with the momentum count chosen on fit-day impact
        m = momentum if momentum is not None else json.load(open(OUT / "impact_calibration.json"))["abides_quoting"][name]
        spec = abides_quoting_spec(name)
        return dataclasses.replace(spec, name=f"{spec.name}_mom{m}", num_momentum_agents=int(m))
    return abides_quoting_spec(name) if sim == "abides_quoting" else abides_spec(name)


def unperturbed_window(sim: str, name: str, seed: int) -> L.Window:
    if sim.startswith("abides"):
        return abides_window(abides_family_spec(sim, name), seed)
    from simrel.lob import Book

    model, book = hawkes_model(name, sim)(seed), Book()
    book.tape = []
    model.init_book(book)
    model.advance(book, 0.0, C.HAWKES_WARMUP_S + C.WINDOW_S, [])
    return S.visible_window(L.tape_to_lobster(book.tape, C.HAWKES_WARMUP_S, C.WINDOW_S))


# ---------------------------------------------------------------- stage: gen
def gen_job(args) -> str:
    sim, name, seed = args
    path = RAW / "gen" / f"{sim}_{name}_{seed}.npz"
    if path.exists():
        return "cached"
    try:
        w = unperturbed_window(sim, name, seed)
        save_windows(path, [w])
        return f"{sim} {name} {seed}: {len(w.message)} events"
    except Exception as err:
        save(path.with_suffix(".failed"), repr(err)[:300])
        return f"{sim} {name} {seed}: FAILED {err!r}"[:160]


def n_gen(name: str) -> int:
    return C.N_GEN


def stage_gen(procs: int, sims: list) -> None:
    (RAW / "gen").mkdir(exist_ok=True)
    jobs = [(sim, name, SEED_GEN + k) for name in usable_slices() for sim in sims for k in range(n_gen(name))]
    jobs = [j for j in jobs if not (RAW / "gen" / f"{j[0]}_{j[1]}_{j[2]}.npz").exists() and not (RAW / "gen" / f"{j[0]}_{j[1]}_{j[2]}.failed").exists()]
    pool_map(gen_job, jobs, procs, "gen")


def generated(sim: str, name: str) -> list:
    out = []
    for k in range(n_gen(name)):
        p = RAW / "gen" / f"{sim}_{name}_{SEED_GEN + k}.npz"
        if p.exists():
            out.extend(load_windows(p))
    return out


# ---------------------------------------------------------------- stage: score
def _memory_guard(path: Path, limit_gb: float = 5.0) -> None:
    """Stop this job if the worker grows past limit_gb, so a pathological input cannot exhaust the machine's memory.

    The job is interrupted (KeyboardInterrupt in the worker's main thread, turned into a failure by pool_map), so the
    pool does not wait forever for a lost task."""
    import _thread
    import resource
    import threading

    start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30

    def watch():
        while True:
            time.sleep(2)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30 > max(limit_gb, start + limit_gb * 0.5):  # bytes on macOS
                save(path, {"failed": f"stopped: memory above {limit_gb} GB"})
                _thread.interrupt_main()
                return

    threading.Thread(target=watch, daemon=True).start()


def two_sided(w: L.Window) -> tuple[L.Window, float]:
    """Drop events recorded while one side of the book was empty; returns the window and the share dropped.

    An empty side is written as a price of +-9,999,999,999, which would turn spreads and distances from the mid
    into astronomically large numbers. Real windows never contain such rows.
    """
    ok = (w.book[:, 0] < L.EMPTY) & (w.book[:, 2] > -L.EMPTY)
    return L.Window(w.symbol, w.date, w.start_s, w.length_s, w.message[ok], w.book[ok], w.book_before), float(1.0 - ok.mean())


def score_job(args) -> str:
    from simrel.metrics import lobbench

    source, name = args
    path = RAW / "score" / f"{source}_{name}.pkl"
    if path.exists():
        return "cached"
    _memory_guard(path)
    def subset(ws):
        keep = sorted(np.random.default_rng(1).permutation(len(ws))[: C.MAX_SCORE_WINDOWS])
        return [ws[i] for i in keep]

    real = subset(real_windows(name, "test"))
    if source == "real_fit_days":  # floor: real windows of the same slice from other days, scored exactly like a simulator
        gen = subset(real_windows(name, "fit"))
    else:
        gen = generated(source, name)
    cleaned = [two_sided(g) for g in gen]
    empty_share = float(np.mean([c[1] for c in cleaned])) if cleaned else 0.0
    gen = [c[0] for c in cleaned if len(c[0].message) > 50]
    if len(gen) < len(real):
        real = real[: len(gen)]
    if not gen:
        save(path, {"failed": "no generated windows"})
        return f"{source} {name}: nothing to score"
    try:
        table = lobbench.score([(w.message, w.book) for w in real], [(w.message, w.book) for w in gen])
        save(path, {"table": table, "aggregate": lobbench.aggregate(table), "n_real": len(real), "n_gen": len(gen), "empty_side_share": empty_share,
                    "features": [L.features(w) for w in gen], "real_features": [L.features(w) for w in real]})
        return f"{source} {name}: L1 {lobbench.aggregate(table)['l1_mean']:.3f}"
    except Exception as err:
        save(path, {"failed": traceback.format_exc()[-800:]})
        return f"{source} {name}: FAILED {err!r}"[:160]


def stage_score(procs: int, sims: list) -> None:
    (RAW / "score").mkdir(exist_ok=True)
    jobs = [(src, name) for name in usable_slices() for src in ["real_fit_days"] + sims]
    jobs = [j for j in jobs if not (RAW / "score" / f"{j[0]}_{j[1]}.pkl").exists()]
    pool_map(score_job, jobs, min(procs, 4), "score")


# ---------------------------------------------------------------- stage: policy
def parent_qty(name: str) -> int:
    v = slice_info()["counts"][name]["volume_visible"]
    return max(int(round(C.SIZE_FRAC * v / 100.0)) * 100, 100)


def policy_job(args) -> str:
    from simrel.metrics.cost import episode_metrics
    from simrel.policies.library import candidate_set

    sim, name, seed = args
    path = RAW / "policy" / f"{sim}_{name}_{seed}{C.SIZE_TAG}.pkl"
    if path.exists():
        return "cached"
    task = Task(side=C.TASK.side, total_qty=parent_qty(name), horizon_s=C.TASK.horizon_s, n_slices=C.TASK.n_slices, post_s=C.TASK.post_s)
    policies = candidate_set(C.SIZE_FRAC)
    if sim not in SIMS or C.POLICY_SUBSET_ONLY:  # later versions of a simulator run the policy subset only
        policies = {k: v for k, v in policies.items() if k in C.VERSION_POLICIES}
    t0 = time.time()
    try:
        if sim.startswith("abides"):
            from simbench import abides_world as AW
            from simrel.adapters import abides

            spec = abides_family_spec(sim, name)
            AW.WORLDS[spec.name] = spec
            recs = abides.run_twin_set(spec.name, seed, task, policies)
        else:
            from simrel.eventsim import run_twin_set

            recs = run_twin_set(hawkes_model(name, sim), f"{sim}_{name}", seed, task, policies, warmup_s=C.HAWKES_WARMUP_S, pre_s=60.0)
        twin = recs["none"]
        rows = [{"sim": sim, "slice": name, "size_frac": C.SIZE_FRAC, **episode_metrics(r, twin)} for k, r in recs.items() if k != "none"]
        g = (twin.grid_t >= 0) & (twin.grid_t <= task.horizon_s)
        tw = {"sim": sim, "slice": name, "seed": seed, "failed": twin.failed}
        if not twin.failed:
            tw.update({"move_bps": float((twin.grid_mid[g][-1] - twin.grid_mid[g][0]) / twin.arrival_mid * 1e4),
                       "spread_bps": float(np.nanmean(twin.grid_spread[g]) / twin.arrival_mid * 1e4), "volume": float(twin.background_volume)})
        save(path, {"rows": rows, "twin": tw})
        return f"{sim} {name} {seed}  {time.time() - t0:.0f}s"
    except Exception as err:
        save(path, {"rows": [], "twin": {"sim": sim, "slice": name, "seed": seed, "failed": repr(err)[:300]}})
        return f"{sim} {name} {seed}: FAILED {err!r}"[:160]


def stage_policy(procs: int, sims: list, shard: tuple = (0, 1)) -> None:
    """shard (i, n): run only every n-th job starting at i, so several machines can share the work."""
    (RAW / "policy").mkdir(exist_ok=True)
    jobs = [(sim, name, SEED_POLICY + k) for sim in sims for k in range(C.N_POLICY_SEEDS if sim in SIMS else C.N_VERSION_SEEDS) for name in usable_slices()]
    jobs = [j for j in jobs if not (RAW / "policy" / f"{j[0]}_{j[1]}_{j[2]}{C.SIZE_TAG}.pkl").exists()][shard[0]::shard[1]]
    pool_map(policy_job, jobs, procs, "policy")


if __name__ == "__main__":
    args = sys.argv[1:]
    procs = 6
    if "--procs" in args:
        i = args.index("--procs")
        procs = int(args[i + 1])
        del args[i: i + 2]
    sims = SIMS
    if "--sims" in args:
        i = args.index("--sims")
        sims = args[i + 1].split(",")
        del args[i: i + 2]
    shard = (0, 1)
    if "--shard" in args:
        i = args.index("--shard")
        shard = tuple(int(x) for x in args[i + 1].split("/"))
        del args[i: i + 2]
    stages = args or ["slices", "fit", "gen", "score", "policy"]
    for st in stages:
        print(f"== {st}", flush=True)
        if st == "slices":
            stage_slices()
        elif st == "fit":
            stage_fit(procs)
        elif st == "gen":
            stage_gen(procs, sims)
        elif st == "score":
            stage_score(procs, sims)
        elif st == "policy":
            stage_policy(procs, sims, shard)
        elif st == "abidesvol":
            stage_abides_vol(procs)
        elif st == "capfits":
            pool_map(fit_hawkes_cap_job, [(n, c) for n in usable_slices() for c in ("0.8", "0.95")], procs, "cap refits")
        else:
            raise SystemExit(f"unknown stage {st}")
