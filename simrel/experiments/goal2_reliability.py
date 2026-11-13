"""
Design addendum 10: mechanism diagnostics (tier 1) for the trust rules (simrel/reports/lob_impact_design_addendum10.md).

An observation sample is one unperturbed run: WARMUP_S of warm-up, then SAMPLE_S (three hours, the duration of the six
30-minute episodes a candidate is fitted to), converted to a ten-level LOBSTER stream. Features of a sample:

    H, gamma_flow, R100_over_R1   LOB-Impact phase-1 estimators (simrel/metrics/lob_impact.py) on the sample's trades
    block_ac25                    lag-1 autocorrelation of mid changes over consecutive 25-trade blocks
    vr30, vr120, vr600            variance ratios of mid changes at 30, 120, 600 s against 5 s (below 1: mean reversion)
    depth_flow                    rank correlation of the best-level depth imbalance with the signed market-order
                                  volume of the preceding 30 s (liquidity withdrawal)

diag   per market: the market's own sample (seed SEED_MKT) and N_SAMPLES samples of each design in D and of the true
       generator (positive control); one small record per sample in RAW / "diag" / f"{world}_{name}_{k}.pkl".

    python simrel/experiments/goal2_reliability.py diag --procs 4 --worlds R1a_qr_resilient --names h_k1,abides
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.experiments import goal2_run as G  # noqa: E402
from simrel.metrics import lob_impact as LI  # noqa: E402

WARMUP_S, SAMPLE_S = 600, 3 * 3600
N_SAMPLES = 50
SEED_MKT, SEED_DIAG = 63_000, 64_000
FEATURES = ["H", "gamma_flow", "R100_over_R1", "block_ac25", "vr30", "vr120", "vr600", "depth_flow"]
DESIGNS = ["h_k1", "h_c_f_m_k1", "abides", "qr"]
OUT = G.RAW / "diag"


def sample_window(world: str, name: str, seed: int, length_s: float = SAMPLE_S):
    """The LOBSTER stream of one observation sample of the market (name "market"), a design, or the true generator."""
    from simrel.data import lobster as L
    from simrel.lob import Book

    R = G.R
    if name == "abides":
        from simbench import abides_world as AW

        spec = G.abides_spec(world)
        AW.WORLDS[spec.name] = spec
        ep = AW.run_episode(spec.name, seed, AW.Task(side="sell", total_qty=0, horizon_s=length_s, n_slices=1, post_s=0),
                            lambda s: 0, "none", warmup_s=WARMUP_S, keep_history=True)
        w = L.abides_to_lobster(ep.history, ep.start_ns, 0, length_s)
    else:
        if name in ("market", "oracle"):
            from simrel.reference_worlds.data import make

            model = make(world)(seed)
        else:
            model = G.make_factory(world, name)(seed)
        book = Book()
        book.tape = []
        model.init_book(book)
        model.advance(book, 0.0, WARMUP_S + length_s, [])
        w = L.tape_to_lobster(book.tape, WARMUP_S, length_s, price_unit=100)
    return R.two_sided(R.S.visible_window(w))[0]


def features(w) -> dict:
    from scipy.stats import spearmanr

    m, b = w.message, w.book
    t = m[:, 0] / 1e9
    ask, bid = b[:, 0].astype(float), b[:, 2].astype(float)
    mid = 0.5 * (ask + bid)
    mid[(ask <= 0) | (bid <= 0) | (ask >= 9e9) | (bid >= 9e9)] = np.nan
    out = {"n_events": int(len(m))}
    sm = LI.signs_and_mid({"msg": m, "ask": ask, "bid": bid})
    if sm is not None:
        p1 = LI.phase1([{"eps": sm[0].astype(np.int8), "tmid": sm[1], "n_rows": len(mid), "mid_first": np.nan, "mid_last": np.nan}])
        out.update({k: p1.get(k, np.nan) for k in ("H", "gamma_flow", "R100_over_R1")})
        tm = sm[1][np.isfinite(sm[1])]
        r = np.diff(tm[::25])
        out["block_ac25"] = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 20 and np.std(r) > 0 else np.nan
    # mid on a 5 s grid (the quote in force)
    g = t[0] + np.arange(0, t[-1] - t[0], 5.0)
    idx = np.searchsorted(t, g, side="right") - 1
    gm = mid[np.maximum(idx, 0)]
    d5 = np.diff(gm)
    v5 = np.nanvar(d5)
    for h in (30, 120, 600):
        k = h // 5
        dh = gm[k:] - gm[:-k]
        out[f"vr{h}"] = float(np.nanvar(dh) / (k * v5)) if v5 > 0 and np.isfinite(dh).sum() > 20 else np.nan
    # depth imbalance at 30 s grid points against the signed market-order volume of the preceding 30 s
    ex = (m[:, 1] == 4) & (m[:, 3] > 0)
    sv = np.where(ex, -m[:, 5] * m[:, 3], 0.0)  # buyer-initiated positive
    cum = np.cumsum(sv)
    g30 = t[0] + np.arange(30.0, t[-1] - t[0], 30.0)
    i1 = np.searchsorted(t, g30, side="right") - 1
    i0 = np.searchsorted(t, g30 - 30.0, side="right") - 1
    flow = cum[i1] - np.where(i0 >= 0, cum[np.maximum(i0, 0)], 0.0)
    bs, as_ = b[i1, 3].astype(float), b[i1, 1].astype(float)
    imb = (bs - as_) / np.maximum(bs + as_, 1.0)
    ok = np.isfinite(flow) & np.isfinite(imb)
    out["depth_flow"] = float(spearmanr(flow[ok], imb[ok]).correlation) if ok.sum() > 30 and np.std(flow[ok]) > 0 else np.nan
    return out


def diag_job(args) -> str:
    world, name, k = args
    path = OUT / f"{world}_{name}_{k}.pkl"
    if path.exists():
        return "cached"
    seed = SEED_MKT + k if name == "market" else SEED_DIAG + k
    try:
        rec = features(sample_window(world, name, seed))
    except Exception as err:
        rec = {"failed": repr(err)[:300]}
    rec.update(world=world, name=name, k=k, seed=seed)
    G.save(path, rec)
    return f"{world} {name} {k}" + (" FAILED" if "failed" in rec else "")


def diag(args) -> None:
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 4
    worlds = args[args.index("--worlds") + 1].split(",")
    names = args[args.index("--names") + 1].split(",") if "--names" in args else DESIGNS + ["oracle"]
    n = int(args[args.index("--samples") + 1]) if "--samples" in args else N_SAMPLES
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(w, "market", 0) for w in worlds] + [(w, nm, k) for w in worlds for nm in names for k in range(n)]
    jobs = [j for j in jobs if not (OUT / f"{j[0]}_{j[1]}_{j[2]}.pkl").exists()]
    G.R.pool_map(diag_job, jobs, procs, "diag")


# ------------------------------------------------------------------ trust rules (section 4; definitions: notes 11-12)
TWAP = "P0_twap"
TRUST, REC = range(0, 50), range(50, 100)
CAPS = {"h_k1": ["h_cap0.8", "h_k1", "h_cap0.95"], "h_c_f_m_k1": ["h_c_f_m_cap0.8", "h_c_f_m_k1", "h_c_f_m_cap0.95"]}
GRID = {"gated": [{"k": k} for k in (0, 1, 2, 3)],
        "agreement": [{"q": q, "g": g} for q in (0.5, 0.75, 1.0) for g in (0, 0.5, 1, 2)],
        "sensitivity": [{"g": g} for g in (0, 0.5, 1, 2)],
        "diagnostic": [{"level": lv, "g": g} for lv in (0.90, 0.9875) for g in (0, 0.5, 1, 2)]}
DEV_CORE = ["R1a_qr_resilient", "R1b_qr_fragile", "R1c_qr_thin", "R1d_qr_momentum", "R1e_qr_deep", "R2a_am_anchored", "R2b_am_trending",
            "R2c_am_anchor", "R2d_am_toxic", "R2e_am_calm"] + G.WELL_WORLDS


def dev_markets() -> list:
    from simrel.reference_worlds import ablation as AB

    return DEV_CORE + list(AB.markets())


def _matrix(w, name, size):
    from simrel.experiments import goal2_budget as GB

    return GB.matrix(w, name, size, 100)


def _ens(mats: dict, designs: list):
    """Recommendation (seeds 50-99), its ensemble gain over TWAP and standard error (seeds 0-49), per-design gains and bests."""
    pols = mats[designs[0]][0]
    rec = pd.concat([pd.Series(np.nanmean(mats[d][1][list(REC)], axis=0), index=mats[d][0]) for d in designs], axis=1).mean(axis=1)
    a = rec.round(9).sort_index().idxmin()
    gains, ses, bests = [], [], []
    for d in designs:
        p, c = mats[d]
        t = c[list(TRUST)]
        diff = t[:, p.index(TWAP)] - t[:, p.index(a)]
        diff = diff[np.isfinite(diff)]
        gains.append(float(diff.mean()) if len(diff) else np.nan)
        ses.append(float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan)
        bests.append(pd.Series(np.nanmean(t, axis=0), index=p).round(9).sort_index().idxmin())
    g = np.array(gains)
    n = len(designs)
    se = float(np.sqrt((np.nanvar(g, ddof=1) if n > 1 else 0.0) / n + np.nanmean(np.square(ses)) / n))
    return a, float(np.nanmean(g)), se, g, bests


def diag_pass(world: str, design: str, level: float) -> bool | None:
    """Whether the market's sample lies in the design's bands for every usable feature (note 7); None without data."""
    mk = OUT / f"{world}_market_0.pkl"
    if not mk.exists():
        return None
    m = G.load(mk)
    sam = [G.load(p) for p in sorted(OUT.glob(f"{world}_{design}_*.pkl"))]
    sam = [s for s in sam if "failed" not in s]
    if "failed" in m:
        return None
    if len(sam) < 45:  # note 13: failed samples are left out; too few usable samples count as not passing
        return False
    lo, hi = (1 - level) / 2, (1 + level) / 2
    for f in FEATURES:
        v = np.array([s.get(f, np.nan) for s in sam], float)
        x = m.get(f, np.nan)
        if not np.isfinite(x) or np.isfinite(v).mean() < 0.8:
            continue
        a, b = np.nanquantile(v, [lo, hi])
        if not (a <= x <= b):
            return False
    return True


def decide(rule: str, par: dict, w: str, mats: dict) -> tuple[str, bool]:
    """(strategy, departed) for one market and size."""
    D = [d for d in DESIGNS if mats.get(d) is not None]
    if rule == "diagnostic":
        D = [d for d in D if diag_pass(w, d, par["level"])]
        if not D:
            return TWAP, False
    a, g, se, gains, bests = _ens(mats, D)
    if a == TWAP:
        return TWAP, False
    if rule == "gated":
        ok = g - par["k"] * se > 0
    elif rule == "agreement":
        ok = np.mean([(b != TWAP) and (x > 0) for b, x in zip(bests, gains)]) >= par["q"] and g > par["g"]
    elif rule == "sensitivity":
        stable = all(len({_ens({v: mats[v]}, [v])[4][0] for v in vs}) == 1 for vs in CAPS.values())
        ok = stable and g > par["g"]
    else:  # diagnostic
        ok = g > par["g"]
    return (a, True) if ok else (TWAP, False)


def rules(args) -> None:
    from simrel.experiments import goal2_addendum8 as A

    worlds = dev_markets()
    truth = A.truth_tables(worlds)
    rows = []
    for w in worlds:
        for size in (0.05, 0.15):
            key = (w, size)
            if key not in truth:
                continue
            mats = {d: _matrix(w, d, size) for d in DESIGNS + [v for vs in CAPS.values() for v in vs] + ["oracle"]}
            base = {"world": w, "size": size, "core": w in DEV_CORE}
            rows.append(dict(base, rule="always TWAP", par="", regret=A.regret_of(truth, key, TWAP), departed=False))
            if mats["oracle"] is not None:
                rows.append(dict(base, rule="oracle", par="", regret=A.regret_of(truth, key, A.pick(*mats["oracle"], range(100))), departed=np.nan))
            for rule, grid in GRID.items():
                if rule == "sensitivity" and any(mats[v] is None for vs in CAPS.values() for v in vs):
                    continue
                if rule == "diagnostic" and any(diag_pass(w, d, 0.9) is None for d in DESIGNS):
                    continue
                for par in grid:
                    s, dep = decide(rule, par, w, mats)
                    rows.append(dict(base, rule=rule, par=json.dumps(par), regret=A.regret_of(truth, key, s), departed=dep, chosen=s))
    t = pd.DataFrame(rows)
    # no-simulator rule: the strategy with the lowest mean regret on the other development markets, by size
    for (w, size), _ in t.groupby(["world", "size"]):
        others = [k for k in truth if k[1] == size and k[0] != w]
        pols = truth[others[0]][0]
        best = min(pols, key=lambda p: (np.mean([A.regret_of(truth, k, p) for k in others]), p))
        t = pd.concat([t, pd.DataFrame([{"world": w, "size": size, "core": w in DEV_CORE, "rule": "no-simulator", "par": "",
                                         "regret": A.regret_of(truth, (w, size), best), "departed": best != TWAP, "chosen": best}])])
    tw = t[t.rule == "always TWAP"].set_index(["world", "size"]).regret
    t["harmful"] = [r > tw[(w, s)] + 1e-9 for w, s, r in zip(t.world, t["size"], t.regret)]
    out = G.BASE / "addendum10"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "dev_rules.csv", index=False)
    pd.set_option("display.width", 220)
    for label, sub in (("16 core development markets (selection set)", t[t.core]), ("all development markets", t)):
        per = sub.groupby(["rule", "par", "world"]).agg(regret=("regret", "mean"), departed=("departed", "mean"), harmful=("harmful", "mean")).reset_index()
        s = per.groupby(["rule", "par"]).agg(markets=("world", "nunique"), regret=("regret", "mean"), departed=("departed", "mean"),
                                             harmful=("harmful", "mean")).reset_index().sort_values("regret")
        print(f"\n{label}: mean regret (bps), departure and harmful-decision rates")
        print(s.round(3).to_string(index=False))


def false_rejection(args) -> None:
    """Positive control (amendment 1): how often the true generator fails its own diagnostic, on the development markets."""
    rows = []
    for w in dev_markets():
        sam = [G.load(p) for p in sorted(OUT.glob(f"{w}_oracle_*.pkl"))]
        sam = [s for s in sam if "failed" not in s]
        if len(sam) < 45:
            continue
        for level in (0.90, 0.9875):
            lo, hi = (1 - level) / 2, (1 + level) / 2
            own = []
            for i, x in enumerate(sam):  # each oracle sample against bands from the other oracle samples
                ok = True
                for f in FEATURES:
                    v = np.array([s.get(f, np.nan) for j, s in enumerate(sam) if j != i], float)
                    if not np.isfinite(x.get(f, np.nan)) or np.isfinite(v).mean() < 0.8:
                        continue
                    a, b = np.nanquantile(v, [lo, hi])
                    ok &= bool(a <= x[f] <= b)
                own.append(not ok)
            rows.append({"world": w, "level": level, "own samples failing": float(np.mean(own)), "market sample fails": not diag_pass(w, "oracle", level)})
    t = pd.DataFrame(rows)
    out = G.BASE / "addendum10"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "dev_false_rejection.csv", index=False)
    print("true generator's false-rejection rate on the development markets (positive control)")
    print(t.groupby("level")[["own samples failing", "market sample fails"]].mean().round(3).to_string())


# ------------------------------------------------------------------ identifiability grid (section 5)
IDENT = G.RAW / "ident"
SEED_ID6, SEED_ID24, N_ID = 66_000, 67_000, 20


def ident_job(args) -> str:
    world, hours, k = args
    path = IDENT / f"{world}_{hours}h_{k}.pkl"
    if path.exists():
        return "cached"
    seed = (SEED_ID6 if hours == 3 else SEED_ID24) + k
    try:
        rec = features(sample_window(world, "market", seed, hours * 3600.0))
    except Exception as err:
        rec = {"failed": repr(err)[:300]}
    rec.update(world=world, hours=hours, k=k, seed=seed)
    G.save(path, rec)
    return f"{world} {hours}h {k}" + (" FAILED" if "failed" in rec else "")


def ident_obs(args) -> None:
    """Observation samples of every grid market: N_ID of 3 hours (six episodes) and N_ID of 12 hours (24 episodes)."""
    from simrel.reference_worlds import identgrid

    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 4
    IDENT.mkdir(parents=True, exist_ok=True)
    jobs = [(w, h, k) for w in identgrid.markets() for h in (12, 3) for k in range(N_ID)]
    jobs = [j for j in jobs if not (IDENT / f"{j[0]}_{j[1]}h_{j[2]}.pkl").exists()]
    G.R.pool_map(ident_job, jobs, procs, "ident")


TEST_A = [f"R1s0{k}" for k in range(1, 6)] + [f"R2s0{k}" for k in range(1, 6)] + [f"R3s0{k}" for k in range(1, 7)]


def test_a(args) -> None:
    """Test A (secondary), read once with the frozen rules (results/simrel/goal2/addendum10/rules10_frozen.json); gate 1."""
    from simrel.experiments import goal2_addendum8 as A

    out = G.BASE / "addendum10"
    fz = json.loads((out / "rules10_frozen.json").read_text())
    dev = dev_markets()
    dtruth = A.truth_tables(dev)
    nosim = {}
    for size in (0.05, 0.15):
        ks = [k for k in dtruth if k[1] == size]
        pols = dtruth[ks[0]][0]
        nosim[size] = min(pols, key=lambda p: (np.mean([A.regret_of(dtruth, k, p) for k in ks]), p))
    truth = A.truth_tables(TEST_A)
    rows = []
    for w in TEST_A:
        for size in (0.05, 0.15):
            key = (w, size)
            mats = {d: _matrix(w, d, size) for d in DESIGNS + [v for vs in CAPS.values() for v in vs] + ["oracle"]}
            base = {"world": w, "size": size, "family": A.group(w) if not w.startswith("R3") else "held-out latent"}
            add = lambda rule, strat, dep: rows.append(dict(base, rule=rule, chosen=strat, departed=dep, regret=A.regret_of(truth, key, strat),
                                                            regret_cf=A.regret_of(truth, key, strat, crossfit=True)))
            add("always TWAP", TWAP, False)
            add("no-simulator", nosim[size], nosim[size] != TWAP)
            add("oracle", A.pick(*mats["oracle"], range(100)), np.nan)
            for rule, b in fz["per_rule_best"].items():
                if rule == "sensitivity" and any(mats[v] is None for vs in CAPS.values() for v in vs):
                    continue
                s_, dep = decide(rule, b["params"], w, mats)
                add(rule + (" (frozen)" if rule == fz["chosen_rule"] else ""), s_, dep)
    t = pd.DataFrame(rows)
    tw = t[t.rule == "always TWAP"].set_index(["world", "size"]).regret
    t["harmful"] = [r > tw[(w, s_)] + 1e-9 for w, s_, r in zip(t.world, t["size"], t.regret)]
    t.to_csv(out / "test_a.csv", index=False)
    pd.set_option("display.width", 200)
    per = t.groupby(["rule", "world"]).agg(regret=("regret", "mean"), regret_cf=("regret_cf", "mean"), departed=("departed", "mean"),
                                          harmful=("harmful", "mean")).reset_index()
    print("Test A (16 markets: 10 drawn, 6 held-out), frozen rules: mean regret (bps; cross-fitted truth), departure, harmful")
    print(per.groupby("rule")[["regret", "regret_cf", "departed", "harmful"]].mean().sort_values("regret").round(3).to_string())
    m = per.pivot(index="world", columns="rule", values="regret")
    tw_m, orc = m["always TWAP"], m["oracle"]
    chosen = fz["chosen_rule"] + " (frozen)"
    rng = np.random.default_rng(20261113)
    idx = [rng.integers(0, len(m), len(m)) for _ in range(2000)]
    print("\ngate 1 (95% market bootstrap, 2,000 resamples; negative = the rule has lower regret)")
    for ref in ("always TWAP", "no-simulator"):
        d = (m[chosen] - m[ref]).to_numpy()
        bs = [d[i].mean() for i in idx]
        print(f"  {chosen} minus {ref}: {d.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
    print(f"  share of the TWAP-to-oracle gap captured by the frozen rule: {(tw_m.mean() - m[chosen].mean()) / (tw_m.mean() - orc.mean()):.2f}")
    print("\nby family, mean regret")
    fam = t.groupby(["family", "rule", "world"]).regret.mean().groupby(["family", "rule"]).mean().unstack(0)
    print(fam.round(2).to_string())


if __name__ == "__main__":
    {"diag": diag, "rules": rules, "false_rejection": false_rejection, "test_a": test_a, "ident_obs": ident_obs}[sys.argv[1]](sys.argv[2:])
