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

import sys
from pathlib import Path

import numpy as np

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


def sample_window(world: str, name: str, seed: int):
    """The LOBSTER stream of one observation sample of the market (name "market"), a design, or the true generator."""
    from simrel.data import lobster as L
    from simrel.lob import Book

    R = G.R
    if name == "abides":
        from simbench import abides_world as AW

        spec = G.abides_spec(world)
        AW.WORLDS[spec.name] = spec
        ep = AW.run_episode(spec.name, seed, AW.Task(side="sell", total_qty=0, horizon_s=SAMPLE_S, n_slices=1, post_s=0),
                            lambda s: 0, "none", warmup_s=WARMUP_S, keep_history=True)
        w = L.abides_to_lobster(ep.history, ep.start_ns, 0, SAMPLE_S)
    else:
        if name in ("market", "oracle"):
            from simrel.reference_worlds.data import make

            model = make(world)(seed)
        else:
            model = G.make_factory(world, name)(seed)
        book = Book()
        book.tape = []
        model.init_book(book)
        model.advance(book, 0.0, WARMUP_S + SAMPLE_S, [])
        w = L.tape_to_lobster(book.tape, WARMUP_S, SAMPLE_S, price_unit=100)
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


if __name__ == "__main__":
    {"diag": diag}[sys.argv[1]](sys.argv[2:])
