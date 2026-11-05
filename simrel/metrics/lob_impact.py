"""
LOB-Impact estimators (Nigmatulin et al., "LOB-Impact Benchmark: Emergent Market Impact in Generative Limit Order
Book Models", ICAIF 2026, arXiv:2610.27100; code github.com/george-nigmatulin/lob_impact @0aaf6bf).

Re-implemented from the paper and the reference code, constant for constant, so the estimators run on our rollouts
in memory. Where the reference code is silent (the amplitude Y and the direct least-squares identifiability test are
described only in the paper) the paper's text is followed and the choice is noted. The reference tool can be run on
the CSV copies of our rollouts (impact_01_run.py export) to check this port.

A rollout is reduced at generation time (reduce) to a small record that every estimator reads. A rollout is a
dict with
    msg     (n, 6)  LOBSTER message rows: time (s), event type, order id, size, price, direction (resting side)
    ask     (n,)    best ask after each row (LOBSTER price units; NaN for an empty side)
    bid     (n,)    best bid after each row
    aggr    (k,)    0-based rows of the injected child orders (empty for a free run)
    day     str     trading day the rollout is labelled with (its conditioning day)
    sign    +1 for a buy metaorder, -1 for a sell metaorder

Phase 1 (free runs): trade-sign memory gamma_flow, Hurst exponent H, drift-corrected response Rc(l), free drift.
Phase 2 (build-up, 100 children): per-child points (q, I), build-up exponent delta (signed binned estimator with a
day-clustered bootstrap), identifiability by direct least squares, amplitude Y, peak impact and its detection.
Phase 3 (relaxation, 10 children then 100 cooling windows): k-clock master curve, end-over-peak e/p, post-execution
slope. Attribution: visible final displacement over blind mechanical part (v/b).
"""

from __future__ import annotations

import numpy as np

TICK = 100.0
# Phase 1 (lob_impact.flow_balance, scorecard, drift)
LMAX, FIT_LO, FIT_HI = 300, 2, 40
MIN_EXEC = 30
R_LAGS = 200
COLLAPSE_TICKS = 5000.0
# Phase 2 (lob_impact.delta, points, trajectory)
NBINS, MIN_BINS, N_BOOT, SEED = 18, 5, 400, 20260820
DELTA_GRID = np.round(np.arange(0.05, 1.5 + 1e-9, 0.01), 2)  # paper, section 2.1
Y_FLOOR = 0.05
KAPPA_P = np.sqrt(4 * np.log(2))
COLLAPSE_BPS = 2000.0
DETECTION_SE = 3.0
# Phase 3 (lob_impact.master_curve, decay_shape)
MIN_COUNT, MIN_FRAC = 30, 0.3
Z0 = 0.2


# ------------------------------------------------------------------ helpers
def mid_of(r: dict) -> np.ndarray:
    ask, bid = np.asarray(r["ask"], float), np.asarray(r["bid"], float)
    mid = 0.5 * (ask + bid)
    mid[~np.isfinite(ask) | ~np.isfinite(bid) | (ask <= 0) | (bid <= 0)] = np.nan
    return mid


def signs_and_mid(r: dict):
    """Trade signs and the mid after each trade; one aggressive order (a same-time, same-sign burst) is one trade."""
    m = r["msg"]
    mid = mid_of(r)
    ex = np.flatnonzero((m[:, 1] == 4) & (m[:, 3] > 0))
    if len(ex) < MIN_EXEC:
        return None
    eps = -np.sign(m[ex, 5])
    ts = m[ex, 0]
    keep = np.ones(len(ex), bool)
    keep[1:] = ~((np.diff(ts) == 0) & (eps[1:] == eps[:-1]))
    last = np.ones(len(ex), bool)
    last[:-1] = keep[1:]
    return eps[last], mid[ex][last]


def reduce(r: dict, n_ins: int = 0, n_cool: int = 0) -> dict:
    """The few numbers of a rollout that the estimators read, so rollouts need not be stored.

    eps, tmid      trade signs and the mid after each trade (phase 1)
    mid_first/last first and last finite mid (free drift, final displacement)
    child_q        child sizes; mid_ref the mid on the row before the first child
    mid_child      mid on each child row; mid_pre the mid on the row before each child (mechanical part)
    kmid           mid at the k-clock rows: the children, then n_cool cooling-window ends (window = smallest child gap)
    """
    mid = mid_of(r)
    sm = signs_and_mid(r)
    f = np.flatnonzero(np.isfinite(mid))
    rec = {"day": r["day"], "sign": int(r["sign"]), "n_rows": int(len(mid)),
           "eps": sm[0].astype(np.int8) if sm else np.zeros(0, np.int8), "tmid": sm[1] if sm else np.zeros(0),
           "mid_first": float(mid[f[0]]) if len(f) else np.nan, "mid_last": float(mid[f[-1]]) if len(f) else np.nan,
           "mid_row0": float(mid[0]) if len(mid) else np.nan}
    idx = np.asarray(r.get("aggr", []), int)
    idx = idx[idx < len(mid)]
    if len(idx) >= 2 and idx[0] >= 1 and np.all(r["msg"][idx, 1] == 4) and not np.any(r["msg"][idx, 3] <= 0):
        rec.update(child_q=r["msg"][idx, 3].astype(float), mid_ref=float(mid[idx[0] - 1]), mid_child=mid[idx], mid_pre=mid[idx - 1])
        if n_ins and len(idx) >= n_ins:
            mb = int(np.min(np.diff(idx[:n_ins])))
            rows = np.r_[idx[:n_ins], idx[n_ins - 1] + mb * np.arange(1, n_cool + 1)].astype(int)
            rec["kmid"] = np.where(rows < len(mid), mid[np.minimum(rows, len(mid) - 1)], np.nan)
    return rec


# ------------------------------------------------------------------ phase 1
def acf_gamma(runs) -> tuple[np.ndarray, float]:
    num = np.zeros(LMAX)
    den = 0.0
    for eps, _ in runs:
        e = eps - eps.mean()
        den += float(np.dot(e, e))
        for l in range(1, min(LMAX, len(e) - 1) + 1):
            num[l - 1] += float(np.dot(e[:-l], e[l:]))
    C = num / max(den, 1e-12)
    if C[0] > 0:
        Cn = C / C[0]
        C = Cn if np.nanmax(np.abs(Cn)) <= 1.0 + 1e-9 else np.full_like(C, np.nan)
    ls = np.arange(1, LMAX + 1)
    m = (ls >= FIT_LO) & (ls <= FIT_HI) & (C > 0)
    gamma = -float(np.polyfit(np.log(ls[m]), np.log(C[m]), 1)[0]) if m.sum() > 10 else np.nan
    return C, gamma


def var_lags() -> np.ndarray:
    return np.unique(np.round(np.logspace(0, np.log10(LMAX), 24)).astype(int))


def var_H(runs) -> tuple[np.ndarray, float]:
    ls = var_lags()
    s, c = np.zeros(len(ls)), np.zeros(len(ls))
    for _, mid in runs:
        m = mid[np.isfinite(mid)]
        for i, l in enumerate(ls):
            if len(m) > l + 10:
                d = m[l:] - m[:-l]
                s[i] += float(np.nansum(d * d))
                c[i] += len(d)
    V = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    m = np.isfinite(V) & (V > 0) & (ls >= 2) & (ls <= 100)
    H = float(np.polyfit(np.log(ls[m]), np.log(V[m]), 1)[0]) / 2.0 if m.sum() > 6 else np.nan
    return V, H


def response(runs) -> np.ndarray:
    """Drift-corrected response Rc(l), l = 1..200, in ticks."""
    ls = np.arange(1, R_LAGS + 1)
    s, c, sm, se = (np.zeros(len(ls)) for _ in range(4))
    for eps, mid in runs:
        for i, l in enumerate(ls):
            if len(mid) > l + 5:
                dm, e = mid[l:] - mid[:-l], eps[:-l]
                d = dm * e
                ok = np.isfinite(d)
                s[i] += float(np.sum(d[ok]))
                c[i] += int(ok.sum())
                sm[i] += float(np.sum(dm[ok]))
                se[i] += float(np.sum(e[ok]))
    cc = np.maximum(c, 1)
    return np.where(c > 0, (s / cc - (se / cc) * (sm / cc)) / TICK, np.nan)


def free_drift(recs) -> tuple[float, float, int]:
    """Mean end-to-end mid displacement in ticks over free runs (collapsed books excluded), its s.e. and count."""
    out = []
    for r in recs:
        if r["n_rows"] < 100 or not np.isfinite(r["mid_first"]) or not np.isfinite(r["mid_last"]):
            continue
        d = (r["mid_last"] - r["mid_first"]) / TICK
        if abs(d) <= COLLAPSE_TICKS:
            out.append(d)
    out = np.asarray(out)
    return (float(out.mean()), float(out.std() / np.sqrt(len(out))), len(out)) if len(out) else (np.nan, np.nan, 0)


def phase1(recs) -> dict:
    """gamma_flow, H, H on the 2-10 and 10-30 trade bands, R(1), R(100), R(100)/R(1), free drift (records from reduce)."""
    runs = [(r["eps"].astype(float), r["tmid"]) for r in recs if len(r["eps"]) >= MIN_EXEC]
    if not runs:
        return {"n_runs": 0}
    _, gamma = acf_gamma(runs)
    V, H = var_H(runs)
    Rc = response(runs)
    ls = var_lags()
    bands = {}
    for lo, hi in ((2, 10), (10, 30)):
        m = np.isfinite(V) & (V > 0) & (ls >= lo) & (ls <= hi)
        bands[f"H_{lo}_{hi}"] = float(np.polyfit(np.log(ls[m]), np.log(V[m]), 1)[0]) / 2.0 if m.sum() >= 3 else np.nan
    d, d_se, n_d = free_drift(recs)
    r1, r100 = float(Rc[0]), float(Rc[99])
    return {"n_runs": len(runs), "gamma_flow": gamma, "H": H, **bands, "R1": r1, "R100": r100,
            "R100_over_R1": r100 / r1 if np.isfinite(r1) and r1 != 0 else np.nan, "drift_ticks": d, "drift_se": d_se}


# ------------------------------------------------------------------ phase 2: points and the build-up exponent
def points(recs) -> dict:
    """One signed point per child: cumulative shares Q_k and relative impact I_k = sign (mid_k - mid_ref) / mid_ref."""
    Q, I, days, kc = [], [], [], []
    for r in recs:
        if "child_q" not in r:
            continue
        ref = r["mid_ref"]
        if not np.isfinite(ref) or ref <= 0:
            continue
        q = np.cumsum(r["child_q"])
        imp = r["sign"] * (r["mid_child"] - ref) / ref
        ok = np.isfinite(imp) & (q > 0)
        Q += list(q[ok]); I += list(imp[ok]); days += [r["day"]] * int(ok.sum()); kc += list(np.arange(1, len(q) + 1)[ok])
    return {"Q": np.asarray(Q, float), "I": np.asarray(I, float), "days": np.asarray(days), "kc": np.asarray(kc, int)}


def delta_binned(x: np.ndarray, y: np.ndarray, nbins: int = NBINS):
    """Signed binned estimator: equal-count bins of x = log(Q/V), signed mean y, OLS of log(mean y) through positive bins."""
    order = np.argsort(x, kind="stable")
    xc, ym = [], []
    for chunk in np.array_split(order, nbins):
        if chunk.size:
            xc.append(x[chunk].mean())
            ym.append(y[chunk].mean())
    xc, ym = np.array(xc), np.array(ym)
    pos = ym > 0
    if pos.sum() < MIN_BINS or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan, np.nan, xc, ym
    slope, icpt = np.polyfit(xc[pos], np.log(ym[pos]), 1)
    return float(slope), float(icpt), xc, ym


def _daysum_delta(SY, SX, N, pick):
    sy, sx, n = SY[pick].sum(0), SX[pick].sum(0), N[pick].sum(0)
    ok = n > 0
    ym = np.where(ok, sy / np.maximum(n, 1), np.nan)
    xc = np.where(ok, sx / np.maximum(n, 1), np.nan)
    pos = ok & (ym > 0)
    if pos.sum() < MIN_BINS or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan
    return float(np.polyfit(xc[pos], np.log(ym[pos]), 1)[0])


def day_bootstrap(x, y, days, rng) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, NBINS + 1))
    edges[-1] += 1e-9
    di = {d: i for i, d in enumerate(np.unique(days))}
    nd = len(di)
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, NBINS - 1)
    didx = np.array([di[d] for d in days])
    SY, SX, N = np.zeros((nd, NBINS)), np.zeros((nd, NBINS)), np.zeros((nd, NBINS))
    for k in range(NBINS):
        m = b == k
        np.add.at(SY[:, k], didx[m], y[m])
        np.add.at(SX[:, k], didx[m], x[m])
        np.add.at(N[:, k], didx[m], 1)
    ds = np.array([_daysum_delta(SY, SX, N, rng.randint(0, nd, nd)) for _ in range(N_BOOT)])
    return ds[np.isfinite(ds)]


def l2_profile(q: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Direct least squares of y = a q^delta on the delta grid; a*(delta) = sum(y q^d) / sum(q^2d). Returns (delta, a, misfit)."""
    mis, amp = [], []
    for d in DELTA_GRID:
        qd = q ** d
        a = float(np.sum(y * qd) / np.sum(qd * qd))
        amp.append(a)
        mis.append(float(np.sum((y - a * qd) ** 2)))
    k = int(np.argmin(mis))
    return float(DELTA_GRID[k]), amp[k], np.asarray(mis)


def build_up(pts: dict, daily: dict, rng_seed: int = SEED) -> dict:
    """delta (binned) with a day-clustered 95% interval, the identifiability verdict, and the amplitude Y.

    daily: day -> {"V": shares executed, "range": log(high / low)} of the real stock on that day. Points are normalised
    as q = Q / V and y = I / sigma_P with sigma_P = range / kappa_P (Parkinson). The paper quotes Y at the data edge,
    Y = a q_ref^(delta - 1/2) / kappa_P with the fitted law I / sigma_P = a q^delta; we take q_ref as the mean q of the
    top bin and a from the binned fit (paper, section 2.1).
    """
    if len(pts["Q"]) == 0:
        return {"identified": False, "n_pts": 0}
    V = np.array([daily[d]["V"] for d in pts["days"]], float)
    sig = np.array([daily[d]["range"] for d in pts["days"]], float) / KAPPA_P
    q, y = pts["Q"] / V, pts["I"] / sig
    ok = np.isfinite(q) & np.isfinite(y) & (q > 0)
    q, y, days = q[ok], y[ok], pts["days"][ok]
    if len(q) < NBINS:  # e.g. a market whose daily volatility is zero: impact cannot be put in volatility units
        return {"identified": False, "n_pts": int(len(q))}
    x = np.log(q)
    d_bin, icpt, xc, ym = delta_binned(x, y)
    ds = day_bootstrap(x, y, days, np.random.RandomState(rng_seed))
    d_l2, a_l2, mis = l2_profile(q, y)
    on_bound = d_l2 <= DELTA_GRID[0] + 1e-9 or d_l2 >= DELTA_GRID[-1] - 1e-9
    q_ref = float(np.exp(xc[-1])) if len(xc) else np.nan
    Y_l2 = a_l2 * q_ref ** (d_l2 - 0.5) / KAPPA_P
    Y_bin = np.exp(icpt) * q_ref ** (d_bin - 0.5) / KAPPA_P if np.isfinite(d_bin) else np.nan
    identified = (not on_bound) and abs(Y_l2) >= Y_FLOOR and np.isfinite(d_bin)
    lo, hi = (np.percentile(ds, [2.5, 97.5]) if len(ds) else (np.nan, np.nan))
    return {"identified": bool(identified), "delta": d_bin, "delta_lo": float(lo), "delta_hi": float(hi), "delta_l2": d_l2,
            "Y": float(Y_bin), "Y_l2": float(Y_l2), "q_ref": q_ref, "n_pts": int(len(q)), "n_days": int(len(np.unique(days)))}


# ------------------------------------------------------------------ phases 2 and 3: the k-clock
def k_clock(recs, n_ins: int, n_cool: int) -> tuple[np.ndarray, list]:
    """Signed impact (bps) at the child rows k = 1..n_ins and at the ends of n_cool cooling windows (k = 0 is 0).

    The cooling-window length is the smallest gap between consecutive children, as in the reference code; our
    message-clock injection makes the gaps nearly equal. Rollouts whose |I| exceeds COLLAPSE_BPS are dropped."""
    K, days = [], []
    for r in recs:
        if "kmid" not in r or len(r["kmid"]) != n_ins + n_cool:
            continue
        ref = r["mid_ref"]
        if not np.isfinite(ref) or ref <= 0:
            continue
        v = np.r_[0.0, r["sign"] * (r["kmid"] - ref) / ref * 1e4]
        if np.nanmax(np.abs(v)) > COLLAPSE_BPS:
            continue
        K.append(v)
        days.append(r["day"])
    return (np.vstack(K) if K else np.zeros((0, n_ins + n_cool + 1))), days


def peak(K: np.ndarray, n_ins: int) -> dict:
    """Mean impact at the end of execution (k = n_ins), its s.e., and the detection gate (|mean| >= 3 s.e.)."""
    col = K[:, n_ins] if len(K) else np.array([])
    col = col[np.isfinite(col)]
    if len(col) < 2:
        return {"peak_bps": np.nan, "peak_se": np.nan, "detected": False, "n_rollouts": len(col)}
    m, se = float(col.mean()), float(col.std(ddof=1) / np.sqrt(len(col)))
    return {"peak_bps": m, "peak_se": se, "detected": bool(abs(m) >= DETECTION_SE * se), "n_rollouts": len(col)}


def relaxation(K: np.ndarray, n_ins: int = 10) -> dict:
    """Master curve <I(v)> / <I(1)> on v = k / n_ins; e/p is its last finite value; slope on the cooling segment."""
    if len(K) == 0:
        return {"ep": np.nan, "slope": np.nan, "sig": False}
    cnt = np.isfinite(K).sum(0)
    mean = np.nanmean(K, axis=0)
    mean[cnt < max(MIN_COUNT, MIN_FRAC * len(K))] = np.nan
    p = K[:, n_ins]
    p = p[np.isfinite(p)]
    pk, pse = float(np.mean(p)), float(np.std(p, ddof=1) / np.sqrt(len(p))) if len(p) > 1 else np.nan
    sig = bool(pk > 0 and pk >= DETECTION_SE * pse)
    master = mean / pk if pk != 0 else np.full_like(mean, np.nan)
    v = np.arange(len(mean)) / n_ins
    seg = (v - 1.0 >= Z0) & np.isfinite(master)
    ep = float(master[np.flatnonzero(np.isfinite(master))[-1]]) if np.isfinite(master).any() else np.nan
    slope = float(np.polyfit(v[seg] - 1.0, master[seg], 1)[0]) if seg.sum() >= 10 else np.nan
    return {"ep": ep, "slope_per_duration": slope, "peak_bps": pk, "peak_se": pse, "sig": sig, "n_rollouts": len(K)}


def attribution(visible, blind) -> dict:
    """v/b: mean visible final displacement over the mean blind mechanical part (both in ticks, buy and sell pooled)."""
    def finals(rs):
        return np.asarray([r["sign"] * (r["mid_last"] - r["mid_row0"]) / TICK for r in rs
                           if np.isfinite(r["mid_last"]) and np.isfinite(r["mid_row0"])])

    def mechs(rs):
        return np.asarray([r["sign"] * np.nansum(r["mid_child"] - r["mid_pre"]) / TICK for r in rs if "mid_child" in r])

    fv, mb = finals(visible), mechs(blind)
    if len(fv) < 2 or len(mb) == 0:
        return {"vb": np.nan}
    vis, se, mech = float(fv.mean()), float(fv.std(ddof=1) / np.sqrt(len(fv))), float(mb.mean())
    ok = abs(vis) >= DETECTION_SE * se and mech != 0
    return {"vb": vis / mech if ok else np.nan, "visible_final_ticks": vis, "blind_mech_ticks": mech}
