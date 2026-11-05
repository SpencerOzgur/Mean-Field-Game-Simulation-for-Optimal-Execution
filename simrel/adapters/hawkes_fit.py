"""
Fit the Hawkes simulator to observed order flow.

Inputs are event logs and quote grids (simrel.reference_worlds.data.Observed)
from any market. The fit has three parts:

  1. Excitation. Baselines mu and the branching matrix alpha of a six-type
     exponential-kernel Hawkes process are estimated by EM, for each decay
     beta on a small grid; the beta with the highest log-likelihood is kept.
     Market orders enter with the mark weight the simulator uses.
  2. Marks. Size and price-level distributions (geometric) by maximum
     likelihood; the probability of improving the quote from the share of
     limit orders placed inside the spread.
  3. State dependence. The simulator's baselines are state-dependent (cancels
     scale with depth, market orders with 1/spread). The EM uses the same
     factors, evaluated on the book state logged just before each event.
"""

from __future__ import annotations

import numpy as np

from .hawkes import CB, CS, LB, LS, MB, MS, HawkesParams


def _em(seqs: list, beta: float, n_iter: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """EM for mu (6,) and alpha (6, 6) with a common decay.

    seqs: (times, types, weights, g, G) per episode, where g[n] is the state factor of event n's own
    baseline and G[i] is the time integral of type i's state factor over the episode.
    """
    D = 6
    feats, types_all, g_all = [], [], []
    G_total = np.zeros(D)
    exposure = np.zeros(D)  # integrated kernel mass of each source type
    for t, k, w, g, G, T in seqs:
        A = np.zeros((len(t), D))
        a = np.zeros(D)
        last = 0.0
        for n in range(len(t)):
            a *= np.exp(-beta * (t[n] - last))
            A[n] = a
            a[k[n]] += beta * w[n]
            last = t[n]
        feats.append(A)
        types_all.append(k)
        g_all.append(g)
        G_total += G
        np.add.at(exposure, k, w * (1.0 - np.exp(-beta * (T - t))))
    A = np.vstack(feats)
    k = np.concatenate(types_all)
    g = np.concatenate(g_all)
    counts = np.bincount(k, minlength=D).astype(float)
    mu = np.maximum(counts / np.maximum(G_total, 1e-9) * 0.5, 1e-4)
    alpha = np.full((D, D), 0.05)
    ll = -np.inf
    for _ in range(n_iter):
        base = mu[k] * g
        lam = np.maximum(base + (alpha[k] * A).sum(axis=1), 1e-12)
        resp = alpha[k] * A / lam[:, None]
        new_mu = np.bincount(k, weights=base / lam, minlength=D) / np.maximum(G_total, 1e-9)
        new_alpha = np.zeros((D, D))
        for i in range(D):
            m = k == i
            if m.any():
                new_alpha[i] = resp[m].sum(axis=0) / np.maximum(exposure, 1e-9)
        mu, alpha = np.maximum(new_mu, 1e-6), new_alpha
        ll = float(np.log(lam).sum() - (mu * G_total).sum() - (alpha * exposure[None, :]).sum())
    return mu, alpha, ll


def _features(seqs: list, betas) -> list:
    """Per episode, the decayed kernel mass of each (component, source type) just before each event: n x (M * 6)."""
    betas = np.atleast_1d(np.asarray(betas, dtype=float))
    out = []
    for t, k, w, g, G, T in seqs:
        A = np.zeros((len(t), len(betas), 6))
        a, last = np.zeros((len(betas), 6)), 0.0
        for n in range(len(t)):
            a *= np.exp(-betas * (t[n] - last))[:, None]
            A[n] = a
            a[:, k[n]] += betas * w[n]
            last = t[n]
        out.append(A.reshape(len(t), -1))
    return out


def _em_multi(seqs: list, betas, n_iter: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """EM for mu (6,) and a stacked excitation matrix (6, M * 6) for a sum of M exponential kernels with fixed decays.

    Column m * 6 + j holds the excitation of every type by a type-j event through kernel m. With M = 1 this is _em."""
    betas = np.atleast_1d(np.asarray(betas, dtype=float))
    M = len(betas)
    A = np.vstack(_features(seqs, betas))
    k = np.concatenate([s[1] for s in seqs])
    g = np.concatenate([s[3] for s in seqs])
    G_total = sum(s[4] for s in seqs)
    exposure = np.zeros((M, 6))
    for t, kk, w, _, _, T in seqs:
        for m, b in enumerate(betas):
            np.add.at(exposure[m], kk, w * (1.0 - np.exp(-b * (T - t))))
    exposure = exposure.reshape(-1)
    counts = np.bincount(k, minlength=6).astype(float)
    mu = np.maximum(counts / np.maximum(G_total, 1e-9) * 0.5, 1e-4)
    alpha = np.full((6, 6 * M), 0.05 / M)
    ll = -np.inf
    for _ in range(n_iter):
        base = mu[k] * g
        lam = np.maximum(base + (alpha[k] * A).sum(axis=1), 1e-12)
        resp = alpha[k] * A / lam[:, None]
        new_mu = np.bincount(k, weights=base / lam, minlength=6) / np.maximum(G_total, 1e-9)
        new_alpha = np.zeros_like(alpha)
        for i in range(6):
            sel = k == i
            if sel.any():
                new_alpha[i] = resp[sel].sum(axis=0) / np.maximum(exposure, 1e-9)
        mu, alpha = np.maximum(new_mu, 1e-6), new_alpha
        ll = float(np.log(lam).sum() - (mu * G_total).sum() - (alpha * exposure[None, :]).sum())
    return mu, alpha, ll


def _quantiles(x: np.ndarray, n: int = 2000) -> np.ndarray:
    return np.quantile(x, (np.arange(n) + 0.5) / n, method="inverted_cdf").astype(np.int64)


def fit(observed: list, betas=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0), lot: int = 100, weight_exponent: float = 1.0, weight_cap: float = 10.0,
        empirical: bool = False, n_iter: int = 60, max_branching: float = 0.9, queue: bool = False,
        multi_betas: tuple | None = None) -> tuple[HawkesParams, dict]:
    """empirical=True also stores the observed size and price-level distributions and switches cancels to whole orders."""
    ev = np.vstack([o.events for o in observed])
    types = ev[:, 1].astype(int)
    sizes = ev[:, 2]
    levels = ev[:, 3]
    is_mkt = (types == MB) | (types == MS)
    is_lim = (types == LB) | (types == LS)
    mean_mkt = float(sizes[is_mkt].mean()) if is_mkt.any() else float(lot)

    spread_grid = np.concatenate([o.ask - o.bid for o in observed])
    ref_depth = float(0.5 * (np.concatenate([o.bid_depth10 for o in observed]).mean() + np.concatenate([o.ask_depth10 for o in observed]).mean()))

    # queue version: cancels at the best scale with the best queue, cancels behind it with the depth behind it
    is_can0 = (types == CB) | (types == CS)
    queue_w = float(np.mean(levels[is_can0] <= 0)) if queue else None
    own_q1 = np.where(types == CB, ev[:, 8], ev[:, 9]) if queue else None
    own_dep = np.where(types == CB, ev[:, 6], ev[:, 7])
    ref_q1 = float(np.mean(np.concatenate([ev[:, 8], ev[:, 9]]))) if queue else 1.0
    ref_rest = float(np.mean(np.concatenate([ev[:, 6] - ev[:, 8], ev[:, 7] - ev[:, 9]]))) if queue else 1.0

    def cancel_g(q1, dep):
        return queue_w * q1 / ref_q1 + (1.0 - queue_w) * np.maximum(dep - q1, 0.0) / ref_rest

    # time spent with the spread at two ticks or more, and the limit orders that improved the quote in that time
    wide_time = inside_n = 0.0
    for o in observed:
        t_, spr_ = o.events[:, 0], np.nan_to_num(o.events[:, 5], nan=5.0)
        wide_time += float((np.diff(t_, prepend=0.0) * (spr_ >= 2)).sum())
        k_ = o.events[:, 1].astype(int)
        inside_n += float((((k_ == LB) | (k_ == LS)) & (o.events[:, 3] < 0)).sum())
    close_rate = float(min(inside_n / max(wide_time, 1e-9) / 2.0, 400.0))  # per side

    # the simulator's baselines are state-dependent: cancels scale with depth / ref_depth, market orders with 1 / spread
    seqs = []
    for o in observed:
        t, k, sz = o.events[:, 0], o.events[:, 1].astype(int), o.events[:, 2]
        spr = np.nan_to_num(o.events[:, 5], nan=5.0)
        w = np.where((k == MB) | (k == MS), np.minimum((sz / mean_mkt) ** weight_exponent, weight_cap), 1.0)
        g = np.ones(len(t))
        mk = (k == MB) | (k == MS)
        g[mk] = 1.0 / np.maximum(spr[mk], 1.0)
        g[k == CB] = o.events[k == CB, 6] / ref_depth
        g[k == CS] = o.events[k == CS, 7] / ref_depth
        dt = np.diff(np.append(o.grid_t, o.duration))
        inv_s = 1.0 / np.maximum(o.ask - o.bid, 1.0)
        G = np.array([o.duration, o.duration, (inv_s * dt).sum(), (inv_s * dt).sum(),
                      (o.bid_depth10 / ref_depth * dt).sum(), (o.ask_depth10 / ref_depth * dt).sum()])
        if queue:  # the same factors as the simulator, integrated over the state between consecutive events
            de = np.diff(t, prepend=0.0)
            g[k == CB] = cancel_g(o.events[k == CB, 8], o.events[k == CB, 6])
            g[k == CS] = cancel_g(o.events[k == CS, 9], o.events[k == CS, 7])
            G[4] = float((cancel_g(o.events[:, 8], o.events[:, 6]) * de).sum())
            G[5] = float((cancel_g(o.events[:, 9], o.events[:, 7]) * de).sum())
        seqs.append((t, k, w, np.maximum(g, 1e-6), G, o.duration))
    best = None
    scan = {}
    for b in betas:
        mu, alpha, ll = _em(seqs, b, n_iter)
        scan[b] = ll
        if best is None or ll > best[3]:
            best = (b, mu, alpha, ll)
    beta, mu, alpha, ll = best
    alpha_stack = None
    if multi_betas is not None:  # a sum of kernels with fixed decays replaces the single best decay
        mu, alpha_st, ll = _em_multi(seqs, multi_betas, n_iter)
        M = len(multi_betas)
        alpha_stack = alpha_st.reshape(6, M, 6).transpose(1, 0, 2)  # (M, target, source)
        alpha = alpha_stack.sum(axis=0)
        scan = {f"multi {tuple(multi_betas)}": ll}

    lim_levels = levels[is_lim]
    behind = lim_levels[lim_levels >= 0]
    lim_wide = is_lim & (np.nan_to_num(ev[:, 5], nan=5.0) >= 2)  # limit orders that could have improved the quote
    inside_prob = float(np.mean(levels[lim_wide] < 0)) if lim_wide.any() else 0.0
    spread = spread_grid
    params = HawkesParams(
        mu=mu, alpha=alpha, beta=float(beta), lot=lot, ref_depth=ref_depth,
        limit_size_p=float(np.clip(lot / sizes[~is_mkt].mean(), 0.05, 0.95)),
        market_size_p=float(np.clip(lot / mean_mkt, 0.05, 0.95)),
        level_p=float(np.clip(1.0 / (1.0 + behind.mean()), 0.05, 0.95)) if len(behind) else 0.35,
        inside_prob=float(np.clip(inside_prob, 0.0, 0.95)),
        init_level_qty=max(int(ref_depth / 10) // lot * lot, lot),
        weight_exponent=weight_exponent, weight_cap=weight_cap,
    )
    eig0 = float(np.max(np.abs(np.linalg.eigvals(alpha))))
    if empirical and eig0 >= max_branching:
        # Real order flow fits as (near-)critical with one exponential kernel. Shrink the excitation to a stable level
        # and reset the baselines so that the stationary event rates still equal the observed ones:
        #   rate = mu * mean(g) + alpha @ (rate * mean weight)   =>   mu = (rate - alpha @ (rate * mean weight)) / mean(g)
        if alpha_stack is not None:
            alpha_stack = alpha_stack * (max_branching / eig0)
        alpha = alpha * (max_branching / eig0)
        T_total = sum(s[5] for s in seqs)
        rate = np.bincount(types, minlength=6) / T_total
        w_all = np.concatenate([s[2] for s in seqs])
        w_mean = np.array([w_all[types == j].mean() if (types == j).any() else 1.0 for j in range(6)])
        g_mean = sum(s[4] for s in seqs) / T_total
        mu = np.maximum(rate - alpha @ (rate * w_mean), 0.02 * rate) / np.maximum(g_mean, 1e-9)
        params.mu, params.alpha = mu, alpha
    if empirical and ev.shape[1] > 9:
        # size of each market order relative to the best queue on the side it hits, just before it
        hit_q1 = np.where(types == MS, ev[:, 8], ev[:, 9])
        r = sizes[is_mkt] / np.maximum(hit_q1[is_mkt], 1.0)
        params.market_ratio_q = np.quantile(r, (np.arange(2000) + 0.5) / 2000, method="inverted_cdf")
        ratio_info = {"market_ratio_median": float(np.median(r)), "market_sweep_share": float(np.mean(r > 1.0 + 1e-9))}
    if empirical:
        # intensities peak just after an event; cap each type at a high quantile of those peaks in the data
        peaks = []
        bs = np.asarray(multi_betas if multi_betas is not None else [beta], dtype=float)
        st = alpha_stack if alpha_stack is not None else alpha[None]
        flat = st.transpose(1, 0, 2).reshape(6, -1)
        for A, (t, k, w, g, G, T) in zip(_features(seqs, bs), seqs):
            after = A.copy()  # just after each event: add its own jump
            for m, b in enumerate(bs):
                after[np.arange(len(t)), m * 6 + k] += b * w
            peaks.append(after @ flat.T)
        params.lam_max = np.maximum(np.quantile(np.vstack(peaks), 0.99, axis=0) + 4.0 * mu, 3.0)
        is_can = (types == 4) | (types == 5)
        params.limit_size_q, params.market_size_q = _quantiles(sizes[is_lim]), _quantiles(sizes[is_mkt])
        params.level_q, params.cancel_level_q = _quantiles(behind), _quantiles(np.maximum(levels[is_can], 0))
        params.whole_order_cancels, params.mean_market = True, mean_mkt
        if queue:
            deep = levels[is_can][levels[is_can] >= 1]
            params.queue_w, params.ref_q1, params.ref_rest = queue_w, ref_q1, ref_rest
            params.deep_cancel_level_q = _quantiles(deep) if len(deep) else np.array([1])
    if alpha_stack is not None:
        params.betas, params.alpha_stack = np.asarray(multi_betas, dtype=float), alpha_stack
    eig = float(np.max(np.abs(np.linalg.eigvals(alpha))))
    info = {"beta": float(beta), "loglik": ll, "beta_scan": {str(k): v for k, v in scan.items()}, "branching_ratio": eig,
            "n_events": int(len(ev)), "events_per_s": float(len(ev) / sum(o.duration for o in observed)), "mean_spread_ticks": float(spread.mean())}
    if params.lam_max is not None:
        info["lam_max"] = [float(x) for x in params.lam_max]
    info.update(locals().get("ratio_info", {}))
    info["close_rate_per_side"], info["wide_spread_time_share"] = close_rate, wide_time / sum(o.duration for o in observed)
    info["queue_w"] = queue_w
    if eig0 > eig + 1e-9:
        info["alpha_rescaled_from"] = eig0
    if eig >= 0.98:  # keep the simulator stable if the fit lands on the boundary
        params.alpha = alpha * (0.95 / eig)
        info["alpha_rescaled_from"] = eig
    return params, info
