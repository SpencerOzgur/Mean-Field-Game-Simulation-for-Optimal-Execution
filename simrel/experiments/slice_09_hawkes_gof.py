"""
Goodness of fit of the fitted Hawkes models, on fit days and on held-out days.

Two standard checks for point-process models of order flow:

1. Time-rescaling residuals. If the model's intensity lambda_i(t) is right, the compensator increments between
   consecutive type-i events, tau = integral of lambda_i over the gap, are independent Exp(1). We report the
   Kolmogorov-Smirnov distance of the residuals from Exp(1) per event type, and Q-Q quantiles.
2. Log-likelihood per event, against a Poisson model with the same average rates (estimated on the fit days).
   Held-out gains show the excitation structure generalises rather than overfits.

The intensity is exactly the one the simulator uses, with its state factors (cancels scale with depth, market orders
with 1/spread) taken from the book state logged before each event, and market-order marks weighted as in the fit.

    python simrel/experiments/slice_09_hawkes_gof.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simrel.adapters.hawkes import CB, CS, MB, MS  # noqa: E402
from simrel.configs import slices as C  # noqa: E402
from simrel.data import lobster as L  # noqa: E402
from simrel.experiments import slice_01_run as R  # noqa: E402

TYPES = ["limit buy", "limit sell", "market buy", "market sell", "cancel bid", "cancel ask"]


def model_parts(p):
    betas = np.atleast_1d(np.asarray(p.betas if p.betas is not None else [p.beta], dtype=float))
    alphas = np.asarray(p.alpha_stack) if p.alpha_stack is not None else np.asarray(p.alpha)[None]
    return betas, alphas


def residuals(o, p) -> tuple[list, float, float, int]:
    """Time-rescaling residuals per type for one window, plus the window's log-likelihood and compensator total."""
    ev = o.events
    t, k, sz = ev[:, 0], ev[:, 1].astype(int), ev[:, 2]
    spr = np.nan_to_num(ev[:, 5], nan=5.0)
    w = np.where((k == MB) | (k == MS), np.minimum((sz / p.mean_market_size()) ** p.weight_exponent, p.weight_cap), 1.0)
    # state factor of each type during the gap that ends at event n (the book does not change inside a gap)
    g = np.ones((len(t), 6))
    g[:, MB] = g[:, MS] = 1.0 / np.maximum(spr, 1.0)
    g[:, CB] = ev[:, 6] / p.ref_depth
    g[:, CS] = ev[:, 7] / p.ref_depth
    betas, alphas = model_parts(p)
    M = len(betas)
    a = np.zeros((M, 6))  # kernel mass by component and source, just after the previous event
    last = 0.0
    comp = np.zeros(6)  # compensator accumulated since the last event of each type
    out = [[] for _ in range(6)]
    ll = 0.0
    total = 0.0
    for n in range(len(t)):
        dt = t[n] - last
        decay = np.exp(-betas * dt)[:, None]
        before = a * decay
        # integral over the gap: baseline * dt + sum over components of alpha_m @ (mass lost to decay) / beta_m
        exc_int = np.einsum("mij,mj->i", alphas, (a - before) / betas[:, None]) if dt > 0 else np.zeros(6)
        inc = p.mu * g[n] * dt + exc_int
        comp += inc
        total += inc.sum()
        lam_n = p.mu[k[n]] * g[n, k[n]] + np.einsum("mj,mj->", alphas[:, k[n], :], before)
        ll += np.log(max(lam_n, 1e-300))
        out[k[n]].append(comp[k[n]])
        comp[k[n]] = 0.0
        a = before
        a[:, k[n]] += betas * w[n]
        last = t[n]
    return out, ll, total, len(t)


def poisson_ll(o, rates) -> float:
    k = o.events[:, 1].astype(int)
    return float(np.log(rates[k]).sum() - rates.sum() * o.duration)


def ks_exp1(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) < 20:
        return np.nan
    cdf = 1.0 - np.exp(-x)
    n = len(x)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n)))


def main() -> None:
    rows, qq = [], []
    qs = np.array([0.5, 0.75, 0.9, 0.95, 0.99])
    for name in R.usable_slices():
        fits = {"single kernel": R.load(R.RAW / f"hawkes_{name}.pkl")["params"], "multi-timescale": R.load(R.RAW / f"hawkes_m_{name}.pkl")["params"]}
        fit_ws = R.real_windows(name, "fit")
        pick = sorted(np.random.default_rng(0).permutation(len(fit_ws))[: C.HAWKES_MAX_FIT_WINDOWS])
        sets = {"fit days (in-sample)": [fit_ws[i] for i in pick], "held-out days": R.real_windows(name, "test")}
        obs = {s: [L.to_observed(w) for w in ws] for s, ws in sets.items()}
        n_fit = sum(len(o.events) for o in obs["fit days (in-sample)"])
        T_fit = sum(o.duration for o in obs["fit days (in-sample)"])
        rates = np.bincount(np.concatenate([o.events[:, 1].astype(int) for o in obs["fit days (in-sample)"]]), minlength=6) / T_fit
        for model, p in fits.items():
            for s, os_ in obs.items():
                res = [[] for _ in range(6)]
                ll = ll_p = 0.0
                n_ev = 0
                for o in os_:
                    r, l_, _, n = residuals(o, p)
                    for i in range(6):
                        res[i].extend(r[i])
                    ll += l_
                    ll_p += poisson_ll(o, rates)
                    n_ev += n
                pooled = np.concatenate([np.asarray(x) for x in res])
                row = {"slice": name, "model": model, "data": s, "events": n_ev, "loglik_per_event": ll / n_ev,
                       "poisson_loglik_per_event": ll_p / n_ev, "gain_per_event": (ll - ll_p) / n_ev,
                       "ks_pooled": ks_exp1(pooled), "mean_residual": float(np.mean(pooled))}
                for i in range(6):
                    row[f"ks {TYPES[i]}"] = ks_exp1(res[i])
                rows.append(row)
                qq.append({"slice": name, "model": model, "data": s, **{f"q{q:g}": float(np.quantile(pooled, q)) for q in qs}})
            print(f"{name} {model}: done", flush=True)
    t = pd.DataFrame(rows)
    out = R.OUT
    t.to_csv(out / "hawkes_gof.csv", index=False)
    pd.DataFrame(qq).to_csv(out / "hawkes_gof_qq.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["loglik_per_event", "poisson_loglik_per_event", "gain_per_event", "ks_pooled", "mean_residual"] + [f"ks {x}" for x in TYPES]
    print(t.groupby(["model", "data"])[cols].mean().round(3).T.to_string())
    print("Exp(1) quantiles for reference:", {f"q{q:g}": round(float(-np.log(1 - q)), 3) for q in qs})
    print(pd.DataFrame(qq).groupby(["model", "data"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
