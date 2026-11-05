"""
Closed-form mean-field-game liquidation schedule.

Setting (sell convention, dq = -nu dt, nu >= 0 is the selling rate):

    execution price   S_t - a * nu_t                    (temporary impact)
    mid price         dS = (A_t - lam * nu_bar_t) dt + sigma dW   (permanent impact
                                                         of the population mean rate)
    agent minimises   E[ int a nu^2 + phi q^2 - q (A - lam nu_bar) dt + psi q_T^2 ]

The Euler-Lagrange condition for a single agent taking nu_bar as given is

    2a q'' = 2 phi q - A + lam nu_bar,     2a q'(T) + 2 psi q(T) = 0.

For a homogeneous population nu_bar = nu = -q', with A = 0 and full liquidation
(psi -> infinity, q(T) = 0), the equilibrium inventory solves

    q'' + (lam / 2a) q' - (phi / a) q = 0,   q(0) = 1,  q(T) = 0,

so q(t) = c1 exp(r1 t) + c2 exp(r2 t) with
    r = ( -lam/(2a) ± sqrt( (lam/(2a))^2 + 4 phi / a ) ) / 2.

phi = lam = 0 gives TWAP. phi > 0 (risk aversion) and lam > 0 (competition for
the same liquidity) both front-load.

Everything is in normalised units: time in [0, 1], inventory as a fraction of
the parent order, costs in bps. `a` is the temporary-impact cost at the TWAP
rate, `lam` the permanent impact of the full parent order, and `phi` the
running-inventory penalty.
"""

from __future__ import annotations

import numpy as np


def mfg_inventory(a: float, lam: float, phi: float, tau: np.ndarray) -> np.ndarray:
    if a <= 0:
        raise ValueError("a must be positive")
    tau = np.asarray(tau, dtype=np.float64)
    g = lam / (2.0 * a)
    disc = np.sqrt(g * g + 4.0 * phi / a)
    r1, r2 = 0.5 * (-g + disc), 0.5 * (-g - disc)
    if abs(r1 - r2) < 1e-10:
        return 1.0 - tau
    # q = c1 e^{r1 t} + c2 e^{r2 t}, q(0)=1, q(1)=0; written to avoid overflow in e^{r1}
    num = np.exp(r2 * tau) - np.exp(r1 * tau + (r2 - r1))
    return num / (1.0 - np.exp(r2 - r1))


def mfg_weights(a: float, lam: float, phi: float, n: int) -> np.ndarray:
    """Fraction of the parent order sold in each of n equal slices."""
    q = mfg_inventory(a, lam, phi, np.linspace(0.0, 1.0, n + 1))
    w = -np.diff(q)
    w = np.clip(w, 0.0, None)
    return w / w.sum()
