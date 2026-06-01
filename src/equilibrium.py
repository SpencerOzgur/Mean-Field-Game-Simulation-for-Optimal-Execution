from control import EquilibriumControlParams
import numpy as np

from control import EquilibriumControlParams
import numpy as np


def solve_mean_field_fixed_point(
    A_hat_list: list[np.ndarray],
    weights: np.ndarray,
    param_list: list[EquilibriumControlParams],
    max_iter: int = 100,
    tol: float = 1e-6,
    relax_bar: float = 0.2,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Solve for mean-field equilibrium by Picard iteration.

    Returns
    -------
    nu_list : list of np.ndarray
        Optimal control for each subpopulation
    q_list : list of np.ndarray
        Inventory path for each subpopulation
    nu_bar : np.ndarray
        Mean-field trading rate
    """
    if len(A_hat_list) == 0:
        raise ValueError("A_hat_list must be non-empty")
    if len(A_hat_list) != len(param_list):
        raise ValueError("A_hat_list and param_list must have same length")

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != len(A_hat_list):
        raise ValueError("weights must be 1-D with one entry per subpopulation")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("weights must sum to 1.0")
    if not (0.0 < relax_bar <= 1.0):
        raise ValueError("relax_bar must lie in (0, 1]")

    num_pops = len(A_hat_list)
    N = param_list[0].N

    for params in param_list:
        if params.N != N:
            raise ValueError("All param_list entries must share the same N")

    nu_bar = np.zeros(N, dtype=np.float64)

    for _ in range(max_iter):
        nu_list = []
        q_list = []

        for k in range(num_pops):
            nu_k, q_k = equilibrium_control_fbsde(
                A_hat=A_hat_list[k],
                nu_bar=nu_bar,
                params=param_list[k],
            )
            nu_list.append(nu_k)
            q_list.append(q_k)

        nu_bar_new = np.zeros(N, dtype=np.float64)
        for w, nu_k in zip(weights, nu_list):
            nu_bar_new += w * nu_k

        err = np.max(np.abs(nu_bar_new - nu_bar))
        nu_bar = (1.0 - relax_bar) * nu_bar + relax_bar * nu_bar_new

        if err < tol:
            break

    return nu_list, q_list, nu_bar


def equilibrium_control_fbsde(
    A_hat: np.ndarray,
    nu_bar: np.ndarray,
    params: EquilibriumControlParams,
    max_iter: int = 50,
    tol: float = 1e-8,
    relax: float = 0.05,
    divergence_tol: float = 1e3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete forward-backward best response solver for one subpopulation.

    Approximates the paper's FBSDE system:
        dq_t = -nu_t dt
        -d(2a nu_t) = (A_hat_t + lam * nu_bar_t - 2 phi q_t) dt - dM_t
        2a nu_T = -2 psi q_T

    using a pathwise forward-backward fixed-point iteration.
    """
    if params.T <= 0:
        raise ValueError("T must be positive")
    if params.N <= 0:
        raise ValueError("N must be positive")
    if params.a <= 0:
        raise ValueError("a must be positive")
    if params.psi < 0:
        raise ValueError("psi must be nonnegative")
    if params.phi < 0:
        raise ValueError("phi must be nonnegative")
    if not (0.0 < relax <= 1.0):
        raise ValueError("relax must lie in (0, 1]")

    A_hat = np.asarray(A_hat, dtype=np.float64)
    nu_bar = np.asarray(nu_bar, dtype=np.float64)

    if A_hat.ndim != 1 or len(A_hat) != params.N:
        raise ValueError("A_hat must be 1-D of length N")
    if nu_bar.ndim != 1 or len(nu_bar) != params.N:
        raise ValueError("nu_bar must be 1-D of length N")
    if not np.all(np.isfinite(A_hat)):
        raise ValueError("A_hat must be finite")
    if not np.all(np.isfinite(nu_bar)):
        raise ValueError("nu_bar must be finite")

    dt = params.T / params.N
    N = params.N

    nu = np.zeros(N, dtype=np.float64)

    for _ in range(max_iter):
        # Forward pass
        q_new = np.empty(N + 1, dtype=np.float64)
        q_new[0] = params.Q0
        for i in range(N):
            q_new[i + 1] = q_new[i] - nu[i] * dt

        if not np.all(np.isfinite(q_new)):
            raise ValueError("q_new became non-finite")
        if np.max(np.abs(q_new)) > divergence_tol:
            raise ValueError("inner solver diverged: q_new too large")

        # Backward pass
        nu_new = np.empty(N, dtype=np.float64)

        terminal_nu = (params.psi / params.a) * q_new[N]
        nu_new[N - 1] = terminal_nu

        for i in range(N - 2, -1, -1):
            rhs = A_hat[i] + params.lam * nu_bar[i] - 2.0 * params.phi * q_new[i]
            nu_new[i] = nu_new[i + 1] + (dt / (2.0 * params.a)) * rhs

        if not np.all(np.isfinite(nu_new)):
            raise ValueError("nu_new became non-finite")
        if np.max(np.abs(nu_new)) > divergence_tol:
            raise ValueError("inner solver diverged: nu_new too large")

        # Relaxation
        nu_next = (1.0 - relax) * nu + relax * nu_new

        if not np.all(np.isfinite(nu_next)):
            raise ValueError("nu_next became non-finite")

        err = np.max(np.abs(nu_next - nu))
        nu = nu_next

        if err < tol:
            break

    # Final forward pass consistent with returned nu
    q_final = np.empty(N + 1, dtype=np.float64)
    q_final[0] = params.Q0
    for i in range(N):
        q_final[i + 1] = q_final[i] - nu[i] * dt

    return nu, q_final
