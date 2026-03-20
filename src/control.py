import numpy as np
from dataclasses import dataclass

@dataclass
class ControlParams():
    T: float
    N: int
    Q0: float

def validate_controls(nu_hat: np.ndarray, params: ControlParams):
    """
    :param nu_hat: mean-field control
    :param params: control parameters
    :return: validated controls
    """
    if np.ndim(nu_hat) != 1:
        raise ValueError('nu_hat must be a 1-D array')
    if len(nu_hat) != params.N:
        raise ValueError('nu_hat must be size N')
    if not np.all(np.isfinite(nu_hat)):
        raise ValueError('nu_hat must be finite')

    return nu_hat

def zero_control(params: ControlParams):
    """
    :param params: control parameters
    :return: zero control
    """
    if params.T <= 0:
        raise ValueError('T must be positive')
    if params.N <= 0:
        raise ValueError('N must be positive')

    return np.zeros(params.N, dtype=np.float64)

def constant_liquidation_control(params: ControlParams):
    """
    :param params: control parameters
    :return: guarateed liquidation control
    """
    if params.T <= 0:
        raise ValueError('T must be positive')
    if params.N <= 0:
        raise ValueError('N must be positive')

    rate = params.Q0 / params.T
    return np.full(params.N, rate, dtype=np.float64)

def simulate_inventory(nu_hat: np.ndarray, params: ControlParams):
    """
    :param nu_hat: mean field control
    :param params: control parameters
    :return: inventory
    """
    if params.T <= 0:
        raise ValueError('T must be positive')
    if params.N <= 0:
        raise ValueError('N must be positive')

    dt = params.T / params.N
    sim_inv = np.empty(params.N + 1, dtype=np.float64)
    sim_inv[0] = params.Q0
    for i in range(params.N):
        sim_inv[i + 1] = sim_inv[i] - nu_hat[i] * dt
    return sim_inv

def alpha_inventory_control(A_hat: np.ndarray,
                            params: ControlParams,
                            kappa: float) -> np.ndarray:
    """
    Toy Control
    Baseline liquidation rule:
        nu_i = Q_i / (T - t_i) - kappa * alpha_hat_i

    Positive nu_i means selling.
    """
    if params.T <= 0:
        raise ValueError("T must be positive")
    if params.N <= 0:
        raise ValueError("N must be positive")

    A_hat = np.asarray(A_hat, dtype=np.float64)
    if A_hat.ndim != 1:
        raise ValueError("A_hat must be 1-D")
    if len(A_hat) != params.N:
        raise ValueError("A_hat must have length N")
    if not np.all(np.isfinite(A_hat)):
        raise ValueError("A_hat must be finite")

    dt = params.T / params.N
    nu_hat = np.empty(params.N, dtype=np.float64)

    Q = params.Q0
    for i in range(params.N):
        t_i = i * dt
        time_left = max(params.T - t_i, dt)
        base_rate = Q / time_left
        signal_adjustment = kappa * A_hat[i]

        nu = base_rate - signal_adjustment
        nu = max(nu, 0.0)
        nu = min(nu, Q / dt)
        nu_hat[i] = nu
        Q = Q - nu * dt
    return nu_hat