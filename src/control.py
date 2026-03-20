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