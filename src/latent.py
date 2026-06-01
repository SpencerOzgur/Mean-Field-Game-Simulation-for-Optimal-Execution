from dataclasses import dataclass
import numpy as np
import random


@dataclass(frozen=True)
class LatentParams:
    """
    parameters for latent markov chain
    """

    T: float
    N: int
    lambda01: float
    lambda10: float
    theta0: int


def build_transition_matrix(dt: float, params: LatentParams) -> np.ndarray:
    """
    :param dt: change in time
    :param params: Latent parameters
    :return: transition matrix 2x2
    """

    if dt < 0:
        raise ValueError("dt must be non-negative")

    if params.lambda01 < 0 or params.lambda10 < 0:
        raise ValueError("lambdas must be non-negative")

    s = params.lambda01 + params.lambda10

    if not s:
        return np.eye(2)

    e = np.exp(-(params.lambda10 + params.lambda01) * dt)

    p00 = params.lambda10 / s + params.lambda01 / s * e
    p01 = params.lambda01 / s * (1 - e)
    p10 = params.lambda10 / s * (1 - e)
    p11 = params.lambda01 / s + params.lambda10 / s * e

    return np.array([[p00, p01], [p10, p11]])


def simulate_latent_path(params: LatentParams) -> np.ndarray:
    """
    :param params: Latent parameters
    :return: simulated latent path theta
    """
    if params.theta0 != 0 and params.theta0 != 1:
        raise ValueError("theta0 must be 0 or 1")

    theta = np.empty(params.N + 1, dtype=np.int8)
    theta[0] = params.theta0

    dt = params.T / params.N
    P = build_transition_matrix(dt, params)

    for i in range(params.N):
        curr = theta[i]
        u = random.uniform(0, 1)
        if u < P[curr][curr]:
            theta[i + 1] = curr
        else:
            theta[i + 1] = 1 - curr
    return theta


def simlate_N_latent_paths(params: LatentParams, N_paths: int) -> np.ndarray:
    """
    :param params: Latent parameters
    :param N_paths: Number of paths to simulate
    :return: list of simulated latent paths
    """
    if params.N <= 0:
        raise ValueError("N must be positive")
    if N_paths <= 0:
        raise ValueError("N_paths must be positive")

    simulated_paths = np.empty((N_paths, params.N + 1), dtype=np.int8)
    for i in range(N_paths):
        simulated_paths[i] = simulate_latent_path(params)
    return simulated_paths
