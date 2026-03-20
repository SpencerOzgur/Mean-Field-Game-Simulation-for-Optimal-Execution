import numpy as np
import latent
import simulate

def gaussian_likelihood(y: float,
                        state: int,
                        sim_params: simulate.SimulationParams,
                        impact: float = 0.0,) -> float:
    """
    :param y: observation
    :param state: current state
    :param sim_params: simulation parameters
    :return: likelihood of y
    """
    dt = sim_params.T / sim_params.N
    mu = ((sim_params.A0 + impact )* dt if state == 0 else (sim_params.A1 + impact) * dt)
    var = sim_params.sigma ** 2 * dt

    if var <= 0:
        raise ValueError('Variance must be positive')

    coeff = 1.0 / np.sqrt(2 * np.pi * var)
    exp = np.exp(-0.5 * ((y - mu) ** 2) / var)
    return coeff * exp


def filter_step(posterior_prev: np.ndarray,
                y: float,
                trans_mat: np.ndarray,
                sim_params: simulate.SimulationParams,
                impact: float = 0.0) -> np.ndarray:
    """
    :param posterior_prev: posterior from previous step
    :param y: observation
    :param trans_mat: latent transition matrix
    :param sim_params: simulation parameters
    :return: 2 x 1 posterier vector
    """
    prediction = posterior_prev @ trans_mat
    likelihood = np.array([
        gaussian_likelihood(y=y, impact=impact, state=0, sim_params=sim_params),
        gaussian_likelihood(y=y, impact=impact, state=1, sim_params=sim_params),
    ])

    unnormalized_likelihood = prediction * likelihood
    return unnormalized_likelihood / np.sum(unnormalized_likelihood)

def filter_fundamental_path(F_t: np.ndarray,
                            latent_params: latent.LatentParams,
                            sim_params: simulate.SimulationParams) -> np.ndarray:
    """
    :param F_t: Unimpacted price path
    :param latent_params: latent parameters
    :param sim_params: simulation parameters
    :return: (N + 1) x 2 posterior matrix
    """
    F_t = np.asarray(F_t, dtype=np.float64)
    if len(F_t) != sim_params.N + 1:
        raise ValueError('F_t must be of length N + 1')

    dt = sim_params.T / sim_params.N
    trans_mat = latent.build_transition_matrix(dt, latent_params)
    dFt = np.diff(F_t)
    posterior = np.zeros((sim_params.N + 1, 2))

    if not latent_params.theta0:
        posterior[0] = [1.0, 0.0]
    else:
        posterior[0] = [0.0, 1.0]

    for i in range(sim_params.N):
        posterior[i + 1] = filter_step(posterior[i], dFt[i], trans_mat, sim_params)

    return posterior

def filter_impacted_path(S_t: np.ndarray,
                         impact: np.ndarray,
                         latent_params: latent.LatentParams,
                         sim_params: simulate.SimulationParams) -> np.ndarray:
    S_t = np.asarray(S_t, dtype=np.float64)
    if len(S_t) != sim_params.N + 1:
        raise ValueError('F_t must be of length N + 1')
    if len(impact) != sim_params.N:
        raise ValueError("impact must be of length N")

    dt = sim_params.T / sim_params.N
    trans_mat = latent.build_transition_matrix(dt, latent_params)
    dSt = np.diff(S_t)
    posterior = np.zeros((sim_params.N + 1, 2))

    if not latent_params.theta0:
        posterior[0] = [1.0, 0.0]
    else:
        posterior[0] = [0.0, 1.0]
    for i in range(sim_params.N):
        posterior[i + 1] = filter_step(posterior[i], dSt[i], trans_mat, sim_params, impact=impact[i])

    return posterior

def filter_fundamental_prob_state_1(F_t: np.ndarray,
                        latent_params: latent.LatentParams,
                        sim_params: simulate.SimulationParams) -> np.ndarray:
    """
    :param F_t: Unimpacted price path
    :param latent_params: latent parameters
    :param sim_params: simulation parameters
    :return: state 1 posteriors
    """
    filter_path = filter_fundamental_path(F_t, latent_params, sim_params)
    return filter_path[:, 1]

def filter_impacted_prob_state_1(S_t: np.ndarray,
                        latent_params: latent.LatentParams,
                        sim_params: simulate.SimulationParams,
                        impact: np.ndarray) -> np.ndarray:
    """
    :param S_t: Impacted price path
    :param latent_params: latent parameters
    :param sim_params: simulation parameters
    :param impact: impact list
    :return: state 1 posteriors
    """
    filter_path = filter_impacted_path(S_t, impact, latent_params, sim_params)
    return filter_path[:, 1]