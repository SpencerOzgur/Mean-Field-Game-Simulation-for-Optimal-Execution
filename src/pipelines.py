import numpy as np
import latent
import simulate
import filtering
from params import SubPopParams

from params import SubPopParams


def make_default_subpops() -> list[SubPopParams]:
    """
    Default two-subpopulation configuration used across experiments.

    Returns
    -------
    subpops : list[SubPopParams]
        List of subpopulation parameter objects.
    """
    subpop1 = SubPopParams(
        name="SubPop1",
        weight=0.5,
        prior=0.8,
        Q0=1.0,
        kappa=0.5,
    )

    subpop2 = SubPopParams(
        name="SubPop2",
        weight=0.5,
        prior=0.2,
        Q0=1.0,
        kappa=2.0,
    )

    subpops = [subpop1, subpop2]

    # sanity check: weights sum to 1
    total_weight = sum(sp.weight for sp in subpops)
    if not abs(total_weight - 1.0) < 1e-8:
        raise ValueError(f"Subpopulation weights must sum to 1 (got {total_weight})")

    return subpops


def build_filtered_signals(
    subpops: list[SubPopParams],
    latent_params,
    sim_params,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Simulate the common latent environment and compute filtered signals
    for each subpopulation.

    Parameters
    ----------
    subpops : list[SubPopParams]
        Heterogeneous subpopulations with different priors.
    latent_params
        Parameters for the latent Markov chain.
    sim_params
        Parameters for price simulation.
    seed : int | None
        Optional NumPy random seed for reproducibility.

    Returns
    -------
    results : dict[str, np.ndarray]
        Dictionary containing:
        - latent_path : shape (N+1,)
        - F_t         : shape (N+1,)
        - pi_k        : shape (K, N+1)
        - A_hat_k     : shape (K, N+1)
    """
    if seed is not None:
        np.random.seed(seed)

    if len(subpops) == 0:
        raise ValueError("subpops must be non-empty")

    K = len(subpops)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=sim_params,
    )

    pi_k = np.empty((K, sim_params.N + 1), dtype=np.float64)
    A_hat_k = np.empty((K, sim_params.N + 1), dtype=np.float64)

    for i, sp in enumerate(subpops):
        pi_k[i] = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=sim_params,
            prior=[1.0 - sp.prior, sp.prior],
        )

        A_hat_k[i] = pi_k[i] * sim_params.A1 + (1.0 - pi_k[i]) * sim_params.A0

    return {
        "latent_path": latent_path,
        "F_t": F_t,
        "pi_k": pi_k,
        "A_hat_k": A_hat_k,
    }
