import numpy as np
import latent
import simulate
import filtering
from params import latent_params, simulation_params


def test_fundamental_filter_shape():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    pi = filtering.filter_fundamental_prob_state_1(
        F_t=F_t,
        latent_params=latent_params,
        sim_params=simulation_params,
    )

    assert isinstance(pi, np.ndarray)
    assert pi.shape == (simulation_params.N + 1,)


def test_fundamental_filter_bounds():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    pi = filtering.filter_fundamental_prob_state_1(
        F_t=F_t,
        latent_params=latent_params,
        sim_params=simulation_params,
    )

    assert np.all(pi >= 0.0)
    assert np.all(pi <= 1.0)


def test_fundamental_filter_respects_prior_at_time_zero():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )
    prior = [0.7, 0.3]

    pi = filtering.filter_fundamental_prob_state_1(
        F_t=F_t,
        latent_params=latent_params,
        sim_params=simulation_params,
        prior=prior,
    )

    # Assuming pi[t] is P(state=1 | info up to t), this should start near prior[1].
    assert np.isclose(pi[0], prior[1])
