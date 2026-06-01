import numpy as np
import latent
import simulate
from params import latent_params, simulation_params


def test_latent_to_drift_shape():
    latent_path = latent.simulate_latent_path(params=latent_params)
    drift = simulate.latent_to_drift(latent_path, simulation_params)
    assert drift.shape == latent_path.shape


def test_latent_to_drift_values():
    latent_path = np.array([0, 1, 0, 1, 1], dtype=np.int8)
    drift = simulate.latent_to_drift(latent_path, simulation_params)
    expected = np.array(
        [
            simulation_params.A0,
            simulation_params.A1,
            simulation_params.A0,
            simulation_params.A1,
            simulation_params.A1,
        ]
    )
    assert np.allclose(drift, expected)


def test_simulate_fundamental_path_shape():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )
    assert isinstance(F_t, np.ndarray)
    assert F_t.shape == (simulation_params.N + 1,)


def test_simulate_fundamental_path_starts_at_S0():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )
    assert np.isclose(F_t[0], simulation_params.S0)


def test_simulate_impacted_price_shape():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )
    nu_hat = np.zeros(simulation_params.N)

    S_t = simulate.simulate_impacted_price(
        F_t=F_t,
        nu_hat=nu_hat,
        params=simulation_params,
    )

    assert isinstance(S_t, np.ndarray)
    assert S_t.shape == (simulation_params.N + 1,)


def test_simulate_impacted_price_equals_fundamental_when_no_trading():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )
    nu_hat = np.zeros(simulation_params.N)

    S_t = simulate.simulate_impacted_price(
        F_t=F_t,
        nu_hat=nu_hat,
        params=simulation_params,
    )

    assert np.allclose(S_t, F_t)


def test_simulate_impacted_price_matches_manual_construction():
    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    nu_hat = np.ones(simulation_params.N)
    dt = simulation_params.T / simulation_params.N

    S_t = simulate.simulate_impacted_price(
        F_t=F_t,
        nu_hat=nu_hat,
        params=simulation_params,
    )

    cumulative_impact = np.zeros(simulation_params.N + 1)
    cumulative_impact[1:] = np.cumsum(nu_hat) * dt

    expected = F_t - simulation_params.lambda_ * cumulative_impact

    assert np.allclose(S_t, expected)
