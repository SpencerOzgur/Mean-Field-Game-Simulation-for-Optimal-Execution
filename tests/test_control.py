import numpy as np
import equilibrium
from control import EquilibriumControlParams
from params import control_params, simulation_params


def test_equilibrium_control_fbsde_shape():
    N = control_params.N
    A_hat = np.zeros(N)
    nu_bar = np.zeros(N)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=control_params.Q0,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    nu_hat, q = equilibrium.equilibrium_control_fbsde(
        A_hat=A_hat,
        nu_bar=nu_bar,
        params=params,
    )

    assert isinstance(nu_hat, np.ndarray)
    assert isinstance(q, np.ndarray)
    assert nu_hat.shape == (N,)
    assert q.shape == (N + 1,)


def test_equilibrium_control_fbsde_outputs_are_finite():
    N = control_params.N
    A_hat = np.zeros(N)
    nu_bar = np.zeros(N)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=control_params.Q0,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    nu_hat, q = equilibrium.equilibrium_control_fbsde(
        A_hat=A_hat,
        nu_bar=nu_bar,
        params=params,
    )

    assert np.all(np.isfinite(nu_hat))
    assert np.all(np.isfinite(q))


def test_equilibrium_control_fbsde_inventory_recursion():
    N = control_params.N
    A_hat = np.zeros(N)
    nu_bar = np.zeros(N)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=control_params.Q0,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    nu_hat, q = equilibrium.equilibrium_control_fbsde(
        A_hat=A_hat,
        nu_bar=nu_bar,
        params=params,
    )

    dt = params.T / params.N
    for i in range(params.N):
        assert np.isclose(q[i + 1], q[i] - nu_hat[i] * dt)


def test_equilibrium_control_fbsde_initial_inventory_matches_Q0():
    N = control_params.N
    A_hat = np.zeros(N)
    nu_bar = np.zeros(N)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=2.5,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    _, q = equilibrium.equilibrium_control_fbsde(
        A_hat=A_hat,
        nu_bar=nu_bar,
        params=params,
    )

    assert np.isclose(q[0], params.Q0)


def test_equilibrium_control_fbsde_rejects_wrong_A_hat_length():
    N = control_params.N
    A_hat = np.zeros(N - 1)
    nu_bar = np.zeros(N)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=control_params.Q0,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    try:
        equilibrium.equilibrium_control_fbsde(
            A_hat=A_hat,
            nu_bar=nu_bar,
            params=params,
        )
        assert False, "Expected ValueError for wrong A_hat length"
    except ValueError:
        assert True


def test_equilibrium_control_fbsde_rejects_wrong_nu_bar_length():
    N = control_params.N
    A_hat = np.zeros(N)
    nu_bar = np.zeros(N - 1)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=control_params.Q0,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    try:
        equilibrium.equilibrium_control_fbsde(
            A_hat=A_hat,
            nu_bar=nu_bar,
            params=params,
        )
        assert False, "Expected ValueError for wrong nu_bar length"
    except ValueError:
        assert True


def test_solve_mean_field_fixed_point_single_population_shapes():
    N = control_params.N
    A_hat = np.zeros(N)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=control_params.Q0,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    nu_list, q_list, nu_bar = equilibrium.solve_mean_field_fixed_point(
        A_hat_list=[A_hat],
        weights=np.array([1.0]),
        param_list=[params],
    )

    assert isinstance(nu_list, list)
    assert isinstance(q_list, list)
    assert len(nu_list) == 1
    assert len(q_list) == 1
    assert nu_list[0].shape == (N,)
    assert q_list[0].shape == (N + 1,)
    assert nu_bar.shape == (N,)
    assert np.all(np.isfinite(nu_bar))


def test_solve_mean_field_fixed_point_weight_sum_must_be_one():
    N = control_params.N
    A_hat = np.zeros(N)

    params = EquilibriumControlParams(
        T=control_params.T,
        N=control_params.N,
        Q0=control_params.Q0,
        a=1.0,
        phi=0.02,
        psi=5.0,
        lam=simulation_params.lambda_,
    )

    try:
        equilibrium.solve_mean_field_fixed_point(
            A_hat_list=[A_hat],
            weights=np.array([0.8]),
            param_list=[params],
        )
        assert False, "Expected ValueError when weights do not sum to one"
    except ValueError:
        assert True
