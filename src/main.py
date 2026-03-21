import latent
import simulate
import filtering
import control
import plotting
import numpy as np
import matplotlib.pyplot as plt
from params import SubPopParams
from params import latent_params, simulation_params, control_params

SubPop1 = SubPopParams(
    name='SubPop1',
    weight=0.5,
    prior=0.8,
    Q0=1.0,
    kappa=0.5
)

SubPop2 = SubPopParams(
    name='SubPop2',
    weight=0.5,
    prior=0.2,
    Q0=1.0,
    kappa=2.0
)

SubPops = np.array([
    SubPop1, SubPop2
])

latent_path = latent.simulate_latent_path(params=latent_params)
Ft = simulate.simulate_fundamental_path(latent_path=latent_path, params=simulation_params)

pi_k = np.empty((2, simulation_params.N + 1))
A_hat_k = np.empty((2, simulation_params.N + 1))
nu_hat_k = np.empty((2, simulation_params.N))

for i, sp in enumerate(SubPops):
    pi_k[i] = filtering.filter_fundamental_prob_state_1(
        F_t=Ft,
        latent_params=latent_params,
        sim_params=simulation_params,
        prior=[1 - sp.prior, sp.prior],
    )

    A_hat_k[i] = (
        pi_k[i] * simulation_params.A1
        + (1 - pi_k[i]) * simulation_params.A0
    )

    nu_hat_k[i] = control.alpha_inventory_control(
        A_hat_k[i, :-1],
        params=control.ControlParams(
            T=control_params.T,
            N=control_params.N,
            Q0=sp.Q0
        ),
        kappa=sp.kappa
    )

nu_bar = SubPop1.weight * nu_hat_k[0] + SubPop2.weight * nu_hat_k[1]

St = simulate.simulate_impacted_price(
    F_t=Ft,
    nu_hat=nu_bar,
    params=simulation_params
)

pi_imp_k = np.empty((len(SubPops), simulation_params.N + 1))

for i, sp in enumerate(SubPops):
    pi_imp_k[i] = filtering.filter_impacted_prob_state_1(
        S_t=St,
        latent_params=latent_params,
        sim_params=simulation_params,
        impact=nu_bar,
        prior=[1 - sp.prior, sp.prior],
    )


plotting.plot_unimpacted_and_impacted(F_t=Ft, S_t=St, latent_path=latent_path, sim_params=simulation_params)


plotting.plot_estimated_drifts(
    A_hat_k=A_hat_k,
    latent_path=latent_path,
    sim_params=simulation_params,
    subpops=SubPops,
    A0=simulation_params.A0,
    A1=simulation_params.A1
)

plotting.plot_controls_subpops(
    nu_hat_k=nu_hat_k,
    nu_bar=nu_bar,
    sim_params=simulation_params,
    subpops=SubPops
)

plotting.plot_inventories_subpops(
    nu_hat_k=nu_hat_k,
    sim_params=simulation_params,
    subpops=SubPops,
    q_bar=True
)

plotting.plot_price_distortion(
    F_t=Ft,
    S_t=St,
    sim_params=simulation_params
)

plotting.plot_fundamental_vs_impacted_posteriors(
    pi_fund_k=pi_k,
    pi_imp_k=pi_imp_k,
    latent_path=latent_path,
    sim_params=simulation_params,
    subpops=SubPops
)
