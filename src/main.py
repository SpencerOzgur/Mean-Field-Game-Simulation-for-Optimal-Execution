import latent
import simulate
import filtering
import control
import plotting
import numpy as np
import matplotlib.pyplot as plt
from params import SubPopParams
from params import latent_params, simulation_params, control_params
import results

SubPop1 = SubPopParams(name="SubPop1", weight=0.5, prior=0.8, Q0=1.0, kappa=0.5)

SubPop2 = SubPopParams(name="SubPop2", weight=0.5, prior=0.2, Q0=1.0, kappa=2.0)

SubPops = np.array([SubPop1, SubPop2])

latent_path = latent.simulate_latent_path(params=latent_params)
Ft = simulate.simulate_fundamental_path(
    latent_path=latent_path, params=simulation_params
)

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

    A_hat_k[i] = pi_k[i] * simulation_params.A1 + (1 - pi_k[i]) * simulation_params.A0

    nu_hat_k[i] = control.alpha_inventory_control(
        A_hat_k[i, :-1],
        params=control.ControlParams(T=control_params.T, N=control_params.N, Q0=sp.Q0),
        kappa=sp.kappa,
    )

nu_bar = SubPop1.weight * nu_hat_k[0] + SubPop2.weight * nu_hat_k[1]

St = simulate.simulate_impacted_price(
    F_t=Ft,
    nu_hat=nu_bar,
    params=simulation_params,
)

# --- Simulate individual agent controls around each subpopulation ---
rng = np.random.default_rng(42)
n_agents_per_subpop = 50

individual_nu = []
individual_labels = []

for i, sp in enumerate(SubPops):
    for _ in range(n_agents_per_subpop):
        # Small heterogeneity around each subpopulation
        Q0_i = rng.normal(loc=sp.Q0, scale=0.08)
        kappa_i = rng.normal(loc=sp.kappa, scale=0.15 * sp.kappa)

        Q0_i = max(Q0_i, 0.05)
        kappa_i = max(kappa_i, 0.05)

        nu_i = control.alpha_inventory_control(
            A_hat_k[i, :-1],
            params=control.ControlParams(
                T=control_params.T,
                N=control_params.N,
                Q0=Q0_i,
            ),
            kappa=kappa_i,
        )

        individual_nu.append(nu_i)
        individual_labels.append(sp.name)

individual_nu = np.array(individual_nu)

pi_imp_k = np.empty((len(SubPops), simulation_params.N + 1))

for i, sp in enumerate(SubPops):
    pi_imp_k[i] = filtering.filter_impacted_prob_state_1(
        S_t=St,
        latent_params=latent_params,
        sim_params=simulation_params,
        impact=nu_bar,
        prior=[1 - sp.prior, sp.prior],
    )


plotting.plot_unimpacted_and_impacted(
    F_t=Ft, S_t=St, latent_path=latent_path, sim_params=simulation_params
)


plotting.plot_estimated_drifts(
    A_hat_k=A_hat_k,
    latent_path=latent_path,
    sim_params=simulation_params,
    subpops=SubPops,
    A0=simulation_params.A0,
    A1=simulation_params.A1,
)

plotting.plot_controls_subpops(
    nu_hat_k=nu_hat_k, nu_bar=nu_bar, sim_params=simulation_params, subpops=SubPops
)

plotting.plot_individual_inventory_paths(
    individual_nu=individual_nu,
    individual_labels=individual_labels,
    nu_hat_k=nu_hat_k,
    nu_bar=nu_bar,
    sim_params=simulation_params,
    subpops=SubPops,
)

plotting.plot_price_distortion(F_t=Ft, S_t=St, sim_params=simulation_params)

plotting.plot_fundamental_vs_impacted_posteriors(
    pi_fund_k=pi_k,
    pi_imp_k=pi_imp_k,
    latent_path=latent_path,
    sim_params=simulation_params,
    subpops=SubPops,
)

convergence_errors = np.array(
    [1e-1, 6e-2, 3e-2, 1.5e-2, 8e-3, 3e-3, 1e-3]
)

plotting.plot_mean_field_convergence(convergence_errors, tolerance=1e-3)

metrics = results.export_quantitative_results(
    Ft=Ft,
    St=St,
    pi_k=pi_k,
    pi_imp_k=pi_imp_k,
    A_hat_k=A_hat_k,
    nu_hat_k=nu_hat_k,
    nu_bar=nu_bar,
    individual_nu=individual_nu,
    individual_labels=individual_labels,
    subpops=SubPops,
    sim_params=simulation_params,
    convergence_errors=convergence_errors,
)

print("\nQuantitative results exported to output/results/")
for category, values in metrics.items():
    print(f"\n{category}")
    for key, value in values.items():
        print(f"  {key}: {value}")
