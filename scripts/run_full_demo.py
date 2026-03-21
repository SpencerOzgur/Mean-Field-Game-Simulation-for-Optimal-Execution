from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import latent
import simulate
import filtering
import control
import plotting
from params import (
    latent_params,
    simulation_params,
    control_params,
    SubPopParams,
)


def main() -> None:
    np.random.seed(42)

    print("Running full demo...")

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
    K = len(subpops)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    pi_fund_k = np.empty((K, simulation_params.N + 1))
    A_hat_k = np.empty((K, simulation_params.N + 1))
    nu_hat_k = np.empty((K, simulation_params.N))

    for i, sp in enumerate(subpops):
        pi_fund_k[i] = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=simulation_params,
            prior=[1.0 - sp.prior, sp.prior],
        )

        A_hat_k[i] = (
            pi_fund_k[i] * simulation_params.A1
            + (1.0 - pi_fund_k[i]) * simulation_params.A0
        )

        local_control_params = control.ControlParams(
            T=control_params.T,
            N=control_params.N,
            Q0=sp.Q0,
        )

        nu_hat_k[i] = control.alpha_inventory_control(
            A_hat_k[i, :-1],
            params=local_control_params,
            kappa=sp.kappa,
        )

    nu_bar = np.zeros(simulation_params.N)
    for i, sp in enumerate(subpops):
        nu_bar += sp.weight * nu_hat_k[i]

    S_t = simulate.simulate_impacted_price(
        F_t=F_t,
        nu_hat=nu_bar,
        params=simulation_params,
    )

    plotting.plot_unimpacted_and_impacted(
        F_t=F_t,
        S_t=S_t,
        latent_path=latent_path,
        sim_params=simulation_params,
        show_latent=True,
    )

    plotting.plot_fundamental_posteriors(
        pi_k=pi_fund_k,
        latent_path=latent_path,
        sim_params=simulation_params,
        subpops=subpops,
    )

    plotting.plot_estimated_drifts(
        A_hat_k=A_hat_k,
        latent_path=latent_path,
        sim_params=simulation_params,
        subpops=subpops,
        A0=simulation_params.A0,
        A1=simulation_params.A1,
    )

    plotting.plot_controls_subpops(
        nu_hat_k=nu_hat_k,
        nu_bar=nu_bar,
        sim_params=simulation_params,
        subpops=subpops,
    )

    plotting.plot_inventories_subpops(
        nu_hat_k=nu_hat_k,
        sim_params=simulation_params,
        subpops=subpops,
        q_bar=True,
    )

    plotting.plot_price_distortion(
        F_t=F_t,
        S_t=S_t,
        sim_params=simulation_params,
    )

    # Optional impacted filtering block
    if hasattr(filtering, "filter_impacted_prob_state_1"):
        pi_imp_k = np.empty((K, simulation_params.N + 1))

        for i, sp in enumerate(subpops):
            pi_imp_k[i] = filtering.filter_impacted_prob_state_1(
                S_t=S_t,
                impact=nu_bar,
                latent_params=latent_params,
                sim_params=simulation_params,
                prior=[1.0 - sp.prior, sp.prior],
            )

        plotting.plot_impacted_posteriors(
            pi_imp_k=pi_imp_k,
            latent_path=latent_path,
            sim_params=simulation_params,
            subpops=subpops,
        )

        plotting.plot_fundamental_vs_impacted_posteriors(
            pi_fund_k=pi_fund_k,
            pi_imp_k=pi_imp_k,
            latent_path=latent_path,
            sim_params=simulation_params,
            subpops=subpops,
        )

    print("Full demo complete.")


if __name__ == "__main__":
    main()