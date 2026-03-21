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
from params import latent_params, simulation_params, control_params


class DemoSubPop:
    def __init__(self, name: str, Q0: float, weight: float = 1.0):
        self.name = name
        self.Q0 = Q0
        self.weight = weight


def main() -> None:
    np.random.seed(42)

    print("Running control demo...")

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

    A_hat = pi * simulation_params.A1 + (1.0 - pi) * simulation_params.A0

    nu_hat = control.alpha_inventory_control(
        A_hat[:-1],
        params=control_params,
        kappa=1.0,
    )

    S_t = simulate.simulate_impacted_price(
        F_t=F_t,
        nu_hat=nu_hat,
        params=simulation_params,
    )

    subpops = [DemoSubPop("Representative Agent", Q0=control_params.Q0)]

    plotting.plot_unimpacted_and_impacted(
        F_t=F_t,
        S_t=S_t,
        latent_path=latent_path,
        sim_params=simulation_params,
        show_latent=True,
    )

    plotting.plot_fundamental_posteriors(
        pi_k=np.expand_dims(pi, axis=0),
        latent_path=latent_path,
        sim_params=simulation_params,
        subpops=subpops,
    )

    plotting.plot_estimated_drifts(
        A_hat_k=np.expand_dims(A_hat, axis=0),
        latent_path=latent_path,
        sim_params=simulation_params,
        subpops=subpops,
        A0=simulation_params.A0,
        A1=simulation_params.A1,
    )

    plotting.plot_controls_subpops(
        nu_hat_k=np.expand_dims(nu_hat, axis=0),
        nu_bar=nu_hat,
        sim_params=simulation_params,
        subpops=subpops,
    )

    plotting.plot_inventories_subpops(
        nu_hat_k=np.expand_dims(nu_hat, axis=0),
        sim_params=simulation_params,
        subpops=subpops,
        q_bar=True,
    )

    plotting.plot_price_distortion(
        F_t=F_t,
        S_t=S_t,
        sim_params=simulation_params,
    )

    print("Control demo complete.")


if __name__ == "__main__":
    main()