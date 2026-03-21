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
import plotting
from params import latent_params, simulation_params


class DemoSubPop:
    def __init__(self, name: str):
        self.name = name


def main() -> None:
    np.random.seed(42)

    print("Running latent demo...")

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

    subpops = [DemoSubPop("Representative Agent")]

    plotting.plot_unimpacted(
        F_t=F_t,
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

    print("Latent demo complete.")


if __name__ == "__main__":
    main()