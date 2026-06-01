from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import population
import plotting
import pipelines
from params import latent_params, simulation_params, control_params


def main():
    np.random.seed(42)

    subpops = pipelines.make_default_subpops()

    signals = pipelines.build_filtered_signals(
        subpops=subpops,
        latent_params=latent_params,
        sim_params=simulation_params,
        seed=42,
    )

    A_hat_k = signals["A_hat_k"]

    pop_results = population.simulate_agent_inventory_paths(
        A_hat_k=A_hat_k,
        subpops=subpops,
        control_params=control_params,
        sim_params=simulation_params,
        n_agents=30,
        seed=42,
        phi=0.2,
        psi=5.0,
        a=1.0,
        q0_scales=[2.0, 2.0],
        clip_q0=False,
        plot=False,
        plot_controls=False,
    )

    plotting.plot_individual_vs_mean_inventory(
        agent_inventories=pop_results["agent_inventories"],
        q_bar_mfg=pop_results["q_bar_mfg"],
        q_bar_emp=pop_results["q_bar_emp"],
        sim_params=simulation_params,
        subpops=subpops,
    )

    plt.show()


if __name__ == "__main__":
    main()
