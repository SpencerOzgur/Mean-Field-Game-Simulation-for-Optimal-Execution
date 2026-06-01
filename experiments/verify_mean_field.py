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
import population
from params import (
    latent_params,
    simulation_params,
    control_params,
    SubPopParams,
)


def make_subpops():
    return [
        SubPopParams(
            name="SubPop1",
            weight=0.5,
            prior=0.8,
            Q0=100.0,
            kappa=0.5,
        ),
        SubPopParams(
            name="SubPop2",
            weight=0.5,
            prior=0.2,
            Q0=100.0,
            kappa=2.0,
        ),
    ]


def main():
    np.random.seed(42)
    subpops = make_subpops()
    K = len(subpops)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    A_hat_k = np.empty((K, simulation_params.N + 1))
    for i, sp in enumerate(subpops):
        pi = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=simulation_params,
            prior=[1.0 - sp.prior, sp.prior],
        )
        A_hat_k[i] = pi * simulation_params.A1 + (1.0 - pi) * simulation_params.A0

    agent_counts = [5, 10, 20, 50, 100]

    print("Mean-field verification")
    print("-" * 60)

    for n_agents in agent_counts:
        results = population.simulate_agent_inventory_paths(
            A_hat_k=A_hat_k,
            subpops=subpops,
            control_params=control_params,
            sim_params=simulation_params,
            n_agents=n_agents,
            seed=42,
        )

        q_bar_mfg = results["q_bar_mfg"]
        q_bar_emp = results["q_bar_emp"]

        errors = np.mean(np.abs(q_bar_emp - q_bar_mfg), axis=1)

        print(f"n_agents = {n_agents}")
        for k, sp in enumerate(subpops):
            print(f"  {sp.name}: mean abs inventory error = {errors[k]:.6f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
