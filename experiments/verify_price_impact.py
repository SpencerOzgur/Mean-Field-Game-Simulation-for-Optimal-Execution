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
from params import latent_params, simulation_params, control_params


def inventory_from_control(nu_hat: np.ndarray, Q0: float, T: float, N: int) -> np.ndarray:
    dt = T / N
    q = np.empty(N + 1)
    q[0] = Q0
    q[1:] = Q0 - np.cumsum(nu_hat) * dt
    return q


def main():
    np.random.seed(42)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    pi = filtering.filter_fundamental_prob_state_1(
        F_t=F_t,
        latent_params=latent_params,
        sim_params=simulation_params,
        prior=[0.5, 0.5],
    )

    A_hat = pi * simulation_params.A1 + (1.0 - pi) * simulation_params.A0

    kappas = [0.25, 0.5, 1.0, 2.0, 5.0]
    dt = control_params.T / control_params.N

    avg_abs_rates = []
    mid_inventories = []

    print("Sensitivity verification: kappa")
    print("-" * 60)

    for kappa in kappas:
        nu_hat = control.alpha_inventory_control(
            A_hat=A_hat[:-1],
            params=control_params,
            kappa=kappa,
        )

        q = inventory_from_control(
            nu_hat=nu_hat,
            Q0=control_params.Q0,
            T=control_params.T,
            N=control_params.N,
        )

        avg_abs_rate = np.mean(np.abs(nu_hat))
        max_abs_rate = np.max(np.abs(nu_hat))
        mid_inventory = q[len(q) // 2]
        final_inventory = q[-1]
        total_traded = np.sum(np.abs(nu_hat)) * dt

        avg_abs_rates.append(avg_abs_rate)
        mid_inventories.append(mid_inventory)

        print(f"kappa = {kappa}")
        print(f"  avg |nu_t|       = {avg_abs_rate:.6f}")
        print(f"  max |nu_t|       = {max_abs_rate:.6f}")
        print(f"  midpoint q_t     = {mid_inventory:.6f}")
        print(f"  final q_T        = {final_inventory:.6f}")
        print(f"  total traded     = {total_traded:.6f}")
        print()

    print("Interpretation:")
    print("  Smaller kappa should typically imply more aggressive trading")
    print("  and therefore lower remaining inventory mid-horizon.")
    print()

    print("Monotonicity check (informal):")
    print(f"  avg |nu_t| values: {avg_abs_rates}")
    print(f"  midpoint q_t values: {mid_inventories}")
    print("Done.")


if __name__ == "__main__":
    main()