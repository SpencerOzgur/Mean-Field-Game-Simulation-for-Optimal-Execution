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
import equilibrium
from control import EquilibriumControlParams
from params import latent_params, simulation_params, control_params


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

    phi_base = 0.02
    psi = 5.0
    a = 1.0

    mean_abs_rates = []
    midpoint_inventories = []
    terminal_inventories = []

    print("Sensitivity verification: kappa in FBSDE equilibrium control")
    print("-" * 60)

    for kappa in kappas:
        phi_eff = phi_base * kappa

        params = EquilibriumControlParams(
            T=control_params.T,
            N=control_params.N,
            Q0=control_params.Q0,
            a=a,
            phi=phi_eff,
            psi=psi,
            lam=simulation_params.lambda_,
        )

        A_hat_list = [A_hat[:-1]]
        weights = np.array([1.0], dtype=np.float64)

        nu_list, q_list, nu_bar = equilibrium.solve_mean_field_fixed_point(
            A_hat_list=A_hat_list,
            weights=weights,
            param_list=[params],
        )

        nu_hat = nu_list[0]
        q = q_list[0]

        mean_nu = np.mean(nu_hat)
        mean_abs_rate = np.mean(np.abs(nu_hat))
        max_abs_rate = np.max(np.abs(nu_hat))
        midpoint_inventory = q[len(q) // 2]
        terminal_inventory = q[-1]
        total_traded = np.sum(np.abs(nu_hat)) * (params.T / params.N)

        mean_abs_rates.append(mean_abs_rate)
        midpoint_inventories.append(midpoint_inventory)
        terminal_inventories.append(terminal_inventory)

        print(f"kappa = {kappa}")
        print(f"  phi_eff          = {phi_eff:.6f}")
        print(f"  mean nu_t        = {mean_nu:.6f}")
        print(f"  mean |nu_t|      = {mean_abs_rate:.6f}")
        print(f"  max |nu_t|       = {max_abs_rate:.6f}")
        print(f"  midpoint inv.    = {midpoint_inventory:.6f}")
        print(f"  terminal inv.    = {terminal_inventory:.6f}")
        print(f"  total traded     = {total_traded:.6f}")
        print()

    print("Interpretation:")
    print(
        "  Here kappa scales the running inventory penalty via phi_eff = phi_base * kappa."
    )
    print("  In the current regime, if the outputs barely move with kappa,")
    print(
        "  that suggests the equilibrium is relatively insensitive to inventory aversion."
    )
    print()

    print("Monotonicity check (informal):")
    print(f"  mean |nu_t| values: {mean_abs_rates}")
    print(f"  midpoint inv. values: {midpoint_inventories}")
    print(f"  terminal inv. values: {terminal_inventories}")
    print("Done.")


if __name__ == "__main__":
    main()
