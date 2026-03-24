from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import latent
import numpy as np
import simulate
import filtering
import control
import plotting
import equilibrium
import pipelines
from params import (
    latent_params,
    simulation_params,
    control_params,
    SubPopParams,
)


def make_default_subpops() -> list[SubPopParams]:
    """
    Construct the default heterogeneous subpopulations used in the full demo.
    """
    return [
        SubPopParams(
            name="Optimistic",
            weight=0.5,
            prior=0.8,
            Q0=1.0,
            kappa=0.5,
        ),
        SubPopParams(
            name="Pessimistic",
            weight=0.5,
            prior=0.2,
            Q0=1.0,
            kappa=2.0,
        ),
    ]


def validate_subpops(subpops: list[SubPopParams]) -> None:
    """
    Validate the basic consistency of subpopulation inputs.
    """
    if len(subpops) == 0:
        raise ValueError("subpops must be non-empty")

    total_weight = sum(sp.weight for sp in subpops)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            f"Subpopulation weights must sum to 1.0, got {total_weight:.6f}"
        )

    for sp in subpops:
        if not (0.0 <= sp.prior <= 1.0):
            raise ValueError(f"{sp.name}: prior must lie in [0, 1]")
        if sp.Q0 < 0:
            raise ValueError(f"{sp.name}: Q0 must be nonnegative")
        if sp.kappa <= 0:
            raise ValueError(f"{sp.name}: kappa must be positive")
        if sp.weight < 0:
            raise ValueError(f"{sp.name}: weight must be nonnegative")


def print_run_summary(subpops: list[SubPopParams]) -> None:
    """
    Print a compact summary of the run configuration.
    """
    print("Running full demo...")
    print(
        f"SimulationParams: T={simulation_params.T}, N={simulation_params.N}, "
        f"sigma={simulation_params.sigma}, S0={simulation_params.S0}, "
        f"A0={simulation_params.A0}, A1={simulation_params.A1}, "
        f"lambda_={simulation_params.lambda_}"
    )
    print(
        f"LatentParams: "
        f"{latent_params}"
    )
    print(
        f"ControlParams: T={control_params.T}, N={control_params.N}, Q0={control_params.Q0}"
    )
    print("Subpopulations:")
    for sp in subpops:
        print(
            f"  - {sp.name}: weight={sp.weight}, prior={sp.prior}, "
            f"Q0={sp.Q0}, kappa={sp.kappa}"
        )


def maybe_plot(plot_fn, *args, **kwargs) -> None:
    """
    Small wrapper so the plotting section reads cleanly.
    """
    plot_fn(*args, **kwargs)


def main() -> None:
    np.random.seed(42)

    subpops = pipelines.make_default_subpops()
    K = len(subpops)

    HAS_IMPACTED_FILTER = hasattr(filtering, "filter_impacted_prob_state_1")
    print(f"Impacted filtering enabled: {HAS_IMPACTED_FILTER}")

    # 1. Build latent path, fundamental price, posteriors, and filtered drifts
    signal_results = pipelines.build_filtered_signals(
        subpops=subpops,
        latent_params=latent_params,
        sim_params=simulation_params,
        seed=42,
    )

    latent_path = signal_results["latent_path"]
    F_t = signal_results["F_t"]
    pi_fund_k = signal_results["pi_k"]
    A_hat_k = signal_results["A_hat_k"]

    # 2. Solve mean-field fixed point
    weights = np.array([sp.weight for sp in subpops], dtype=np.float64)

    param_list = [
        control.EquilibriumControlParams(
            T=control_params.T,
            N=control_params.N,
            Q0=sp.Q0,
            a=1.0,
            phi=0.05,
            psi=5.0,
            lam=simulation_params.lambda_,
        )
        for sp in subpops
    ]

    A_hat_list = [A_hat_k[i, :-1] for i in range(K)]

    nu_list, q_list, nu_bar = equilibrium.solve_mean_field_fixed_point(
        A_hat_list=A_hat_list,
        weights=weights,
        param_list=param_list,
    )

    nu_hat_k = np.array(nu_list)
    q_hat_k = np.array(q_list)

    # 3. Simulate impacted price
    S_t = simulate.simulate_impacted_price(
        F_t=F_t,
        nu_hat=nu_bar,
        params=simulation_params,
    )

    # 4. Core plots
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
        q_hat_k=q_hat_k,
        sim_params=simulation_params,
        subpops=subpops,
        q_bar=True,
    )

    plotting.plot_price_distortion(
        F_t=F_t,
        S_t=S_t,
        sim_params=simulation_params,
    )

    # 5. Optional impacted filtering block
    if HAS_IMPACTED_FILTER:
        pi_imp_k = np.empty((K, simulation_params.N + 1), dtype=np.float64)

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