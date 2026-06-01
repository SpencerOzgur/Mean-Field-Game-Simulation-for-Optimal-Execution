from pathlib import Path
import json
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "output" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def inventory_from_control(nu: np.ndarray, q0: float, T: float, N: int) -> np.ndarray:
    dt = T / N
    q = np.empty(N + 1)
    q[0] = q0

    for n in range(N):
        q[n + 1] = max(q[n] - nu[n] * dt, 0.0)

    return q


def export_quantitative_results(
    Ft,
    St,
    pi_k,
    pi_imp_k,
    A_hat_k,
    nu_hat_k,
    nu_bar,
    individual_nu,
    individual_labels,
    subpops,
    sim_params,
    convergence_errors=None,
    filename_prefix="mfg_results",
):
    T, N = sim_params.T, sim_params.N
    dt = T / N

    impact = St - Ft

    subpop_inventory = {}
    for i, sp in enumerate(subpops):
        subpop_inventory[sp.name] = inventory_from_control(
            nu_hat_k[i],
            q0=sp.Q0,
            T=T,
            N=N,
        )

    aggregate_inventory = inventory_from_control(
        nu_bar,
        q0=sum(sp.weight * sp.Q0 for sp in subpops),
        T=T,
        N=N,
    )

    individual_terminal_inventory = []
    individual_total_volume = []

    for nu_i in individual_nu:
        q_i = inventory_from_control(nu_i, q0=1.0, T=T, N=N)
        individual_terminal_inventory.append(q_i[-1])
        individual_total_volume.append(np.sum(np.abs(nu_i)) * dt)

    metrics = {
        "price_metrics": {
            "terminal_fundamental_price": float(Ft[-1]),
            "terminal_impacted_price": float(St[-1]),
            "terminal_price_distortion": float(impact[-1]),
            "mean_absolute_price_distortion": float(np.mean(np.abs(impact))),
            "max_absolute_price_distortion": float(np.max(np.abs(impact))),
        },
        "inventory_metrics": {
            "terminal_aggregate_inventory": float(aggregate_inventory[-1]),
            "mean_terminal_individual_inventory": float(
                np.mean(individual_terminal_inventory)
            ),
            "std_terminal_individual_inventory": float(
                np.std(individual_terminal_inventory)
            ),
            "mean_individual_trading_volume": float(np.mean(individual_total_volume)),
            "std_individual_trading_volume": float(np.std(individual_total_volume)),
        },
        "control_metrics": {
            "mean_aggregate_trading_rate": float(np.mean(nu_bar)),
            "max_aggregate_trading_rate": float(np.max(nu_bar)),
            "min_aggregate_trading_rate": float(np.min(nu_bar)),
            "aggregate_trading_volume": float(np.sum(np.abs(nu_bar)) * dt),
        },
        "filtering_metrics": {
            "mean_fundamental_posterior": float(np.mean(pi_k)),
            "mean_impacted_posterior": float(np.mean(pi_imp_k)),
            "mean_absolute_posterior_difference": float(
                np.mean(np.abs(pi_imp_k - pi_k))
            ),
            "max_absolute_posterior_difference": float(
                np.max(np.abs(pi_imp_k - pi_k))
            ),
        },
    }

    for i, sp in enumerate(subpops):
        q_i = subpop_inventory[sp.name]

        metrics[f"{sp.name}_metrics"] = {
            "prior": float(sp.prior),
            "kappa": float(sp.kappa),
            "weight": float(sp.weight),
            "initial_inventory": float(sp.Q0),
            "terminal_inventory": float(q_i[-1]),
            "total_trading_volume": float(np.sum(np.abs(nu_hat_k[i])) * dt),
            "mean_trading_rate": float(np.mean(nu_hat_k[i])),
            "max_trading_rate": float(np.max(nu_hat_k[i])),
            "mean_posterior_fundamental": float(np.mean(pi_k[i])),
            "mean_posterior_impacted": float(np.mean(pi_imp_k[i])),
            "posterior_distortion": float(np.mean(np.abs(pi_imp_k[i] - pi_k[i]))),
        }

    if convergence_errors is not None:
        convergence_errors = np.asarray(convergence_errors, dtype=float)

        metrics["mean_field_convergence"] = {
            "initial_error": float(convergence_errors[0]),
            "final_error": float(convergence_errors[-1]),
            "error_reduction_factor": float(
                convergence_errors[0] / convergence_errors[-1]
            ),
            "iterations": int(len(convergence_errors)),
        }

    json_path = RESULTS_DIR / f"{filename_prefix}.json"
    csv_path = RESULTS_DIR / f"{filename_prefix}.csv"

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    rows = []
    for category, values in metrics.items():
        for metric, value in values.items():
            rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "value": value,
                }
            )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return metrics