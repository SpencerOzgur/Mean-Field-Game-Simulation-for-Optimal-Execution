import simulate
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(name: str):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filepath = OUTPUT_DIR / f"{name}_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_unimpacted(
    F_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True,
) -> None:
    """
    Plot the unimpacted price path, with optional latent-drift overlay.
    """
    F_t = np.asarray(F_t, dtype=np.float64)
    latent_path = np.asarray(latent_path)

    if F_t.shape != (sim_params.N + 1,):
        raise ValueError("F_t must have length N + 1")
    if latent_path.shape != (sim_params.N + 1,):
        raise ValueError("latent_path must have length N + 1")

    t_grid = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(t_grid, F_t, label="Fundamental Price $F_t$")

    if show_latent:
        drift = simulate.latent_to_drift(latent_path, sim_params).astype(np.float64)

        # map drift to the price scale for visibility
        if np.allclose(drift, drift[0]):
            drift_scaled = np.full_like(F_t, np.mean(F_t))
        else:
            drift_scaled = (drift - drift.mean()) / drift.std()
            drift_scaled = drift_scaled * (0.25 * np.std(F_t)) + np.mean(F_t)

        plt.step(
            t_grid,
            drift_scaled,
            where="post",
            linestyle="--",
            label="Latent Drift (scaled)",
        )

    plt.title("Unimpacted Price Path")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.tight_layout()
    _save_fig("Fundamental_Price_Path")


def plot_impacted(
    S_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True,
) -> None:
    """
    Plot the impacted price path, with optional latent-drift overlay.
    """

    S_t = np.asarray(S_t, dtype=np.float64)
    latent_path = np.asarray(latent_path)

    if S_t.shape != (sim_params.N + 1,):
        raise ValueError("S_t must have length N + 1")
    if latent_path.shape != (sim_params.N + 1,):
        raise ValueError("latent_path must have length N + 1")

    t_grid = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(t_grid, S_t, label="Impacted Price $S_t$")

    if show_latent:
        drift = simulate.latent_to_drift(latent_path, sim_params).astype(np.float64)

        if np.allclose(drift, drift[0]):
            drift_scaled = np.full_like(S_t, np.mean(S_t))
        else:
            drift_scaled = (drift - drift.mean()) / drift.std()
            drift_scaled = drift_scaled * (0.25 * np.std(S_t)) + np.mean(S_t)

        plt.step(
            t_grid,
            drift_scaled,
            where="post",
            linestyle="--",
            label="Latent Drift (scaled)",
        )

    plt.title("Impacted Price Path")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig("Impacted_Price_Path")


def plot_unimpacted_and_impacted(
    F_t: np.ndarray,
    S_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True,
) -> None:
    """
    Plot both fundamental and impacted price paths together.
    """

    F_t = np.asarray(F_t, dtype=np.float64)
    S_t = np.asarray(S_t, dtype=np.float64)
    latent_path = np.asarray(latent_path)

    if F_t.shape != (sim_params.N + 1,):
        raise ValueError("F_t must have length N + 1")
    if S_t.shape != (sim_params.N + 1,):
        raise ValueError("S_t must have length N + 1")
    if latent_path.shape != (sim_params.N + 1,):
        raise ValueError("latent_path must have length N + 1")

    t_grid = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(t_grid, F_t, label="Fundamental Price $F_t$")
    plt.plot(t_grid, S_t, label="Impacted Price $S_t$")

    if show_latent:
        drift = simulate.latent_to_drift(latent_path, sim_params).astype(np.float64)

        if np.allclose(drift, drift[0]):
            drift_scaled = np.full_like(F_t, np.mean(F_t))
        else:
            drift_scaled = (drift - drift.mean()) / drift.std()
            drift_scaled = drift_scaled * (0.25 * np.std(F_t)) + np.mean(F_t)

        plt.step(
            t_grid,
            drift_scaled,
            where="post",
            linestyle="--",
            label="Latent Drift (scaled)",
        )

    plt.title("Fundamental vs Impacted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig("Fundamental_vs_Impacted_Price")


def plot_fundamental_posteriors(pi_k, latent_path, sim_params, subpops):
    """
    Plot the posteriors of fundamental path.
    """
    t = np.linspace(0, sim_params.T, sim_params.N + 1)
    plt.figure(figsize=(10, 5))
    for i, sp in enumerate(subpops):
        plt.plot(t, pi_k[i], label=f"{sp.name} posterior")
    plt.step(t, latent_path, where="post", label="True State", color="black", alpha=0.5)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.title("Fundamental Filtering vs True Latent State")
    plt.xlabel("Time")
    plt.ylabel("Posterior Probability")
    plt.tight_layout()
    _save_fig("Fundamental_Posteriors")


def plot_impacted_posteriors(pi_imp_k, latent_path, sim_params, subpops):
    """
    Plot the posteriors of impacted path.
    """
    t = np.linspace(0, sim_params.T, sim_params.N + 1)
    plt.figure(figsize=(10, 5))
    for i, sp in enumerate(subpops):
        plt.plot(t, pi_imp_k[i], label=f"{sp.name} impacted posterior")
    plt.step(t, latent_path, where="post", label="True State", color="black", alpha=0.5)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.title("Impacted Filtering vs True Latent State")
    plt.xlabel("Time")
    plt.ylabel("Posterior Probability")
    plt.tight_layout()
    _save_fig("Impacted_Posteriors")


def plot_fundamental_vs_impacted_posteriors(
    pi_fund_k, pi_imp_k, latent_path, sim_params, subpops
):
    """
    Plot fundamental vs impacted posteriors for subpopulations
    """
    t = np.linspace(0, sim_params.T, sim_params.N + 1)
    plt.figure(figsize=(12, 6))
    for i, sp in enumerate(subpops):
        plt.plot(t, pi_fund_k[i], label=f"{sp.name} fundamental")
        plt.plot(t, pi_imp_k[i], linestyle="--", label=f"{sp.name} impacted")
    plt.step(t, latent_path, where="post", label="True State", color="black", alpha=0.4)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.title("Fundamental vs Impacted Filtering by Subpopulation")
    plt.xlabel("Time")
    plt.ylabel("Posterior Probability")
    plt.tight_layout()
    _save_fig("Fundamental_vs_Impacted_Posteriors")


def plot_estimated_drifts(A_hat_k, latent_path, sim_params, subpops, A0, A1):
    """
    Plot estimated drift for each subpopulation
    """
    t = np.linspace(0, sim_params.T, sim_params.N + 1)
    true_drift = np.where(latent_path == 0, A0, A1)

    plt.figure(figsize=(12, 6))

    plt.step(t, true_drift, where="post", label="True Drift", color="black", alpha=0.7)

    for i, sp in enumerate(subpops):
        plt.plot(t, A_hat_k[i], label=f"{sp.name} estimated drift")

    plt.legend()
    plt.title("Estimated Drift by Subpopulation")
    plt.xlabel("Time")
    plt.ylabel("Drift")
    plt.tight_layout()
    _save_fig("Estimated_Drifts")


def plot_controls_subpops(nu_hat_k, nu_bar, sim_params, subpops):
    """
    Plot trading rates for each subpopulation and the aggregate rate.
    """
    t = np.linspace(0, sim_params.T, sim_params.N)

    plt.figure(figsize=(12, 6))

    for i, sp in enumerate(subpops):
        plt.plot(t, nu_hat_k[i], label=f"{sp.name} control")

    plt.plot(
        t, nu_bar, label="Aggregate control", color="black", linestyle="--", linewidth=2
    )

    plt.legend()
    plt.title("Trading Rates by Subpopulation")
    plt.xlabel("Time")
    plt.ylabel("Trading Rate")
    plt.tight_layout()
    _save_fig("SubPop_Controls")


def plot_inventories_subpops(q_hat_k, sim_params, subpops, q_bar=False):
    q_hat_k = np.asarray(q_hat_k, dtype=np.float64)

    if q_hat_k.ndim != 2:
        raise ValueError("q_hat_k must be a 2-D array")
    if q_hat_k.shape[0] != len(subpops):
        raise ValueError("q_hat_k must have one row per subpopulation")
    if q_hat_k.shape[1] != sim_params.N + 1:
        raise ValueError("q_hat_k must have shape (K, N+1)")

    t = np.linspace(0, sim_params.T, sim_params.N + 1)

    plt.figure(figsize=(12, 6))

    for i, sp in enumerate(subpops):
        plt.plot(t, q_hat_k[i], label=f"{sp.name} inventory")

    if q_bar:
        q_agg = np.zeros(sim_params.N + 1, dtype=np.float64)
        for i, sp in enumerate(subpops):
            q_agg += sp.weight * q_hat_k[i]

        plt.plot(
            t,
            q_agg,
            label="Aggregate inventory",
            color="black",
            linestyle="--",
            linewidth=2,
        )

    plt.legend()
    plt.title("Inventory Paths by Subpopulation")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.tight_layout()
    _save_fig("Inventory_Path_by_SubPop")


def plot_price_distortion(F_t, S_t, sim_params):
    """
    Plot how much price is distorted by mean-field
    """
    t = np.linspace(0, sim_params.T, sim_params.N + 1)
    distortion = S_t - F_t

    plt.figure(figsize=(12, 5))
    plt.plot(t, distortion, label="S_t - F_t")
    plt.axhline(0.0, color="black", linestyle="--", alpha=0.6)

    plt.legend()
    plt.title("Price Distortion from Market Impact")
    plt.xlabel("Time")
    plt.ylabel("Distortion")
    plt.tight_layout()
    _save_fig("Price_Distortion")


def plot_individual_vs_mean_inventory(
    agent_inventories,
    q_bar_mfg,
    q_bar_emp,
    sim_params,
    subpops,
):
    t = np.linspace(0, sim_params.T, sim_params.N + 1)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    plt.figure(figsize=(12, 7))

    for k, sp in enumerate(subpops):
        color = colors[k % len(colors)]

        for q_i in agent_inventories[k]:
            plt.plot(t, q_i, color=color, alpha=0.2, linewidth=1)

        plt.plot(
            t,
            q_bar_mfg[k],
            color=color,
            linewidth=3,
            label=f"{sp.name} mean field",
        )

        plt.plot(
            t,
            q_bar_emp[k],
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f"{sp.name} empirical mean",
        )

    plt.title("Individual vs Mean Field Inventory")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.tight_layout()
    _save_fig("Individual_Vs_Mean_Inventory")
