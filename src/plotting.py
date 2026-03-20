import simulate
import latent
import filtering
import numpy as np
import matplotlib.pyplot as plt


def plot_unimpacted(
    F_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True) -> None:
    """
    Plot the unimpacted price path, with optional latent-drift overlay.

    :param F_t: Unimpacted price path of length N + 1
    :param latent_path: Latent state path of length N + 1
    :param sim_params: Simulation parameters
    :param show_latent: Whether to overlay the latent drift
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
    plt.show()


def plot_impacted(
    S_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True) -> None:
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
    plt.show()

def plot_unimpacted_and_impacted(
    F_t: np.ndarray,
    S_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True) -> None:
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
    plt.show()


def plot_fundamental_filter_results(F_t, latent_path, latent_params, sim_params):
    posterior = filtering.filter_fundamental_prob_state_1(F_t, latent_params, sim_params)

    t = np.linspace(0, sim_params.T, sim_params.N + 1)

    plt.figure(figsize=(10,5))

    # posterior
    plt.plot(t, posterior, label="P(Theta=1 | data)", color="blue")

    # latent (step plot)
    plt.step(t, latent_path, where='post', label="True State", color="red", alpha=0.5)

    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.title("Filtering vs True Latent State")
    plt.show()

def plot_impacted_filter_results(S_t, impact, latent_path, latent_params, sim_params):
    posterior = filtering.filter_impacted_prob_state_1(
        S_t, latent_params, sim_params, impact
    )

    t = np.linspace(0, sim_params.T, sim_params.N + 1)

    plt.figure(figsize=(10,5))

    # posterior
    plt.plot(t, posterior, label="P(Theta=1 | data, impacted)", color="green")

    # latent (step plot)
    plt.step(t, latent_path, where='post',
             label="True State", color="red", alpha=0.5)

    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.title("Impacted Filtering vs True Latent State")
    plt.show()

def plot_fundamental_vs_impacted(F_t, S_t, impact,
                                latent_path,
                                latent_params,
                                sim_params):

    posterior_F = filtering.filter_fundamental_prob_state_1(
        F_t, latent_params, sim_params
    )

    posterior_S = filtering.filter_impacted_prob_state_1(
        S_t, latent_params, sim_params, impact
    )

    t = np.linspace(0, sim_params.T, sim_params.N + 1)

    plt.figure(figsize=(12,6))

    # fundamental
    plt.plot(t, posterior_F, label="Fundamental Filter", color="blue")

    # impacted
    plt.plot(t, posterior_S, label="Impacted Filter", color="green", linestyle="--")

    # latent
    plt.step(t, latent_path, where='post',
             label="True State", color="red", alpha=0.5)

    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.title("Fundamental vs Impacted Filtering")
    plt.show()