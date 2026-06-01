import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import simulate

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("dark_background")

COLORS = {
    "blue": "#4C9AFF",
    "orange": "#FF9F43",
    "green": "#2ED573",
    "red": "#FF4757",
    "purple": "#A55EEA",
    "cyan": "#00D2D3",
    "white": "#F5F6FA",
    "gray": "#A4B0BE",
    "background": "#0B0F14",
}


def _save_fig(name: str) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filepath = OUTPUT_DIR / f"{name}_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()


def _format_ax(ax, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    ax.set_title(title, fontsize=16, weight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_facecolor(COLORS["background"])
    ax.grid(True, alpha=0.18)

    for spine in ax.spines.values():
        spine.set_alpha(0.35)

    if legend:
        ax.legend(frameon=True, framealpha=0.15, fontsize=10)


def _smooth(x: np.ndarray, window: int = 9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    if window <= 1:
        return x

    if window % 2 == 0:
        window += 1

    pad = window // 2
    x_pad = np.pad(x, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def _inventory_from_control(nu: np.ndarray, q0: float, sim_params) -> np.ndarray:
    dt = sim_params.T / sim_params.N
    q = np.empty(sim_params.N + 1)
    q[0] = q0

    for n in range(sim_params.N):
        q[n + 1] = max(q[n] - nu[n] * dt, 0.0)

    return q


def plot_unimpacted(
    F_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True,
) -> None:
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, F_t, color=COLORS["blue"], linewidth=2.4, label=r"Fundamental price $F_t$")

    if show_latent:
        drift = simulate.latent_to_drift(latent_path, sim_params).astype(np.float64)
        drift_scaled = np.full_like(F_t, np.mean(F_t))

        if not np.allclose(drift, drift[0]):
            drift_scaled = (drift - drift.mean()) / drift.std()
            drift_scaled = drift_scaled * (0.25 * np.std(F_t)) + np.mean(F_t)

        ax.step(
            t,
            drift_scaled,
            where="post",
            color=COLORS["white"],
            linestyle="--",
            alpha=0.55,
            label="Latent drift scaled",
        )

    _format_ax(ax, "Unimpacted Fundamental Price Path", "Time", "Price")
    fig.tight_layout()
    _save_fig("Fundamental_Price_Path")


def plot_impacted(
    S_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True,
) -> None:
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, S_t, color=COLORS["orange"], linewidth=2.4, label=r"Impacted price $S_t$")

    if show_latent:
        drift = simulate.latent_to_drift(latent_path, sim_params).astype(np.float64)
        drift_scaled = np.full_like(S_t, np.mean(S_t))

        if not np.allclose(drift, drift[0]):
            drift_scaled = (drift - drift.mean()) / drift.std()
            drift_scaled = drift_scaled * (0.25 * np.std(S_t)) + np.mean(S_t)

        ax.step(
            t,
            drift_scaled,
            where="post",
            color=COLORS["white"],
            linestyle="--",
            alpha=0.55,
            label="Latent drift scaled",
        )

    _format_ax(ax, "Impacted Price Path", "Time", "Price")
    fig.tight_layout()
    _save_fig("Impacted_Price_Path")


def plot_unimpacted_and_impacted(
    F_t: np.ndarray,
    S_t: np.ndarray,
    latent_path: np.ndarray,
    sim_params: simulate.SimulationParams,
    show_latent: bool = True,
) -> None:
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)
    impact = S_t - F_t

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(t, F_t, linewidth=2.4, color=COLORS["blue"], label=r"Fundamental price $F_t$")
    ax1.plot(t, S_t, linewidth=2.4, color=COLORS["orange"], label=r"Impacted price $S_t$")

    ax2 = ax1.twinx()
    ax2.plot(
        t,
        impact,
        linestyle="--",
        linewidth=1.8,
        color=COLORS["cyan"],
        alpha=0.85,
        label=r"Impact component $S_t - F_t$",
    )
    ax2.set_ylabel("Impact Component", color=COLORS["cyan"], fontsize=12)
    ax2.tick_params(axis="y", colors=COLORS["cyan"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, framealpha=0.15, fontsize=10)

    _format_ax(ax1, "Fundamental vs Impacted Price", "Time", "Price", legend=False)
    fig.tight_layout()
    _save_fig("Fundamental_vs_Impacted_Price")


def plot_price_distortion(F_t: np.ndarray, S_t: np.ndarray, sim_params) -> None:
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)
    distortion = S_t - F_t

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, distortion, linewidth=2.5, color=COLORS["cyan"], label=r"$S_t - F_t$")
    ax.axhline(0.0, color=COLORS["white"], linestyle="--", alpha=0.45)

    _format_ax(ax, "Price Distortion from Endogenous Market Impact", "Time", "Distortion")
    fig.tight_layout()
    _save_fig("Price_Distortion")


def plot_fundamental_posteriors(pi_k, latent_path, sim_params, subpops):
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    _shade_regimes(ax, t, latent_path)

    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    for i, sp in enumerate(subpops):
        ax.plot(
            t,
            _smooth(pi_k[i], window=7),
            color=colors[i % len(colors)],
            linewidth=2.5,
            label=f"{sp.name} posterior",
        )

    ax.set_ylim(-0.03, 1.03)
    _format_ax(
        ax,
        "Posterior Beliefs from Fundamental Price Observations",
        "Time",
        r"Posterior Probability $P(A_t = A_1 \mid \mathcal{F}_t)$",
    )
    fig.tight_layout()
    _save_fig("Fundamental_Posteriors")


def plot_impacted_posteriors(pi_imp_k, latent_path, sim_params, subpops):
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    _shade_regimes(ax, t, latent_path)

    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    for i, sp in enumerate(subpops):
        ax.plot(
            t,
            _smooth(pi_imp_k[i], window=7),
            color=colors[i % len(colors)],
            linewidth=2.5,
            label=f"{sp.name} impacted posterior",
        )

    ax.set_ylim(-0.03, 1.03)
    _format_ax(
        ax,
        "Posterior Beliefs from Impacted Price Observations",
        "Time",
        r"Posterior Probability $P(A_t = A_1 \mid \mathcal{F}_t)$",
    )
    fig.tight_layout()
    _save_fig("Impacted_Posteriors")


def _shade_regimes(ax, t, latent_path) -> None:
    ax.fill_between(
        t,
        0,
        1,
        where=latent_path == 1,
        color=COLORS["orange"],
        alpha=0.10,
        step="post",
        label="Bullish regime",
    )

    ax.fill_between(
        t,
        0,
        1,
        where=latent_path == 0,
        color=COLORS["blue"],
        alpha=0.08,
        step="post",
        label="Bearish regime",
    )


def plot_fundamental_vs_impacted_posteriors(
    pi_fund_k,
    pi_imp_k,
    latent_path,
    sim_params,
    subpops,
):
    """
    Cleaner posterior plot.

    Shows:
    - shaded latent regimes
    - faint subpopulation-level posteriors
    - bold aggregate fundamental posterior
    - bold aggregate impacted posterior
    """
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    pi_fund_k = np.asarray(pi_fund_k, dtype=np.float64)
    pi_imp_k = np.asarray(pi_imp_k, dtype=np.float64)

    weights = np.asarray([sp.weight for sp in subpops], dtype=np.float64)
    weights = weights / weights.sum()

    fund_mean = np.average(pi_fund_k, axis=0, weights=weights)
    imp_mean = np.average(pi_imp_k, axis=0, weights=weights)

    fund_mean_smooth = _smooth(fund_mean, window=9)
    imp_mean_smooth = _smooth(imp_mean, window=9)

    fig, ax = plt.subplots(figsize=(12, 6))
    _shade_regimes(ax, t, latent_path)

    for i, sp in enumerate(subpops):
        ax.plot(
            t,
            _smooth(pi_fund_k[i], window=7),
            color=COLORS["blue"],
            linewidth=1.2,
            alpha=0.30,
        )
        ax.plot(
            t,
            _smooth(pi_imp_k[i], window=7),
            color=COLORS["orange"],
            linewidth=1.2,
            alpha=0.30,
            linestyle="--",
        )

    ax.plot(
        t,
        fund_mean_smooth,
        color=COLORS["blue"],
        linewidth=3.2,
        label="Aggregate posterior: fundamental observations",
    )

    ax.plot(
        t,
        imp_mean_smooth,
        color=COLORS["orange"],
        linewidth=3.2,
        linestyle="--",
        label="Aggregate posterior: impacted observations",
    )

    ax.set_ylim(-0.03, 1.03)

    _format_ax(
        ax,
        "Aggregate Posterior Beliefs Under Fundamental vs Impacted Observations",
        "Time",
        r"Posterior Probability $P(A_t = A_1 \mid \mathcal{F}_t)$",
    )

    fig.tight_layout()
    _save_fig("Aggregate_Posterior_Beliefs")


def plot_estimated_drifts(A_hat_k, latent_path, sim_params, subpops, A0, A1):
    """
    Cleaner drift plot.

    Shows:
    - true latent drift as a step function
    - smoothed estimated drift by subpopulation
    """
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)
    true_drift = np.where(latent_path == 0, A0, A1)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.step(
        t,
        true_drift,
        where="post",
        color=COLORS["white"],
        linewidth=3.0,
        alpha=0.65,
        label="True latent drift",
    )

    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    for i, sp in enumerate(subpops):
        smooth_drift = _smooth(A_hat_k[i], window=11)

        ax.plot(
            t,
            smooth_drift,
            color=colors[i % len(colors)],
            linewidth=2.6,
            label=f"{sp.name} filtered drift",
        )

    ax.set_ylim(min(A0, A1) - 0.15 * abs(A1 - A0), max(A0, A1) + 0.15 * abs(A1 - A0))

    _format_ax(
        ax,
        "Smoothed Filtered Drift Estimates by Subpopulation",
        "Time",
        "Estimated Drift",
    )

    fig.tight_layout()
    _save_fig("Estimated_Drifts")


def plot_controls_subpops(nu_hat_k, nu_bar, sim_params, subpops):
    t = np.linspace(0.0, sim_params.T, sim_params.N)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    for i, sp in enumerate(subpops):
        ax.plot(
            t,
            nu_hat_k[i],
            color=colors[i % len(colors)],
            linewidth=2.3,
            label=f"{sp.name} control",
        )

    ax.plot(
        t,
        nu_bar,
        color=COLORS["white"],
        linestyle="--",
        linewidth=3.0,
        label="Aggregate control",
    )

    _format_ax(ax, "Trading Rates by Subpopulation", "Time", "Trading Rate")
    fig.tight_layout()
    _save_fig("SubPop_Controls")


def plot_inventories_subpops(q_hat_k, sim_params, subpops, q_bar=False):
    q_hat_k = np.asarray(q_hat_k, dtype=np.float64)

    if q_hat_k.ndim != 2:
        raise ValueError("q_hat_k must be a 2-D array")
    if q_hat_k.shape[0] != len(subpops):
        raise ValueError("q_hat_k must have one row per subpopulation")
    if q_hat_k.shape[1] != sim_params.N + 1:
        raise ValueError("q_hat_k must have shape (K, N+1)")

    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    for i, sp in enumerate(subpops):
        ax.plot(
            t,
            q_hat_k[i],
            color=colors[i % len(colors)],
            linewidth=2.5,
            label=f"{sp.name} inventory",
        )

    if q_bar:
        q_agg = np.zeros(sim_params.N + 1, dtype=np.float64)
        for i, sp in enumerate(subpops):
            q_agg += sp.weight * q_hat_k[i]

        ax.plot(
            t,
            q_agg,
            label="Aggregate inventory",
            color=COLORS["white"],
            linestyle="--",
            linewidth=3.0,
        )

    _format_ax(ax, "Inventory Paths by Subpopulation", "Time", "Inventory")
    fig.tight_layout()
    _save_fig("Inventory_Path_by_SubPop")


def plot_individual_vs_mean_inventory(
    agent_inventories,
    q_bar_mfg,
    q_bar_emp,
    sim_params,
    subpops,
):
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    fig, ax = plt.subplots(figsize=(12, 7))

    for k, sp in enumerate(subpops):
        color = colors[k % len(colors)]

        for q_i in agent_inventories[k]:
            ax.plot(t, q_i, color=color, alpha=0.10, linewidth=1.0)

        ax.plot(
            t,
            q_bar_mfg[k],
            color=color,
            linewidth=3.0,
            label=f"{sp.name} mean field",
        )

        ax.plot(
            t,
            q_bar_emp[k],
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f"{sp.name} empirical mean",
        )

    _format_ax(ax, "Individual vs Mean-Field Inventory", "Time", "Inventory")
    fig.tight_layout()
    _save_fig("Individual_Vs_Mean_Inventory")


def plot_individual_inventory_paths(
    individual_nu,
    individual_labels,
    nu_hat_k,
    nu_bar,
    sim_params,
    subpops,
):
    t = np.linspace(0.0, sim_params.T, sim_params.N + 1)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    subpop_colors = {
        subpops[0].name: COLORS["blue"],
        subpops[1].name: COLORS["orange"],
    }

    grouped_q = {sp.name: [] for sp in subpops}

    for nu_i, label in zip(individual_nu, individual_labels):
        q_i = _inventory_from_control(nu_i, q0=1.0, sim_params=sim_params)
        grouped_q[label].append(q_i)

        ax.plot(
            t,
            q_i,
            color=subpop_colors.get(label, COLORS["gray"]),
            alpha=0.10,
            linewidth=1.0,
        )

    for i, sp in enumerate(subpops):
        color = subpop_colors.get(sp.name, COLORS["gray"])
        paths = np.asarray(grouped_q[sp.name])

        if len(paths) > 0:
            mean_q = paths.mean(axis=0)
            std_q = paths.std(axis=0)

            ax.fill_between(
                t,
                np.maximum(mean_q - std_q, 0.0),
                np.minimum(mean_q + std_q, 1.05),
                color=color,
                alpha=0.16,
            )

        q_mean = _inventory_from_control(nu_hat_k[i], q0=sp.Q0, sim_params=sim_params)

        ax.plot(
            t,
            q_mean,
            color=color,
            linewidth=3.0,
            label=f"{sp.name} mean inventory",
        )

    q_bar = _inventory_from_control(
        nu_bar,
        q0=sum(sp.weight * sp.Q0 for sp in subpops),
        sim_params=sim_params,
    )

    ax.plot(
        t,
        q_bar,
        color=COLORS["white"],
        linestyle="--",
        linewidth=3.5,
        label="Aggregate inventory",
    )

    ax.set_ylim(-0.02, 1.05)

    _format_ax(
        ax,
        "Finite-Agent Inventory Dynamics by Subpopulation",
        "Time",
        "Inventory",
    )

    fig.tight_layout()
    _save_fig("Inventory_Dynamics_Individual_Agents")


def plot_mean_field_convergence(errors, tolerance=None):
    """
    Plot mean-field fixed-point convergence.

    Parameters
    ----------
    errors : array-like
        Sequence of fixed-point errors, e.g.
        ||nu_bar^{k+1} - nu_bar^k||.
    tolerance : float, optional
        Optional convergence tolerance to show as a horizontal line.
    """
    errors = np.asarray(errors, dtype=np.float64)

    if errors.ndim != 1:
        raise ValueError("errors must be a one-dimensional array")

    iterations = np.arange(1, len(errors) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        iterations,
        errors,
        marker="o",
        linewidth=2.5,
        color=COLORS["cyan"],
        label=r"Fixed-point error $\|\bar{\nu}^{k+1} - \bar{\nu}^{k}\|$",
    )

    ax.set_yscale("log")

    if tolerance is not None:
        ax.axhline(
            tolerance,
            color=COLORS["red"],
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            label="Tolerance",
        )

    _format_ax(
        ax,
        "Mean-Field Fixed-Point Convergence",
        "Picard Iteration",
        "Error Norm",
    )

    fig.tight_layout()
    _save_fig("Mean_Field_Convergence")