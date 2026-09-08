"""
HJN cross-sectional experiment for the current MFG pipeline.

Runs the existing filtering + mean-field equilibrium solver across the
17 Huang-Jaimungal-Nourian (mid_price, sigma) calibrations.

Expected repo layout:
    data/hjn_market_params.csv
    src/
        latent.py
        simulate.py
        filtering.py
        params.py
        pipelines.py
        control.py
        equilibrium.py
    experiments/
        hjn_cross_section.py

Run from repo root:
    python experiments/hjn_cross_section.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from latent import LatentParams
from simulate import SimulationParams
from control import EquilibriumControlParams
from equilibrium import solve_mean_field_fixed_point
from pipelines import make_default_subpops, build_filtered_signals


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = REPO_ROOT / "data" / "hjn_market_params.csv"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "hjn_cross_section_results.csv"


# ---------------------------------------------------------------------
# Baseline structural parameters
# Only S0 and sigma vary across HJN stocks.
# ---------------------------------------------------------------------

T = 1.0
N = 1000

A0 = -1.0
A1 = 1.0

LATENT_LAMBDA_01 = 3.0
LATENT_LAMBDA_10 = 2.0
THETA0 = 1

PRICE_IMPACT = 0.05

# Control parameters.
# These match the fields required by your current EquilibriumControlParams.
TEMP_IMPACT_A = 1.0
TERMINAL_PENALTY_PSI = 1.0

SEED = 42


LATENT_PARAMS = LatentParams(
    T=T,
    N=N,
    lambda01=LATENT_LAMBDA_01,
    lambda10=LATENT_LAMBDA_10,
    theta0=THETA0,
)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_market_params() -> pd.DataFrame:
    """Load and validate the HJN market parameter table."""
    df = pd.read_csv(DATA_PATH)

    required = {"ticker", "mid_price", "sigma"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{DATA_PATH} missing required columns: {sorted(missing)}"
        )

    if df["ticker"].duplicated().any():
        duplicates = df.loc[df["ticker"].duplicated(), "ticker"].tolist()
        raise ValueError(f"Duplicate tickers found: {duplicates}")

    if (df["mid_price"] <= 0).any():
        raise ValueError("mid_price must be positive")

    if (df["sigma"] <= 0).any():
        raise ValueError("sigma must be positive")

    return df


# ---------------------------------------------------------------------
# One cross-sectional run
# ---------------------------------------------------------------------

def run_one_ticker(
    ticker: str,
    mid_price: float,
    sigma: float,
    seed: int = SEED,
) -> dict:
    """
    Run the current MFG pipeline for one HJN stock calibration.

    The empirical inputs are:
        S0    = HJN mid_price
        sigma = HJN volatility

    All other model parameters are held fixed.
    """

    # ---------------------------------------------------------------
    # 1. Stock-specific simulation parameters
    # ---------------------------------------------------------------

    sim_params = SimulationParams(
        T=T,
        N=N,
        sigma=float(sigma),
        A1=A1,
        A0=A0,
        S0=float(mid_price),
        lambda_=PRICE_IMPACT,
    )

    # ---------------------------------------------------------------
    # 2. Existing heterogeneous subpopulations
    # ---------------------------------------------------------------

    subpops = make_default_subpops()

    # ---------------------------------------------------------------
    # 3. Simulate common latent path + fundamental price + filters
    # ---------------------------------------------------------------

    signals = build_filtered_signals(
        subpops=subpops,
        latent_params=LATENT_PARAMS,
        sim_params=sim_params,
        seed=seed,
    )

    # build_filtered_signals returns:
    #   A_hat_k.shape == (K, N+1)
    #
    # equilibrium_control_fbsde requires each A_hat to have length N.
    # Therefore drop the final observation for the control interval grid.
    A_hat_list = [
        np.asarray(signals["A_hat_k"][k, :-1], dtype=np.float64)
        for k in range(len(subpops))
    ]

    weights = np.asarray(
        [sp.weight for sp in subpops],
        dtype=np.float64,
    )

    # ---------------------------------------------------------------
    # 4. Build one control parameter object per subpopulation
    # ---------------------------------------------------------------

    param_list = [
        EquilibriumControlParams(
            T=T,
            N=N,
            Q0=sp.Q0,
            a=TEMP_IMPACT_A,
            phi=sp.kappa,
            lam=PRICE_IMPACT,
            psi=TERMINAL_PENALTY_PSI,
        )
        for sp in subpops
    ]

    # ---------------------------------------------------------------
    # 5. Solve mean-field fixed point
    #
    # Your current solver returns exactly:
    #   nu_list, q_list, nu_bar
    # ---------------------------------------------------------------

    nu_list, q_list, nu_bar = solve_mean_field_fixed_point(
        A_hat_list=A_hat_list,
        weights=weights,
        param_list=param_list,
    )

    # ---------------------------------------------------------------
    # 6. Summary statistics
    # ---------------------------------------------------------------

    nu_bar = np.asarray(nu_bar, dtype=np.float64)

    terminal_inventory = np.asarray(
        [q[-1] for q in q_list],
        dtype=np.float64,
    )

    initial_inventory = np.asarray(
        [q[0] for q in q_list],
        dtype=np.float64,
    )

    mean_abs_rates = np.asarray(
        [np.mean(np.abs(nu)) for nu in nu_list],
        dtype=np.float64,
    )

    max_abs_rates = np.asarray(
        [np.max(np.abs(nu)) for nu in nu_list],
        dtype=np.float64,
    )

    # Weighted population summaries
    weighted_terminal_inventory = float(
        np.dot(weights, terminal_inventory)
    )

    weighted_abs_terminal_inventory = float(
        np.dot(weights, np.abs(terminal_inventory))
    )

    weighted_mean_abs_rate = float(
        np.dot(weights, mean_abs_rates)
    )

    weighted_max_abs_rate = float(
        np.dot(weights, max_abs_rates)
    )

    # Filter-quality / latent-environment summaries
    latent_path = np.asarray(signals["latent_path"])
    F_t = np.asarray(signals["F_t"])
    pi_k = np.asarray(signals["pi_k"])
    A_hat_k = np.asarray(signals["A_hat_k"])

    result = {
        "ticker": ticker,
        "mid_price": float(mid_price),
        "sigma": float(sigma),

        "F_start": float(F_t[0]),
        "F_end": float(F_t[-1]),
        "fundamental_return": float((F_t[-1] - F_t[0]) / F_t[0]),

        "latent_state_1_fraction": float(np.mean(latent_path == 1)),

        "mean_abs_nu_bar": float(np.mean(np.abs(nu_bar))),
        "max_abs_nu_bar": float(np.max(np.abs(nu_bar))),
        "rms_nu_bar": float(np.sqrt(np.mean(nu_bar ** 2))),

        "weighted_mean_abs_agent_rate": weighted_mean_abs_rate,
        "weighted_max_abs_agent_rate": weighted_max_abs_rate,

        "weighted_terminal_inventory": weighted_terminal_inventory,
        "weighted_abs_terminal_inventory": weighted_abs_terminal_inventory,

        "initial_inventory_pop1": float(initial_inventory[0]),
        "terminal_inventory_pop1": float(terminal_inventory[0]),
        "mean_abs_rate_pop1": float(mean_abs_rates[0]),

        "initial_inventory_pop2": float(initial_inventory[1]),
        "terminal_inventory_pop2": float(terminal_inventory[1]),
        "mean_abs_rate_pop2": float(mean_abs_rates[1]),

        "mean_pi_pop1": float(np.mean(pi_k[0])),
        "mean_pi_pop2": float(np.mean(pi_k[1])),

        "mean_Ahat_pop1": float(np.mean(A_hat_k[0])),
        "mean_Ahat_pop2": float(np.mean(A_hat_k[1])),
    }

    return result


# ---------------------------------------------------------------------
# Full cross-sectional experiment
# ---------------------------------------------------------------------

def main():
    market_df = load_market_params()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"Running HJN cross-section for {len(market_df)} stocks...")
    print("=" * 80)

    for j, row in market_df.reset_index(drop=True).iterrows():
        ticker = str(row["ticker"]).upper()
        mid_price = float(row["mid_price"])
        sigma = float(row["sigma"])

        print(
            f"[{j + 1:02d}/{len(market_df):02d}] "
            f"{ticker:5s}  "
            f"S0={mid_price:8.2f}  "
            f"sigma={sigma:.2f}"
        )

        try:
            result = run_one_ticker(
                ticker=ticker,
                mid_price=mid_price,
                sigma=sigma,
                seed=SEED,
            )
            result["status"] = "ok"

            print(
                "    "
                f"|nu_bar| mean={result['mean_abs_nu_bar']:.6f}  "
                f"terminal q={result['weighted_terminal_inventory']:.6f}"
            )

        except Exception as exc:
            result = {
                "ticker": ticker,
                "mid_price": mid_price,
                "sigma": sigma,
                "status": "failed",
                "error": repr(exc),
            }

            print(f"    FAILED: {exc}")

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("=" * 80)
    print(f"Saved results to: {RESULTS_PATH}")

    successful = results_df[results_df["status"] == "ok"]

    print(
        f"Successful runs: {len(successful)}/{len(results_df)}"
    )

    if len(successful) > 0:
        print("\nCross-sectional summary sorted by sigma:\n")

        summary_cols = [
            "ticker",
            "sigma",
            "mean_abs_nu_bar",
            "weighted_mean_abs_agent_rate",
            "weighted_terminal_inventory",
        ]

        print(
            successful[
                [c for c in summary_cols if c in successful.columns]
            ]
            .sort_values("sigma")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
