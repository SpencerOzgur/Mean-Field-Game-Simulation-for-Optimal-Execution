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
from params import latent_params, simulation_params


def posterior_metrics(pi: np.ndarray, latent_path: np.ndarray) -> dict:
    pred_state = (pi >= 0.5).astype(int)
    accuracy = np.mean(pred_state == latent_path)
    mae = np.mean(np.abs(pi - latent_path))
    mse = np.mean((pi - latent_path) ** 2)
    return {
        "accuracy": accuracy,
        "mae": mae,
        "mse": mse,
    }


def main():
    np.random.seed(42)

    priors = [
        [0.5, 0.5],
        [0.8, 0.2],
        [0.2, 0.8],
    ]

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    print("Filtering verification")
    print("-" * 60)

    for prior in priors:
        pi = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=simulation_params,
            prior=prior,
        )

        metrics = posterior_metrics(pi, latent_path)

        print(f"prior = {prior}")
        print(f"  accuracy = {metrics['accuracy']:.4f}")
        print(f"  mean abs error = {metrics['mae']:.4f}")
        print(f"  mean sq error = {metrics['mse']:.4f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()