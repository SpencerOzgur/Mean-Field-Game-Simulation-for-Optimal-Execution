# Parameters

| Parameter | Meaning | Value | Source |
|---------- |-------- |------ |--------|
| T        | Time horizon | 1.0 | Paper |
| dt       | Time step size | TBD (e.g. 0.001) | Chosen |
| σ        | Volatility of price noise | TBD | Inferred |
| λ        | Permanent price impact coefficient | TBD | Inferred |
| P        | Transition matrix of latent Markov chain | TBD | Assumed |
| N₁, N₂   | Number of agents in each subpopulation | TBD (e.g. 20, 20) | Assumed |
| ψ₁, ψ₂   | Running inventory penalty (holding cost) | TBD | Assumed |
| φ₁, φ₂   | Terminal inventory penalty (liquidation cost) | TBD | Assumed |
| β₁, β₂   | Own-inventory feedback strength | TBD | Assumed |
| ζ₁, ζ₂   | Mean-field coupling strength | TBD | Assumed |
| γ₁, γ₂   | Sensitivity to filtered signal (alpha) | TBD | Assumed |
| θ₁, θ₂   | Latent state values (drift / mean-reversion levels) | ~4.95, 5.05 | From Figure |
| π₀¹, π₀² | Initial beliefs (priors) of each subpopulation | TBD (different across groups) | Inferred |
