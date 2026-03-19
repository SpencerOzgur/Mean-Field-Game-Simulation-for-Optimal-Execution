# Equation Map

This document maps the core mathematical objects in the paper to code-level components in the repository.

Its purpose is to make the implementation reproducible and to clarify which equations are simulated directly, which are approximated, and which are simplified during the replication stage.

## 1. Latent State Process

**Equation**
$`
\Theta_{t+\Delta t} \sim P(\Theta_t, \cdot)`$


**Interpretation**  
Latent Markov chain representing the hidden market regime.

**Code location**  
`src/mfg_replication/latent.py`

**Function**  
`simulate_latent_path(params)`

**Notes**  
Two-state chain used in replication.
