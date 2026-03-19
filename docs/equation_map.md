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
