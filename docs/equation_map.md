# Equation Map

This document maps the core mathematical objects in the paper to code-level components in the repository.

Its purpose is to make the implementation reproducible and to clarify which equations are simulated directly, which are approximated, and which are simplified during the replication stage.

---

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

---

## 2. Unimpacted Price Process

**Equation**
$`
dF_t = A_t\,dt + \sigma dW_t
`$

**Interpretation**  
Fundamental price driven by latent drift and exogenous noise, without price impact.

**Code location**  
`src/mfg_replication/simulation.py`

**Function**  
`simulate_fundamental_price(latent_path, params)`

**Notes**  
Discrete-time Euler approximation used.

---

## 3. Mean Field Trading Rate

**Equation**
$`
\bar{\nu}_t = \frac{1}{N} \sum_{i=1}^N \nu_t^i
`$

**Interpretation**  
Average trading rate across all agents.

**Code location**  
`src/mfg_replication/simulation.py`

**Function**  
`compute_mean_flow(trade_rates)`

---

## 4. Impacted Midprice

**Equation**
`$
dS_t = (A_t + \lambda \bar{\nu}_t)\,dt + \sigma dW_t
$`

**Interpretation**  
Observed midprice including both latent drift and price impact from aggregate order flow.

**Code location**  
`src/mfg_replication/simulation.py`

**Function**  
`update_price(current_price, mean_flow, latent_signal, noise, params)`

---

## 5. Inventory Dynamics

**Equation**
`$
dQ_t^i = \nu_t^i dt
$`

**Interpretation**  
Agent inventory evolves according to its trading rate.

**Code location**  
`src/mfg_replication/simulation.py`

**Function**  
`update_inventory(current_inventory, trade_rate, dt)`

---

## 6. Trading Control (Approximation)

**Equation**
`$
\nu_t^i = -\beta_k Q_t^i - \zeta_k \bar{Q}_t^k + \gamma_k \hat{A}_t
$`

**Interpretation**  
Feedback control combining:
- liquidation pressure  
- interaction with other agents  
- response to filtered signal  

**Code location**  
`src/mfg_replication/control.py`

**Function**  
`compute_trade_rate(...)`

**Notes**  
Approximation to full MFG optimal strategy used for initial replication.

---

## 7. Posterior Update (Filtering)

**Equation**
`$
\pi_{t+\Delta t}(m) \propto 
\mathbb{P}(\Delta S_t \mid \Theta_t = m)\,
\sum_{\ell} P_{\ell m}\pi_t(\ell)
$`

**Interpretation**  
Bayesian update of latent-state belief based on observed price changes.

**Code location**  
`src/mfg_replication/filtering.py`

**Function**  
`update_posterior(...)`

**Notes**  
Each subpopulation maintains its own posterior.

---

## 8. Filtered Signal

**Equation**
`$
\hat{A}_t = \mathbb{E}[A_t \mid \mathcal{F}_t^S]
$`

**Interpretation**  
Filtered estimate of latent drift used in trading decisions.

**Code location**  
`src/mfg_replication/filtering.py`

**Function**  
`compute_filtered_signal(...)`
