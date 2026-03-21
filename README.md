# Algorithmic Trading via Mean Field Games

## Objective

This project replicates and extends the results of:

> *Algorithmic Trading in Competitive Markets with Mean Field Games*  
> Philippe Casgrain & Sebastian Jaimungal

The focus is on simulating **heterogeneous agents** trading under:
- latent market regimes
- endogenous price impact
- mean-field interactions

After replication, the project explores:
- model misspecification
- finite-population effects
- sensitivity to impact parameters

---

## Model Components

### Latent Market
- Hidden Markov model driving drift
- Regime switching between bullish/bearish states

### Price Dynamics
- Fundamental price:
  \[
  dF_t = A_t dt + \sigma dW_t
  \]

- Impacted price:
  \[
  S_t = F_t + \lambda \int_0^t \bar{\nu}_s ds
  \]

### Agent Behavior
- Continuous trading rate \( \nu_t \)
- Depends on:
  - filtered drift estimate
  - inventory level
  - risk aversion (\(\kappa\))

### Objective
Agents minimize:
- inventory risk
- execution cost
- deviation from optimal liquidation

### Equilibrium
- Mean-field fixed point via aggregate order flow

---

## 🔁 Replication Targets

### 1. Inventory Dynamics
- Heterogeneous urgency levels
- Subpopulation vs aggregate inventory

### 2. Price Decomposition
- Fundamental vs impacted price
- Effect of aggregate order flow

### 3. Posterior Filtering
- Estimation of latent regime
- Belief differences across agents

---


## Setup

```bash
git clone
cd

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
