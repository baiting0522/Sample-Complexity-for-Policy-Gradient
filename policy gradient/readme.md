# Policy Gradient for Regularized Reinforcement Learning

This repository studies **policy gradient methods in regularized reinforcement learning**, with a focus on understanding how **discretization and regularization** affect the learning dynamics and estimation error.

In particular, we analyze how the learned policy parameters deviate from their reference (or population-level) counterparts under different settings.

---

## 🔍 Problem Setup

We consider a parameterized stochastic policy of the form:

$\pi_\omega(a \mid s) = \mathcal{N}(a \mid \omega_1 s + \omega_2, \omega_3^2)$,

where $\omega = (\omega_1, \omega_2, \omega_3)$ are the policy parameters.

The goal is to optimize the policy using **natural policy gradient (NPG)** under a regularized RL framework.

---

## ⚖️ What We Study

We focus on the discrepancy between:

- **Data-driven policy gradient updates** (finite sample, discrete-time)
- **PhiBE solutions**

We measure:

- Policy parameter error  
- Value function error  
- Advantage function error  
- Gradient error  

and study how these errors depend on:
- time discretization \(dt\)
- regularization parameter \(\beta\)
- number of iterations

---

## 🧪 Experiments

### 1. Error vs Time Discretization \(dt\)

We fix the regularization parameter and sample size, and vary \(dt\) to examine how discretization affects learning.

Notebook:  
`test_new.ipynb`

Key goal:
- Understand how error scales with \(dt\)
- Identify whether smaller \(dt\) improves approximation or introduces instability

---

### 2. Error vs Regularization Parameter $\beta$

We fix $dt$ and sample size, and vary $\beta$ to study the effect of entropy regularization.

Notebook:  
`test-beta.ipynb`

Key goal:
- Understand how \(\beta\) affects policy and value estimation
- Analyze scaling and conditioning effects

---

## 📁 Repository Structure

- `data_driven_on_policy_natural.py`  
  Implements the data-driven natural policy gradient algorithm.

- `on_policy_func.py`  
  Contains core functions for policy evaluation, advantage computation, and dynamics.

- `utils.py`  
  Utility functions for sampling, error computation, and helper operations.

- `test_new.ipynb`  
  Experiments for **error vs \(dt\)**.

- `test-beta.ipynb`  
  Experiments for **error vs \(\beta\)**.

- `readme.md`  
  This file.

---
