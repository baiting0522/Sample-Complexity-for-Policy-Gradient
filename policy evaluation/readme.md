# Policy Evaluation for Regularized Reinforcement Learning

This repository studies **policy evaluation in regularized reinforcement learning**, with a focus on comparing the error between:

- **data-driven policy evaluation** using LSTD, and
- the **PhiBE solution**, which is computed from the corresponding expectation-based equations.

The main goal is to understand how the policy evaluation error depends on the choice of basis functions, the discretization step size, the regularization parameter, and the sample size.

## Overview

Given a parameterized policy, we evaluate the corresponding value function under regularized RL and compare:

1. **LSTD-based policy evaluation** from simulated data
2. **PhiBE-based policy evaluation** from the population-level Bellman equation

We study the gap between these two solutions under several basis parameterizations.

## Basis Choices

We consider the following value-function parameterizations.

### 1. Polynomial basis: `[1, s, s^2]`

This notebook uses the standard quadratic basis
\[
\phi(s) = [1, s, s^2].
\]

Notebook:  
[Regularized RL.ipynb](https://github.com/baiting0522/Sample-Complexity-for-Policy-Gradient/blob/main/policy%20evaluation/Regularized%20RL.ipynb)

### 2. Scaled basis: `[1/beta, s, s^2]`

This notebook uses a beta-scaled basis
\[
\phi(s) = [1/\beta, s, s^2].
\]

Notebook:  
[Regularied RL_beta.ipynb](https://github.com/baiting0522/Sample-Complexity-for-Policy-Gradient/blob/main/policy%20evaluation/Regularied%20RL_beta.ipynb)

### 3. Scalar quadratic basis: single-coefficient representation

This notebook uses the quadratic value function form
$$
V^\omega(s) = \frac{1}{2} k_2^\omega s^2 + k_1^\omega s + k_0^\omega,
$$
where the coefficients are not treated as independent free parameters. In particular, once \(k_2^\omega\) is determined, the remaining coefficients $k_1^\omega$ and $k_0^\omega$ can be expressed accordingly. Therefore, the value function is effectively characterized by a single scalar coefficient.

Notebook:  
[Regularized RL_scalar basis.ipynb](https://github.com/baiting0522/Sample-Complexity-for-Policy-Gradient/blob/main/policy%20evaluation/Regularized%20RL_scalar%20basis.ipynb)

## What Is Compared

For a fixed parameterized policy, we compare:

- the **data-driven LSTD estimator** of the value function parameters, and
- the **PhiBE solution** obtained from the expectation form of the policy evaluation equation.

Typical quantities of interest include:

- parameter error
- value-function error
- error versus sample size
- error versus time step `dt`
- error versus regularization parameter `beta`
