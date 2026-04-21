# Regularized Reinforcement Learning: Policy Evaluation & Policy Gradient

This repository studies **regularized reinforcement learning (RL)** in continuous-time / discretized settings, with a focus on:

- **Policy Evaluation**: comparing data-driven vs. population solutions  
- **Policy Gradient**: understanding optimization dynamics under discretization and regularization  

The goal is to systematically analyze how **sampling, discretization (dt), and regularization (beta)** affect both evaluation and optimization.

---

## 📁 Repository Structure
├── policy evaluation/
├── policy gradient/

---

## 🔍 Policy Evaluation

This module studies **value function estimation** under a fixed policy.

We compare:

- **LSTD (data-driven)**  
- **PhiBE (population / expectation-based solution)**  

Key questions:
- How does LSTD converge to PhiBE?
- How do errors depend on sample size, dt, and beta?
- How does basis choice affect stability?

📂 Directory: `policy evaluation/`  
👉 See the README inside this folder for details.

---

## 🚀 Policy Gradient

This module studies **policy optimization** using natural policy gradient (NPG).

We analyze:

- Discrepancy between **data-driven updates** and **reference solutions**
- Error propagation through:
  - policy parameters
  - value function
  - advantage function
  - gradients

### Experiments

- `test_new.ipynb`  
  → Examine **error vs. time discretization (dt)**  

- `test-beta.ipynb`  
  → Examine **error vs. regularization parameter (beta)**  

📂 Directory: `policy gradient/`  
👉 See the README inside this folder for details.

---

## ⚙️ Core Components

- Data-driven simulation from **true dynamics**
- Closed-form / expectation-based benchmarks
- On-policy sampling
- Quadratic value function parameterization

---

## 📊 What We Learn

Across both modules, we study:

- Effect of **finite samples vs. population solutions**
- Role of **time discretization (dt)**
- Impact of **regularization (beta)**
- Interaction between **evaluation error and optimization error**

---

## 🛠️ How to Use

1. Clone the repository:
```bash
git clone https://github.com/baiting0522/Sample-Complexity-for-Policy-Gradient.git
Navigate to a module:
cd "policy evaluation"
# or
cd "policy gradient"
