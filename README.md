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
- **PhiBE**  

📂 Directory: `policy evaluation/`  
👉 See the README inside this folder for details.

---

## 🚀 Policy Gradient

This module studies **policy optimization** using natural policy gradient (NPG).

We analyze:

- Discrepancy between **data-driven updates** and **reference solutions**
- 
### Experiments

- `test_new.ipynb`  
  → Examine **error vs. time discretization (dt)**  

- `test-beta.ipynb`  
  → Examine **error vs. regularization parameter (beta)**  

📂 Directory: `policy gradient/`  
👉 See the README inside this folder for details.

---

