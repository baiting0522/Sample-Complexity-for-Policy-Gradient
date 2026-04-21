import numpy as np
from typing import Tuple, Callable


# def sample_data( n_traj, T, dt, omega_val,A, B, sigma,) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Sample on-policy trajectory data.
    
#     Returns:
#         s_all: State trajectories (n_traj, n_steps+1)
#         a_all: Action trajectories (n_traj, n_steps+1)
#     """
#     n_steps = int(T / dt)
#     delta_t = dt
#     t_grid = np.linspace(0, T, n_steps + 1)
#     exp_A = np.exp(A * t_grid)
    
#     s_all = np.zeros((n_traj, n_steps + 1))
#     a_all = np.zeros((n_traj, n_steps + 1))
    
#     for k in range(n_traj):
#         # Random initial state
#         s_0 = 2 * np.random.rand() - 1
        
#         s = np.zeros(n_steps + 1)
#         a = np.zeros(n_steps + 1)
        
#         s[0] = s_0
#         a[0] = omega_val[0] * s[0] + omega_val[1] + omega_val[2] * np.random.randn()
        
#         # Generate trajectory
#         for i in range(1, n_steps + 1):
#             deterministic = (B * a[i-1] / A) * (exp_A[i] - 1) + s_0 * exp_A[i]
#             stochastic = np.sqrt(sigma**2 / (2*A) * (np.exp(2*A*delta_t) - 1)) * np.random.randn()
            
#             s[i] = deterministic + stochastic
#             a[i] = omega_val[0] * s[i] + omega_val[1] + omega_val[2] * np.random.randn()
        
#         s_all[k, :] = s
#         a_all[k, :] = a
    
#     return s_all, a_all

def sample_data(
    n_traj: int,
    T: float,
    dt: float,
    omega_val: np.ndarray,
    A: float,
    B: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample on-policy trajectory data using the exact discretization of

        dS_t = (A S_t + B a_t) dt + sigma dW_t,

    where the action is held constant on each interval [t_i, t_{i+1}].

    Policy:
        a_i = omega_1 * s_i + omega_2 + omega_3 * epsilon_i,
        epsilon_i ~ N(0, 1)

    Args:
        n_traj: number of trajectories
        T: total time horizon
        dt: time step
        omega_val: policy parameter array of shape (3,)
        A, B, sigma: system parameters
        seed: random seed

    Returns:
        s_all: shape (n_traj, n_steps + 1)
        a_all: shape (n_traj, n_steps + 1)
    """
    rng = np.random.default_rng(42)

    n_steps = int(T / dt)
    if not np.isclose(n_steps * dt, T):
        raise ValueError("T must be divisible by dt.")

    omega1, omega2, omega3 = omega_val

    s_all = np.zeros((n_traj, n_steps + 1))
    a_all = np.zeros((n_traj, n_steps + 1))

    # Exact one-step coefficients
    exp_Adt = np.exp(A * dt)

    if abs(A) < 1e-12:
        # limit as A -> 0
        coef_a = B * dt
        noise_var = sigma**2 * dt
    else:
        coef_a = B * (exp_Adt - 1.0) / A
        noise_var = sigma**2 * (np.exp(2.0 * A * dt) - 1.0) / (2.0 * A)

    noise_std = np.sqrt(max(noise_var, 0.0))

    for k in range(n_traj):
        # Initial state sampled uniformly from [-1, 1]
        #s0 = 2.0 * rng.random() - 1.0
        s0 = rng.normal(loc=0.0, scale=1.0)

        s = np.zeros(n_steps + 1)
        a = np.zeros(n_steps + 1)

        s[0] = s0
        a[0] = omega1 * s[0] + omega2 + omega3 * rng.normal()

        for i in range(n_steps):
            # Exact state transition from s[i] to s[i+1]
            noise = noise_std * rng.normal()
            s[i + 1] = exp_Adt * s[i] + coef_a * a[i] + noise

            # Sample next action from current policy
            a[i + 1] = omega1 * s[i + 1] + omega2 + omega3 * rng.normal()

        s_all[k, :] = s
        a_all[k, :] = a

    return s_all, a_all


def compute_theta( omega_val, delta_t, s_all, a_all, basis_length, f_on_policy, r_on_policy,) -> np.ndarray:
    """
    Compute theta parameters by solving the equation.
    """
    n_traj = s_all.shape[0]
    n_steps = s_all.shape[1] - 1
    
    A_f = np.zeros((basis_length, basis_length))
    b = np.zeros(basis_length)
    
    for i in range(1, n_steps + 1):
        for traj in range(n_traj):
            s_t2 = s_all[traj, i]
            s_t1 = s_all[traj, i-1]
            a_val = a_all[traj, i-1]
            
            A_f += f_on_policy(s_t2, s_t1, a_val, omega_val[0], omega_val[1], omega_val[2], delta_t)
            b += r_on_policy(s_t2, s_t1, a_val, omega_val[0], omega_val[1], omega_val[2], delta_t)
    
    # Solve the linear system
    theta = np.linalg.solve(A_f.T, -b)
    return theta


def compute_F_inv(
    s_all: np.ndarray,
    a_all: np.ndarray,
    omega_val: np.ndarray
) -> np.ndarray:
    """
    Compute the inverse of the Fisher information matrix.
    
    Args:
        s_all: State samples
        a_all: Action samples
        omega_val: Policy parameters [omega1, omega2, omega3]
    
    Returns:
        F_inv: Inverse Fisher information matrix (3x3)
    """
    omega1, omega2, omega3 = omega_val[0], omega_val[1], omega_val[2]
    
    s_all = s_all.flatten()
    a_all = a_all.flatten()
    mean_a = omega1 * s_all + omega2
    
    F = np.zeros((3, 3))
    
    # Compute Fisher information matrix elements
    F[0, 0] = np.mean((1 / omega3**4) * s_all**2 * (a_all - mean_a)**2)
    F[0, 1] = np.mean((1 / omega3**4) * s_all * (a_all - mean_a)**2)
    F[0, 2] = np.mean((-1 / omega3**3) * s_all * (a_all - mean_a) + 
                      (1 / omega3**5) * s_all * (a_all - mean_a)**3)
    
    F[1, 0] = F[0, 1]
    F[1, 1] = np.mean((1 / omega3**4) * (a_all - mean_a)**2)
    F[1, 2] = np.mean((-1 / omega3**3) * (a_all - mean_a) + 
                      (1 / omega3**5) * (a_all - mean_a)**3)
    
    F[2, 0] = F[0, 2]
    F[2, 1] = F[1, 2]
    F[2, 2] = np.mean((1 / omega3**2) + (1 / omega3**6) * (a_all - mean_a)**4 - 
                      (2 / omega3**4) * (a_all - mean_a)**2)
    
    F_inv = np.linalg.inv(F)
    return F_inv


def compute_grad(
    theta: np.ndarray,
    s_all: np.ndarray,
    a_all: np.ndarray,
    omega_iter: np.ndarray,
    q_func: Callable,
    beta: float,
    lambda_reg: float
) -> np.ndarray:
    """
    Compute policy gradient (on-policy).

    Args:
        theta: Value function parameters
        s_all: State samples
        a_all: Action samples
        omega_iter: Current policy parameters
        q_func: Advantage function
        beta: Discount factor
        lambda_reg: Regularization parameter

    Returns:
        grad: Policy gradient (3,)
    """
    omega1, omega2, omega3 = omega_iter[0], omega_iter[1], omega_iter[2]

    s_all = s_all.flatten()
    a_all = a_all.flatten()
    n_samples = len(s_all)

    # Compute Q values
    q_vals = np.array([q_func(s_all[i], a_all[i], theta, omega1, omega2, omega3)
                       for i in range(n_samples)])

    mu_a = omega1 * s_all + omega2

    # Gradient of log policy
    dlogpi1 = (a_all - mu_a) * s_all / omega3**2
    dlogpi2 = (a_all - mu_a) / omega3**2
    dlogpi3 = ((a_all - mu_a)**2 - omega3**2) / omega3**3

    # Weighted gradient
    weights = -lambda_reg + q_vals

    grad = np.array([
        np.mean(weights * dlogpi1),
        np.mean(weights * dlogpi2),
        np.mean(weights * dlogpi3)
    ])

    return grad


def compute_grad_off_policy(
    theta: np.ndarray,
    s_all: np.ndarray,
    a_all: np.ndarray,
    omega_iter: np.ndarray,
    q_func: Callable,
    beta: float,
    lambda_reg: float
) -> np.ndarray:
    """
    Compute policy gradient using off-policy data with importance sampling.

    Assumes behavior policy is uniform on [-1, 1].

    Args:
        theta: Value function parameters
        s_all: State samples (from behavior policy)
        a_all: Action samples (from behavior policy)
        omega_iter: Current policy parameters
        q_func: Advantage function
        beta: Discount factor
        lambda_reg: Regularization parameter

    Returns:
        grad: Policy gradient (3,)
    """
    omega1, omega2, omega3 = omega_iter[0], omega_iter[1], omega_iter[2]

    s_all = s_all.flatten()
    a_all = a_all.flatten()
    n_samples = len(s_all)

    # Compute Q values (advantage function)
    q_vals = np.array([q_func(s_all[i], a_all[i], theta, omega1, omega2, omega3)
                       for i in range(n_samples)])

    mu_a = omega1 * s_all + omega2

    # Target policy: π^ω(a|s) = N(a; ω1*s + ω2, ω3^2)
    target_policy_prob = (1 / (np.sqrt(2 * np.pi) * omega3)) * \
                        np.exp(-(a_all - mu_a)**2 / (2 * omega3**2))

    # Behavior policy: uniform on [-1, 1]
    behavior_policy_prob = 0.5

    # Importance sampling weights: w = π(a|s) / μ(a)
    importance_weights = target_policy_prob / behavior_policy_prob

    # Gradient of log policy
    dlogpi1 = (a_all - mu_a) * s_all / omega3**2
    dlogpi2 = (a_all - mu_a) / omega3**2
    dlogpi3 = ((a_all - mu_a)**2 - omega3**2) / omega3**3

    # Weighted gradient with importance sampling
    weights = importance_weights * (-lambda_reg + q_vals)

    grad = np.array([
        np.mean(weights * dlogpi1),
        np.mean(weights * dlogpi2),
        np.mean(weights * dlogpi3)
    ])

    return grad

def compute_grad_hybrid(
    theta: np.ndarray,
    omega_iter: np.ndarray,
    q_func: Callable,
    beta: float,
    lambda_reg: float,
    n_samples: int
) -> np.ndarray:
    """
    Compute policy gradient using freshly sampled on-policy data.

    This is a hybrid approach where:
    - theta is computed from off-policy data (passed in)
    - gradient is computed from fresh on-policy single-step samples

    Args:
        theta: Value function parameters (computed from off-policy data)
        omega_iter: Current policy parameters
        q_func: Advantage function
        beta: Discount factor
        lambda_reg: Regularization parameter
        n_samples: Number of (s, a) samples to draw

    Returns:
        grad: Policy gradient (3,)
    """
    omega1, omega2, omega3 = omega_iter[0], omega_iter[1], omega_iter[2]

    # Sample fresh on-policy data (single timestep samples, not trajectories)
    # We only need (s, a) pairs for gradient computation
    s_samples = 2 * np.random.rand(n_samples) - 1  # Random states from [-1, 1]
    a_samples = omega1 * s_samples + omega2 + omega3 * np.random.randn(n_samples)  # Sample actions from π(a|s)

    # Compute Q values using the theta from off-policy data
    q_vals = np.array([q_func(s_samples[i], a_samples[i], theta, omega1, omega2, omega3)
                       for i in range(n_samples)])

    mu_a = omega1 * s_samples + omega2

    # Gradient of log policy
    dlogpi1 = (a_samples - mu_a) * s_samples / omega3**2
    dlogpi2 = (a_samples - mu_a) / omega3**2
    dlogpi3 = ((a_samples - mu_a)**2 - omega3**2) / omega3**3

    # Weighted gradient
    weights = -lambda_reg + q_vals

    grad = np.array([
        np.mean(weights * dlogpi1),
        np.mean(weights * dlogpi2),
        np.mean(weights * dlogpi3)
    ])

    return grad


def compute_grad_analytical_off_policy(theta: np.ndarray,
                                       omega: np.ndarray) -> np.ndarray:
    """
    Compute policy gradient analytically:
    G_omega = first_term - second_term
            = first_term - 0
            = first_term
    
    where first_term = integral of phi(s,a) nabla_omega pi(a|s) da rho(s) ds
    """
    omega1, omega2, omega3 = omega[0], omega[1], omega[2]
    
    # First term only (second term = 0)
    phi_omega1 = np.array([0, 0, 0, 1/3, 0, 0])
    phi_omega2 = np.array([0, 0, 0, 0, 1, 2*omega2])
    phi_omega3 = np.array([0, 0, 0, 0, 0, 2*omega3])
    
    grad = np.array([
        phi_omega1 @ theta,
        phi_omega2 @ theta,
        phi_omega3 @ theta
    ])
    
    return grad


def mean_nonzero_real_3d(A: np.ndarray) -> np.ndarray:
    """
    Compute mean of non-zero real values across the third dimension.
    
    Args:
        A: 3D array (m, n, T)
    
    Returns:
        mean_oh: 2D array of means (m, n)
    """
    m, n, T = A.shape
    mean_oh = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            vals = np.real(A[i, j, :])
            vals = vals[vals != 0]
            if len(vals) == 0:
                mean_oh[i, j] = 0
            else:
                mean_oh[i, j] = np.mean(vals)
    
    return mean_oh