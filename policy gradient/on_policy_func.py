import numpy as np

EPS_STD = 1e-3

def gaussian_log_prob(a: float, mean: float, std: float) -> float:
    """
    Compute log probability of action under Gaussian policy.

    Args:
        a: Action value
        mean: Mean of Gaussian (μ(s))
        std: Standard deviation (σ)

    Returns:
        log π(a|s) for Gaussian distribution
    """
    return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((a - mean) / std) ** 2


def create_basis_functions_for_V() -> dict:
    """
    Create basis functions for Q-function approximation.
    
    Returns:
        Dictionary containing basis function information
    """
    def phi(s: float) -> np.ndarray:
        """Polynomial basis functions"""
        return np.array([1, s, s**2])
    
    def grad_phi_s(s: float) -> np.ndarray:
        """Gradient of basis functions w.r.t. state"""
        return np.array([0, 1, 2*s])
    
    def hess_phi_s(s: float) -> np.ndarray:
        """Hessian of basis functions w.r.t. state"""
        return np.array([0, 0, 2])
    
    return {
        'phi': phi,
        'grad_phi_s': grad_phi_s,
        'hess_phi_s': hess_phi_s,
        'length': 3
    }

def create_A_matrix(beta: float):
    """
    Create the matrix for computing transition dynamics contribution.
    """
    def A_func(s_t2: float, s_t1: float, delta_t: float) -> np.ndarray:
        """
        Compute transition contribution to Bellman equation.
        """
        basis = np.array([1, s_t1, s_t1**2])
        grad_basis = np.array([0, 1, 2*s_t1])
        hess_basis = np.array([0, 0, 2])
        
        # term = ((s_t2 - s_t1) / delta_t) * grad_basis + \
        #        ((s_t2 - s_t1)**2 / (2 * delta_t)) * hess_basis + \
        #        (1 - beta) * basis - basis

        term = ((s_t2 - s_t1) / delta_t) * grad_basis + \
               (1 - beta) * basis - basis
        
        return np.outer(term, basis)
    
    return A_func

def create_b_vector(M: float, N: float, lambda_reg: float) :
    """
    Create vector b.
    
    Args:
        M: State cost weight
        N: Action cost weight
        lambda_reg: Regularization parameter
    """
    def b_func(s_t2: float, s_t1: float, a: float,
               omega1: float, omega2: float, omega3: float, delta_t: float) -> np.ndarray:
        """
        Compute reward contribution.
        """
        basis = np.array([1, s_t1, s_t1**2])

        # Policy: π(a|s) = N(a; μ(s), σ²) where μ(s) = omega1*s + omega2, σ = omega3
        mean = omega1 * s_t1 + omega2
        std = omega3
        log_pi = gaussian_log_prob(a, mean, std)

        # Reward: state cost + action cost + entropy regularization
        reward = (-M/2) * s_t1**2 + (-N/2) * a**2 - lambda_reg * log_pi

        return reward * basis
    
    return b_func

def create_b_vector_phibe(M: float, N: float, lambda_reg: float):
    """
    Create expectation version of b:
    E[r(s,a) | s] * phi(s),
    where a|s ~ N(omega1*s + omega2, omega3^2)
    """
    def b_func_phibe(s_t1: float,
                     omega1: float, omega2: float, omega3: float) -> np.ndarray:
        basis = np.array([1.0, s_t1, s_t1**2])

        mu = omega1 * s_t1 + omega2
        sigma = omega3

        expected_reward = (
            -0.5 * M * s_t1**2
            -0.5 * N * (mu**2 + sigma**2)
            + 0.5 * lambda_reg * np.log(2.0 * np.pi * sigma**2)
            + 0.5 * lambda_reg
        )

        return expected_reward * basis

    return b_func_phibe
    
def compute_grad_on_policy(theta, s_all, a_all, omega, beta, lambda_reg, M, N, dt):
    """
    Compute on-policy update gradient.

    Args:
        theta: Value function parameters
        s_all: State trajectories (n_traj, n_steps+1)
        a_all: Action trajectories (n_traj, n_steps+1)
        omega: Current policy parameters
        beta: Discount factor
        lambda_reg: Regularization parameter
        M: State cost weight
        N: Action cost weight
        dt: Time step

    Returns:
        grad: Policy gradient (3,)
    """
    omega1, omega2, omega3 = omega[0], omega[1], omega[2]

    n_traj, n_steps_plus_1 = s_all.shape

    # Extract transitions: use all but last time step for s_t1, a_t1
    # and all but first time step for s_t2
    s_t1 = s_all[:, :-1]  # (n_traj, n_steps)
    s_t2 = s_all[:, 1:]   # (n_traj, n_steps)
    a_t1 = a_all[:, :-1]  # (n_traj, n_steps)

    # Flatten for vectorized computation
    s_t1_flat = s_t1.flatten()
    s_t2_flat = s_t2.flatten()
    a_t1_flat = a_t1.flatten()
    n_samples = len(s_t1_flat)

    # Compute basis functions for s_t1
    basis = np.array([np.ones(n_samples), s_t1_flat, s_t1_flat**2])  # (3, n_samples)

    # Compute value function V(s_t1) = theta^T * phi(s_t1)
    V = theta @ basis  # (n_samples,)

    # Compute policy mean
    mu_a = omega1 * s_t1_flat + omega2

    # Compute log probabilities
    log_pi = gaussian_log_prob(a_t1_flat, mu_a, omega3)

    # Compute rewards: r(s,a) = -M/2 * s^2 - N/2 * a^2 - lambda_reg * log π(a|s)
    rewards = (-M/2) * s_t1_flat**2 + (-N/2) * a_t1_flat**2 - lambda_reg * log_pi

    # Compute HJB operator term from A matrix approximation
    # operator = grad_V * (s_t2 - s_t1)/dt + 0.5 * hess_V * (s_t2 - s_t1)^2/dt
    # For basis [1, s, s^2]: grad_phi = [0, 1, 2*s], hess_phi = [0, 0, 2]
    grad_basis = np.array([np.zeros(n_samples), np.ones(n_samples), 2*s_t1_flat])  # (3, n_samples)
    hess_basis = np.array([np.zeros(n_samples), np.zeros(n_samples), 2*np.ones(n_samples)])  # (3, n_samples)

    # Gradient and Hessian of V
    grad_V = theta @ grad_basis  # scalar for each sample
    hess_V = theta @ hess_basis  # scalar for each sample

    # State transition terms
    delta_s = s_t2_flat - s_t1_flat

    # Operator term (HJB equation approximation)
    # operator = grad_V * delta_s / dt + 0.5 * hess_V * (delta_s**2) / dt
    operator = grad_V * delta_s / dt

    # Compute advantage: A(s,a) = r(s,a) + operator - beta * V(s)
    advantage = rewards + operator - beta * V

    # Gradient of log policy w.r.t. omega parameters
    dlogpi1 = (a_t1_flat - mu_a) * s_t1_flat / omega3**2
    dlogpi2 = (a_t1_flat - mu_a) / omega3*2
    dlogpi3 = ((a_t1_flat - mu_a)**2 - omega3**2) / omega3**3

    weights = advantage

    grad = np.array([
        np.mean(weights * dlogpi1),
        np.mean(weights * dlogpi2),
        np.mean(weights * dlogpi3)
    ])

    return grad


def compute_theta_on_policy(omega_val, delta_t, s_all, a_all, basis_length, A_func, b_func):
    """
    Compute theta parameters for on-policy value function.

    Args:
        omega_val: Policy parameters [omega1, omega2, omega3]
        delta_t: Time step
        s_all: State trajectories (n_traj, n_steps+1)
        a_all: Action trajectories (n_traj, n_steps+1)
        basis_length: Length of basis functions (should be 3)
        A_func: Function from create_A_matrix (takes s_t2, s_t1, delta_t)
        b_func: Function from create_b_vector (takes s_t2, s_t1, a, omega1, omega2, omega3, delta_t)

    Returns:
        theta: Value function parameters (basis_length,)
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

            # A_func takes (s_t2, s_t1, delta_t)
            A_f += A_func(s_t2, s_t1, delta_t)
            # b_func takes (s_t2, s_t1, a, omega1, omega2, omega3, delta_t)
            b += b_func(s_t2, s_t1, a_val, omega_val[0], omega_val[1], omega_val[2], delta_t)

    # Solve the linear system
    theta = np.linalg.solve(A_f.T, -b)

    s_curr = s_all[:, :-1].reshape(-1)
    A_t = create_A_matrix_phibe(1, -1, 1, omega_val, s_curr, delta_t)
    A_diff = np.linalg.norm(A_t - A_f/(n_traj * n_steps))

    # PhiBE b
    b_phibe_func = create_b_vector_phibe(1, 1, 0.1)
    b_t = np.zeros(basis_length)
    for s in s_curr:
        b_t += b_phibe_func(s, omega_val[0], omega_val[1], omega_val[2])
    b_t /= len(s_curr)

    b_diff = np.linalg.norm(b_t - b/(n_traj * n_steps))
    
    return theta, A_diff, b_diff

def create_A_matrix_phibe(beta, A, B, omega, s_all, dt):
    w1, w2, w3 = omega
    A_dt = (np.exp(A * dt) - 1.0) / dt

    if abs(A) < 1e-12:
        B_dt = B
        #sigma_t2 = sigma**2
    else:
        B_dt = B * (np.exp(A * dt) - 1.0) / (A * dt)
        #sigma_t2 = 0.5**2 * (np.exp(2.0 * A * dt) - 1.0) / (2.0 * A * dt)

    A_phibe = np.zeros((3, 3))

    s_vals = np.asarray(s_all).reshape(-1)

    for s in s_vals:
        basis = np.array([1.0, s, s**2])
        grad_basis = np.array([0.0, 1.0, 2.0 * s])

        drift_dt = (A_dt + B_dt * w1) * s + B_dt * w2
        term = drift_dt * grad_basis - beta * basis

        A_phibe += np.outer(term, basis)

    A_phibe /= len(s_vals)
    return A_phibe
    
def compute_theta_phibe(omega, beta, lambda_reg, M, N, dt, A, B):
    """
    Compute PhiBE/population value coefficients theta for the current policy omega,
    without using sampled transitions.

    Value parametrization:
        V(s) = theta[0] + theta[1] * s + theta[2] * s^2

    Policy:
        a ~ N(mu(s), omega3^2),  mu(s) = omega1 * s + omega2

    PhiBE drift:
        b_hat(s,a) = A_t s + B_t a
    """
    omega1, omega2, omega3 = omega

    # discretized PhiBE coefficients
    if abs(A) < 1e-12:
        A_t = A
        B_t = B
    else:
        A_t = (np.exp(A * dt) - 1.0) / dt
        B_t = B * (np.exp(A * dt) - 1.0) / (A * dt)

    # effective closed-loop drift coefficient under policy mean
    alpha = A_t + B_t * omega1

    # E[a | s] = omega1 s + omega2
    # E[a^2 | s] = omega3^2 + (omega1 s + omega2)^2
    #
    # reward:
    #   r(s,a) = -M/2 s^2 - N/2 a^2 - lambda_reg log pi(a|s)
    #
    # Under a ~ pi(.|s):
    #   E[-lambda_reg log pi(a|s)] = -lambda_reg * E[log pi(a|s)]
    # For Gaussian N(mu, sigma^2):
    #   E[log pi(a|s)] = -0.5 * log(2*pi*omega3^2) - 0.5
    #
    # so the expected regularization contribution is:
    reg_const = 0.5 * lambda_reg * (np.log(2.0 * np.pi * omega3**2) + 1.0)

    # Match coefficients in the PhiBE Bellman equation:
    #
    # beta V(s) = E[r(s,a)|s] + V_s(s) * E[b_hat(s,a)|s]
    #
    # with V(s) = theta0 + theta1 s + theta2 s^2
    # and V_s(s) = theta1 + 2 theta2 s

    # Quadratic coefficient
    denom2 = beta - 2.0 * alpha
    numer2 = 0.5 * M + 0.5 * N * omega1**2
    theta2 = - numer2 / denom2

    # Linear coefficient
    denom1 = beta - alpha
    numer1 = N * omega1 * omega2 - 2.0 * theta2 * B_t * omega2
    theta1 = - numer1 / denom1

    # Constant coefficient
    numer0 = 0.5 * N * (omega3**2 + omega2**2) - reg_const - theta1 * B_t * omega2
    theta0 = - numer0 / beta

    theta = np.array([theta0, theta1, theta2], dtype=float)
    return theta


def compute_advantage_phibe(theta, s, a, omega, beta, lambda_reg, M, N, dt, A, B):
    """
    Compute PhiBE advantage A^Phi(s,a) in closed form, using drift expectation
    instead of sampled transition increments.
    """
    omega1, omega2, omega3 = omega

    if abs(A) < 1e-12:
        A_t = A
        B_t = B
    else:
        A_t = (np.exp(A * dt) - 1.0) / dt
        B_t = B * (np.exp(A * dt) - 1.0) / (A * dt)

    # Value and its derivative
    V = theta[0] + theta[1] * s + theta[2] * s**2
    grad_V = theta[1] + 2.0 * theta[2] * s

    # Policy mean and log-prob
    mu_a = omega1 * s + omega2
    log_pi = gaussian_log_prob(a, mu_a, omega3)

    # Regularized reward
    rewards = (-M / 2.0) * s**2 + (-N / 2.0) * a**2 - lambda_reg * log_pi

    # PhiBE drift
    drift = A_t * s + B_t * a

    # Advantage
    advantage = rewards + grad_V * drift - beta * V
    return advantage


def compute_grad_phibe(theta, s_all, a_all, omega, beta, lambda_reg, M, N, dt, A, B):
    """
    Version A:
    - theta is PhiBE / population theta (no data-driven theta)
    - gradient still uses sampled (s,a) to approximate expectation
    - no sampled s' is used
    """
    omega1, omega2, omega3 = omega

    if abs(A) < 1e-12:
        A_t = A
        B_t = B
    else:
        A_t = (np.exp(A * dt) - 1.0) / dt
        B_t = B * (np.exp(A * dt) - 1.0) / (A * dt)

    mu_s = 0
    m2_s = 1.0

    # coefficient in C1(s) = alpha1 * s + alpha0
    alpha1 = 2.0 * B_t * theta[2] - N * omega1
    alpha0 = B_t * theta[1] - N * omega2
    grad1 = alpha1 * m2_s + alpha0 * mu_s
    grad2 = alpha1 * mu_s + alpha0
    grad3 = lambda_reg / omega3 - N * omega3

    grad = np.array([grad1, grad2, grad3])

    # s_t = s_all[:, :-1].flatten()
    # a_t = a_all[:, :-1].flatten()

    # PhiBE advantage
    # advantage = compute_advantage_phibe(
    #     theta=theta,
    #     s=s_t,
    #     a=a_t,
    #     omega=omega,
    #     beta=beta,
    #     lambda_reg=lambda_reg,
    #     M=M,
    #     N=N,
    #     dt=dt,
    #     A=A,
    #     B=B,
    # )

    # mu_a = omega1 * s_t + omega2

    # # score functions
    # dlogpi1 = (a_t - mu_a) * s_t / (omega3**2)
    # dlogpi2 = (a_t - mu_a) / (omega3**2)
    # dlogpi3 = ((a_t - mu_a)**2 - omega3**2) / (omega3**3)

    # grad = np.array([
    #     np.mean(advantage * dlogpi1),
    #     np.mean(advantage * dlogpi2),
    #     np.mean(advantage * dlogpi3),
    # ])

    return grad


def compute_grad_phibe_full(omega, s_all, a_all, beta, lambda_reg, M, N, dt, A, B):
    """
    Convenience wrapper:
    given omega, first compute theta_phibe(omega), then compute grad_phibe.
    """
    theta_phibe = compute_theta_phibe(
        omega=omega,
        beta=beta,
        lambda_reg=lambda_reg,
        M=M,
        N=N,
        dt=dt,
        A=A,
        B=B,
    )

    grad_phibe = compute_grad_phibe(
        theta=theta_phibe,
        s_all=s_all,
        a_all=a_all,
        omega=omega,
        beta=beta,
        lambda_reg=lambda_reg,
        M=M,
        N=N,
        dt=dt,
        A=A,
        B=B,
    )

    return theta_phibe, grad_phibe