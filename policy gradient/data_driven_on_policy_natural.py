"""
Natural Policy Gradient (on-policy)
"""
import numpy as np
from typing import Tuple, Optional
from utils import sample_data, compute_F_inv, mean_nonzero_real_3d
from on_policy_func import (
    create_A_matrix, create_b_vector,
    compute_grad_on_policy, compute_theta_on_policy,
    compute_advantage_phibe, compute_grad_phibe, compute_grad_phibe_full, compute_theta_phibe, create_A_matrix_phibe
)


class NaturalPG_on_policy:
    """Natural Policy Gradient algorithm for LQR problems."""

    def __init__(
        self,
        A: float = -1.0,
        B: float = 1.0,
        sigma: float = 1.0,
        M: float = 1.0,
        N: float = 1.0,
        lambda_reg: float = 0.1,
        beta: float = 0.1,
        learning_rate: float = 0.05,
        T: float = 0.1,
        dt: float = 0.1,
        n_traj: int = 300,
        window_size: int = 1,
        epsilon_avg: float = 1e-8,
        max_iter: int = 1500
    ):
        """
        Initialize Natural Policy Gradient.

        Args:
            A: Drift parameter
            B: Control coefficient
            sigma: Diffusion coefficient
            M: State cost weight
            N: Action cost weight
            lambda_reg: Regularization parameter
            beta: Discount factor
            learning_rate: Learning rate (eta)
            T: Total simulation time
            dt: Time step
            n_traj: Number of trajectories per iteration
            window_size: Window size for convergence check
            epsilon_avg: Convergence threshold
            max_iter: Maximum number of iterations
        """
        # System parameters
        self.A = A
        self.B = B
        self.sigma = sigma
        self.M = M
        self.N = N
        self.lambda_reg = lambda_reg
        self.beta = beta

        # Algorithm parameters
        self.eta = learning_rate
        self.T = T
        self.dt = dt
        self.n_traj = n_traj
        self.window_size = window_size
        self.epsilon_avg = epsilon_avg
        self.max_iter = max_iter

        # Create functions (A and b replace f and r)
        self.A_matrix = create_A_matrix(beta)
        self.b_vector = create_b_vector(M, N, lambda_reg)

        # History
        self.omega_history = None
        self.grad_history = None

    def run_single_trial(
        self,
        omega_init: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a single trial of natural policy gradient.

        Args:
            omega_init: Initial policy parameters (default: random)
            verbose: Whether to print progress

        Returns:
            omega_val: Final policy parameters
            omega_history: History of policy parameters
            grad_history: History of natural gradients
        """
        # Initialize omega
        if omega_init is None:
            omega_val = np.array([
                0.5 * np.random.rand() + 0.5,
                0.5 * np.random.rand() + 0.5,
                0.5 * np.random.rand() + 0.5
            ])
        else:
            omega_val = omega_init.copy()

        omega_history = np.zeros((self.max_iter, 3))
        grad_history = np.zeros((self.max_iter, 3))

        theta_diff_history = np.zeros(self.max_iter)
        grad_diff_history = np.zeros(self.max_iter)
        A_diff_history = np.zeros(self.max_iter)
        value_diff_history = np.zeros(self.max_iter)
        b_diff_history = np.zeros(self.max_iter)

        n_repeat = 10
        theta_list = []
        A_diff_list = []
        b_diff_list = []

        try:
            for iter_num in range(self.max_iter):
                # Sample on-policy data
                # s_all, a_all = sample_data(
                #     self.n_traj, self.T, self.dt, omega_val,
                #     self.A, self.B, self.sigma
                # )

                # # Compute theta (value function parameters)
                # theta, A_diff = compute_theta_on_policy(
                #     omega_val, self.dt, s_all, a_all, 3,
                #     self.A_matrix, self.b_vector
                # )

                for _ in range(n_repeat):
                    # Sample on-policy data
                    s_all, a_all = sample_data(
                        self.n_traj, self.T, self.dt, omega_val,
                        self.A, self.B, self.sigma
                    )

                    # Compute theta
                    theta, A_diff, b_diff = compute_theta_on_policy(
                        omega_val, self.dt, s_all, a_all, 3,
                        self.A_matrix, self.b_vector
                    )

                    theta_list.append(theta)
                    A_diff_list.append(A_diff)
                    b_diff_list.append(b_diff)

                theta = np.mean(theta_list, axis=0)
                A_diff = np.mean(A_diff_list)
                b_diff = np.mean(b_diff_list)

                # Compute vanilla gradient
                grad = compute_grad_on_policy(
                    theta, s_all, a_all, omega_val,
                    self.beta, self.lambda_reg, self.M, self.N, self.dt
                )

                theta_phibe = compute_theta_phibe(
                    omega=omega_val,
                    beta=self.beta,
                    lambda_reg=self.lambda_reg,
                    M=self.M,
                    N=self.N,
                    dt=self.dt,
                    A=self.A,
                    B=self.B,
                    )

                def value_from_theta(s, theta):
                    s = np.asarray(s)
                    return theta[0] + theta[1] * s + theta[2] * s**2
                    
                s_grid = np.linspace(-4, 4, 400)
                V_theta = value_from_theta(s_grid, theta)
                V_phibe = value_from_theta(s_grid, theta_phibe)
                value_diff_history[iter_num] = np.sqrt(np.mean((V_theta - V_phibe) ** 2))
                
                grad_phibe = compute_grad_phibe(
                    theta=theta_phibe,
                    s_all=s_all,
                    a_all=a_all,
                    omega=omega_val,
                    beta=self.beta,
                    lambda_reg=self.lambda_reg,
                    M=self.M,
                    N=self.N,
                    dt=self.dt,
                    A=self.A,
                    B=self.B,
                    )

                theta_diff = np.linalg.norm(theta - theta_phibe)
                grad_diff = np.linalg.norm(grad - grad_phibe)

                theta_diff_history[iter_num] = theta_diff
                grad_diff_history[iter_num] = grad_diff
                A_diff_history[iter_num] = A_diff
                b_diff_history[iter_num] = b_diff

                # Compute Fisher information matrix inverse
                F_inv = compute_F_inv(s_all, a_all, omega_val)

                # Natural gradient
                grad = F_inv @ grad

                # Update omega
                omega_val = omega_val + self.eta * grad

                # Store values
                omega_history[iter_num, :] = omega_val
                grad_history[iter_num, :] = grad

                # Check convergence
                if iter_num > self.window_size:
                    omega_window = omega_history[iter_num - self.window_size + 1:iter_num + 1, :].T
                    omega_avg_curr = np.mean(omega_window, axis=1)

                    omega_window_prev = omega_history[iter_num - self.window_size:iter_num, :].T
                    omega_avg_prev = np.mean(omega_window_prev, axis=1)
                    avg_diff = np.linalg.norm(omega_avg_curr - omega_avg_prev)

                    if avg_diff < self.epsilon_avg:
                        if verbose:
                            print(f'Converged by average omega stability at iteration {iter_num}')
                        break

                # Display progress
                if verbose and (iter_num + 1) % 200 == 0:
                    print(f'Iteration {iter_num + 1}: omega = [{omega_val[0]:.6f}, {omega_val[1]:.6f}, {omega_val[2]:.6f}]')
                    print(f'Gradient at iteration {iter_num + 1}: [{grad[0]:.6f}, {grad[1]:.6f}, {grad[2]:.6f}]')

        except KeyboardInterrupt:
            if verbose:
                print(f"\n\nManually interrupted at iteration {iter_num}. Returning results collected so far...")

        return omega_val, omega_history, grad_history, theta_diff_history, grad_diff_history, A_diff_history, value_diff_history, b_diff_history

    def run_multiple_trials(
        self,
        n_repeat: int = 50,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Run multiple trials and compute average trajectory.

        Args:
            n_repeat: Number of trials to run
            verbose: Whether to print progress

        Returns:
            mean_history: Average omega history across successful trials
        """
        omega_history_all = np.zeros((self.max_iter, 3, n_repeat))

        successful_trials = 0
        trial_num = 0

        try:
            while successful_trials < n_repeat:
                if verbose:
                    print(f"\n=== Trial {trial_num + 1} ===")

                omega_val, omega_history, grad_history = self.run_single_trial(verbose=verbose)

                # Check if trial was successful (has non-zero values)
                if np.any(omega_history != 0):
                    omega_history_all[:, :, successful_trials] = omega_history
                    successful_trials += 1

                trial_num += 1

                if trial_num > n_repeat * 3:  # Safety limit
                    print(f"Warning: Reached maximum trial attempts. Only {successful_trials} successful trials.")
                    break

        except KeyboardInterrupt:
            if verbose:
                print(f"\n\nManually interrupted. Completed {successful_trials} trials. Computing results...")

        # Compute mean across successful trials (even if interrupted)
        if successful_trials > 0:
            mean_history = mean_nonzero_real_3d(omega_history_all[:, :, :successful_trials])
        else:
            print("Warning: No successful trials completed.")
            mean_history = np.zeros((self.max_iter, 3))

        return mean_history


def main():
    """Example usage of Natural Policy Gradient."""
    print("Natural Policy Gradient for LQR\n")

    # Initialize algorithm
    npg = NaturalPG_on_policy(
        A=-1.0,
        B=1.0,
        sigma=1.0,
        M=1.0,
        N=1.0,
        lambda_reg=0.1,
        beta=1,
        learning_rate=0.05,
        T=0.1,
        dt=0.1,
        n_traj=300,
        max_iter=1500
    )

    # Run single trial
    print("Running single trial...")
    omega_val, omega_history, grad_history, theta_diff_history, grad_diff_history = npg.run_single_trial(verbose=True)

    # Get final omega
    final_iter = np.max(np.where(np.any(omega_history != 0, axis=1))[0])
    final_omega = omega_history[final_iter, :]
    print(f"\nFinal omega: {final_omega}")


if __name__ == "__main__":
    main()
