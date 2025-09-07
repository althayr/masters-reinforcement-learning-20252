import sys
import numpy as np


def initialize_R(N):
    R_bar = np.zeros(N + 1)
    R_bar[:N] = 1.0
    return R_bar


def initialize_J(R):
    # Initialize value function
    N = R.shape[0]
    J = R
    J_next = np.zeros(N + 1)
    J_next[N] = 0.0
    return J, J_next


def initialize_P(N, A, B):
    P = np.zeros((N + 1, N + 1))
    P[N, N] = 1.0
    # Constructing P
    # From state i we can transition to any state from i+A until i+min(N, B), state N absorbs all the remaining probabilities
    for i in range(0, N):
        for k in range(A, B + 1):
            next_state = i + k if B <= N - i else N
            P[i, next_state] += 1 / (B - A + 1)
    return P


def value_iteration(N, A, B, gamma=0.5, epsilon=1e-5, max_iterations=10000):
    """
    Computes discounted expected rewards using value iteration method.

    Args:
        N: Number of stickers needed to complete album
        A: Minimum stickers per packet
        B: Maximum stickers per packet
        gamma: Discount factor
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        J_bar: Expected discounted costs for each state
        iterations: Number of iterations until convergence
    """
    R = initialize_R(N)
    J, J_next = initialize_J(R)
    P = initialize_P(N, A, B)

    print(f"Transition matrix P:\n{P}\n\nState rewards:\n{R}\n")
    print(f"Initial expected discounted rewards in k-steps: \n{J}\n")

    for i, iteration in enumerate(range(max_iterations), 1):
        print(f"Iteration {i} - J_{{k+1}}: {J_next}")
        J_next = R + gamma * P @ J

        # Check convergence
        max_diff = np.max(np.abs(J_next - J))
        if max_diff < epsilon:
            return J_next, iteration + 1

        J = J_next.copy()

    print(f"Warning: Did not converge after {max_iterations} iterations")
    return J_next, max_iterations


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: uv run python3 compute_value_iteration.py N A B gamma epsilon")
        sys.exit(1)

    N, A, B, gamma = *[int(s) for s in sys.argv[1:4]], np.float32(sys.argv[4])
    epsilon = 1e-5

    print(
        f"Computing discounted rewards for N={N}, A={A}, B={B}, gamma={gamma}, epsilon={epsilon}\n"
    )
    J_bar_final, iterations = value_iteration(N, A, B, gamma, epsilon)

    print(f"\nValue iteration converged after {iterations} iterations\n")
    print(f"Expected discounted rewards J_bar:\n{J_bar_final}\n")
    print(f"Expected discounted cost to complete album: {J_bar_final[0]:.6f}")
