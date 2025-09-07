import sys
import numpy as np


def initialize_R(N):
    R_bar = np.zeros(N + 1)
    R_bar[:N] = 1.0
    return R_bar


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


def compute_discounted_rewards(N, A, B, gamma=0.5):
    """
    Computes discounted expected rewards using matrix inversion method.

    Args:
        N: Number of stickers needed to complete album
        A: Minimum stickers per packet
        B: Maximum stickers per packet
        gamma: Discount factor

    Returns:
        P: Transition matrix
        J_bar: Expected discounted costs for each state
    """
    # States: i = 0, ..., N for each the amount of stickers already collected
    # State N is absorbing (album complete)
    num_states = N + 1
    R_bar = initialize_R(N)
    P = initialize_P(N, A, B)

    print(f"Transition matrix P:\n{P}\n\nState rewards:\n{R_bar}\n")

    # Solve (I - gamma * P) * J_bar = R_bar
    I = np.identity(num_states)
    J_bar = np.linalg.solve(I - gamma * P, R_bar)
    return J_bar


def main():
    if len(sys.argv) != 5:
        print("Usage: uv run python3 compute_discounted_rewards.py N A B gamma")
        sys.exit(1)

    N, A, B, gamma = *[int(s) for s in sys.argv[1:4]], np.float32(sys.argv[4])

    print(f"Computing discounted rewards for N={N}, A={A}, B={B}, gamma={gamma}\n")

    J_bar = compute_discounted_rewards(N, A, B, gamma)

    print(f"Expected discounted rewards J_bar:\n{J_bar}\n")
    print(f"Expected discounted cost to complete album: {J_bar[0]:.6f}")


if __name__ == "__main__":
    main()
