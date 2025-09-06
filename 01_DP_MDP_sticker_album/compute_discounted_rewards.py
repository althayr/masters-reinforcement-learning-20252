import numpy as np
import sys


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
    # States: 0 (complete) to N (need N stickers)
    # State 0 is absorbing (album complete)
    num_states = N + 1
    
    # Initialize transition matrix P
    P = np.zeros((num_states, num_states))
    
    # State 0 is absorbing
    P[0, 0] = 1.0
    
    # For states 1 to N (need i stickers)
    lower_bound = 1 if A == 0 else A
    
    for i in range(1, N + 1):
        # From state i, we can transition to states i-k where k is packet size
        upper_bound = min(B, i) + 1
        
        if A == 0:
            # Uniform distribution over [0, B], but 0-sticker packets don't help
            # So we only consider [1, min(B, i)]
            prob_per_packet = 1.0 / (B + 1)
            for k in range(lower_bound, upper_bound):
                P[i, i - k] += prob_per_packet
            # Probability of 0-sticker packet (stay in same state)
            P[i, i] = prob_per_packet
        else:
            # Uniform distribution over [A, B]
            total_packets = B - A + 1
            prob_per_packet = 1.0 / total_packets
            for k in range(lower_bound, upper_bound):
                P[i, i - k] += prob_per_packet
    
    # Reward vector R_bar (cost of 1 for each transition from transient states)
    R_bar = np.zeros(num_states)
    R_bar[1:] = 1.0  # Cost 1 for all transient states (1 to N)
    
    # Solve (I - gamma * P) * J_bar = R_bar
    I = np.eye(num_states)
    A_matrix = I - gamma * P
    J_bar = np.linalg.solve(A_matrix, R_bar)
    
    return P, J_bar


def print_matrix(matrix, name, precision=4):
    """Pretty print matrix with proper formatting."""
    print(f"\n{name}:")
    print("-" * (len(name) + 1))
    n_rows, n_cols = matrix.shape
    
    # Print column headers
    print("     ", end="")
    for j in range(n_cols):
        print(f"{j:>8}", end="")
    print()
    
    # Print matrix rows
    for i in range(n_rows):
        print(f"{i:>3}: ", end="")
        for j in range(n_cols):
            print(f"{matrix[i,j]:>8.{precision}f}", end="")
        print()


def print_vector(vector, name, precision=4):
    """Pretty print vector."""
    print(f"\n{name}:")
    print("-" * (len(name) + 1))
    for i, val in enumerate(vector):
        print(f"State {i}: {val:.{precision}f}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python compute_discounted_rewards.py N A B")
        sys.exit(1)
    
    N, A, B = map(int, sys.argv[1:])
    gamma = 0.5
    
    print(f"Computing discounted rewards for N={N}, A={A}, B={B}, gamma={gamma}")
    
    # Compute using matrix inversion method
    P, J_bar = compute_discounted_rewards(N, A, B, gamma)
    
    # Print results
    print_matrix(P, "Transition Matrix P")
    print_vector(J_bar, "Expected Discounted Costs J")
    
    print(f"\nExpected discounted cost to complete album: {J_bar[N]:.6f}")


if __name__ == "__main__":
    main()