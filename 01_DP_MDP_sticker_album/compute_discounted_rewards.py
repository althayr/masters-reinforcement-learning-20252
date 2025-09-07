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
    # States: i = 0, ..., N for each the amount of stickers already collected
    # State N is absorbing (album complete)
    num_states = N + 1
    
    # Initialize transition matrix P
    P = np.zeros((num_states, num_states))
    P[N, N] = 1.0

    # Constructing P
    # From state i we can transition to any state from i+A until i+min(N, B), state N absorbs all the remaining probabilities
    for i in range(0, N):
        for k in range(A, B+1):
            next_state = i + k if B <= N-i else N
            P[i, next_state] += 1/(B - A + 1)
    
    # Reward vector R_bar
    R_bar = np.zeros(num_states)
    R_bar[:N] = 1.0
    
    # Solve (I - gamma * P) * J_bar = R_bar
    I = np.identity(num_states)
    J_bar = np.linalg.solve(I - gamma * P, R_bar)
    return P, J_bar


def main():
    if len(sys.argv) != 4:
        print("Usage: python compute_discounted_rewards.py N A B")
        sys.exit(1)
    
    N, A, B = [int(s) for s in sys.argv[1:]]
    gamma = 0.5
    
    print(f"Computing discounted rewards for N={N}, A={A}, B={B}, gamma={gamma}\n")
    
    # Compute using matrix inversion method
    P, J_bar = compute_discounted_rewards(N, A, B, gamma)
    
    # Print results
    print("Transition matrix P:")
    print(P)
    print('\n', "="*20, '\n')
    print("Expected discounted rewards J_bar")
    print(J_bar)
    
    print('\n', "="*20, '\n')
    print(f"\nExpected discounted cost to complete album: {J_bar[0]:.6f}")


if __name__ == "__main__":
    main()