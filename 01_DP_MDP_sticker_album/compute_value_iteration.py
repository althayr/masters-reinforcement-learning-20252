import numpy as np
import sys
from compute_discounted_rewards import compute_discounted_rewards, print_vector


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
    # States: 0 (complete) to N (need N stickers)
    num_states = N + 1
    
    # Initialize value function
    J = np.zeros(num_states)
    J_new = np.zeros(num_states)
    
    # State 0 is absorbing with value 0
    J[0] = 0.0
    J_new[0] = 0.0
    
    lower_bound = 1 if A == 0 else A
    
    for iteration in range(max_iterations):
        # Update all transient states
        for i in range(1, N + 1):
            expected_future_cost = 0.0
            upper_bound = min(B, i) + 1
            
            if A == 0:
                # Uniform distribution over [0, B]
                prob_per_packet = 1.0 / (B + 1)
                
                # Probability of 0-sticker packet (stay in same state)
                expected_future_cost += prob_per_packet * J[i]
                
                # Probabilities for 1 to min(B, i) stickers
                for k in range(lower_bound, upper_bound):
                    expected_future_cost += prob_per_packet * J[i - k]
            else:
                # Uniform distribution over [A, B]
                total_packets = B - A + 1
                prob_per_packet = 1.0 / total_packets
                
                for k in range(lower_bound, upper_bound):
                    expected_future_cost += prob_per_packet * J[i - k]
            
            # Bellman equation: J(i) = 1 + gamma * E[J(next_state)]
            J_new[i] = 1.0 + gamma * expected_future_cost
        
        # Check convergence
        max_diff = np.max(np.abs(J_new - J))
        if max_diff < epsilon:
            return J_new, iteration + 1
        
        # Update for next iteration
        J = J_new.copy()
    
    print(f"Warning: Did not converge after {max_iterations} iterations")
    return J_new, max_iterations


def main():
    if len(sys.argv) != 4:
        print("Usage: python compute_value_iteration.py N A B")
        sys.exit(1)
    
    N, A, B = map(int, sys.argv[1:])
    gamma = 0.5
    epsilon = 1e-5
    
    print(f"Computing discounted rewards for N={N}, A={A}, B={B}, gamma={gamma}")
    print(f"Using value iteration with epsilon={epsilon}")
    
    # Compute using value iteration method
    J_bar_iter, iterations = value_iteration(N, A, B, gamma, epsilon)
    
    print(f"\nValue iteration converged after {iterations} iterations")
    print_vector(J_bar_iter, "Expected Discounted Costs J̄ (Value Iteration)")
    
    print(f"\nExpected discounted cost to complete album: {J_bar_iter[N]:.6f}")
    
    # Compare with matrix inversion method
    print("\n" + "="*60)
    print("COMPARISON WITH MATRIX INVERSION METHOD")
    print("="*60)
    
    P, J_bar_matrix = compute_discounted_rewards(N, A, B, gamma)
    print_vector(J_bar_matrix, "Expected Discounted Costs J̄ (Matrix Inversion)")
    
    print(f"\nMatrix inversion result: {J_bar_matrix[N]:.6f}")
    print(f"Value iteration result: {J_bar_iter[N]:.6f}")
    print(f"Absolute difference: {abs(J_bar_matrix[N] - J_bar_iter[N]):.8f}")
    
    # Print detailed comparison for all states
    print("\nDetailed comparison:")
    print("State | Matrix Inv | Value Iter | Difference")
    print("-" * 45)
    for i in range(N + 1):
        diff = abs(J_bar_matrix[i] - J_bar_iter[i])
        print(f"{i:>4} | {J_bar_matrix[i]:>10.6f} | {J_bar_iter[i]:>10.6f} | {diff:>10.8f}")


if __name__ == "__main__":
    main()