import math
import numpy as np
import matplotlib.pyplot as plt


def plot_matrices(P, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(P, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(P.shape[1]))
    ax.set_yticks(np.arange(P.shape[0]))
    ax.set_xticklabels(np.arange(P.shape[1]))
    ax.set_yticklabels(np.arange(P.shape[0]))
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Labels
    ax.set_xlabel('Next State (N\')')
    ax.set_ylabel('Current State (N)')
    ax.set_title(f'Transition Matrix Heatmap\nShape: {P.shape[0]}x{P.shape[1]}')
    
    # Add grid for clarity (optional)
    ax.set_xticks(np.arange(P.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(P.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Remove ticks
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Add text annotations for small matrices (optional, only for matrices < 20x20)
    if P.shape[0] <= 20 and P.shape[1] <= 20:
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                text = ax.text(j, i, f'{P[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {filename}")
    return


def pmf(k, lamb):
    if k < 0:
        return 0
    return np.exp(-lamb) * lamb**k/math.factorial(k)

def cdf(k, lamb):
    if k < 0:
        return 0
    return np.sum(np.exp(-lamb) * lamb ** np.arange(k+1) / np.array([math.factorial(i) for i in range(k+1)]))
    

def get_z_prime_dist(N, lambda_x):
    p_z_prime = np.zeros(N+1)
    p_z_prime[0] = 1 - cdf(N-1, lambda_x)
    for z_prime in range(1, N+1):
        p_z_prime[z_prime] = pmf(N - z_prime, lambda_x)
    return p_z_prime
        

def get_conditional_s_prime_dist(s_new, z_prime, N_max, lambda_y):
    if s_new < z_prime:
        return 0
    elif s_new == N_max:
        return 1 - cdf(N_max - z_prime - 1, lambda_y)
    else:
        return pmf(s_new - z_prime, lambda_y)


def get_single_location_transitions(N_max, lambda_x, lambda_y):
    # All the math is conditional on a initial N, to compute for all states we will need a tensor
    P = np.zeros((N_max+1, N_max+1))
    for N in range(0, N_max+1):
        # Z' goes from 0 to N
        p_z_prime = get_z_prime_dist(N, lambda_x)
        # print("P(z') shape: {}".format(p_z_prime.shape))
        
        # Computing the output dist for every S_t+1 = s'
        for N_new in range(0, N_max+1):
            # print("Computing P {} - {}".format(N, N_new))
            p_final = 0
            for z_prime in range(N+1):
                p_cond = get_conditional_s_prime_dist(N_new, z_prime, N_max, lambda_y)
                # print("N: {}, Z': {}, S': {}, P(S'|z'): {}".format(N, z_prime, N_new, p_cond))
                p_final += p_cond * p_z_prime[z_prime]
            P[N, N_new] = p_final
    return P

def get_joint_transition_matrix(a, P_n1, P_n2, N1_max, N2_max):
    n_states = (N1_max + 1) * (N2_max + 1)
    P_a = np.zeros((n_states, n_states))
    
    for s_idx in range(0, n_states):
        n1, n2 = index_to_state(s_idx, N2_max)
        
        n1_after_action = max(0, n1 - a) if a >= 0 else min(N1_max, n1 - a)
        n2_after_action = min(N2_max, n2 + a) if a >= 0 else max(0, n2 + a)
        
        for s_prime_idx in range(0, n_states):
            n1_prime, n2_prime = index_to_state(s_prime_idx, N2_max)
            P_a[s_idx, s_prime_idx] = P_n1[n1_after_action, n1_prime] * P_n2[n2_after_action, n2_prime]
    return P_a


def state_to_index(n1, n2, N2_max):
    """Convert 2D state (n1, n2) to 1D index"""
    return n1 * (N2_max + 1) + n2


def index_to_state(idx, N2_max):
    """Convert 1D index back to 2D state (n1, n2)"""
    n1 = idx // (N2_max + 1)
    n2 = idx % (N2_max + 1)
    return n1, n2


if __name__ == "__main__":
    N1_max = 20
    N2_max = 20
    lambda_x1, lambda_y1 = 3, 3
    lambda_x2, lambda_y2 = 4, 2
    a_min, a_max = -5, 5

    P_n1 = get_single_location_transitions(N1_max, lambda_x1, lambda_y1)
    plot_matrices(P_n1, "transition_matrix_location1.png")
    P_n2 = get_single_location_transitions(N2_max, lambda_x2, lambda_y2)
    plot_matrices(P_n2, "transition_matrix_location2.png")
    
    print("Computing state transition matrices for all actions...")
    n_actions = a_max - a_min + 1
    n_states = (N1_max + 1) * (N2_max + 1)

    P_actions = {}
    for a in range(a_min, a_max+1):
        P_a = get_joint_transition_matrix(a, P_n1, P_n2, N1_max, N2_max)
        # plot_matrices(P_a, f"transition_matrix_action_{a}.png")
        row_sums = np.sum(P_a, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-5):
            raise
        P_actions[a] = P_a

    print("Computing the rewards for each state-action")
    
