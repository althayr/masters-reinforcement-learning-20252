# Sticker Album Problem - Dynamic Programming and MDP Solutions

This folder contains solutions to the sticker album problem from competitive programming, implemented using different approaches: classic dynamic programming and Markov Decision Process (MDP) formulations.

## Problem Description

The sticker album problem asks: given an album that needs N stickers to be complete, and sticker packets containing a uniform random number of stickers between A and B (inclusive), what is the expected number of packets needed to complete the album?

**Input:** Three integers N, A, B where:
- N: number of stickers needed to fill the album (1 ≤ N ≤ 10⁶)
- A: minimum stickers per packet (0 ≤ A ≤ B ≤ 10⁶)  
- B: maximum stickers per packet (B > 0)

**Output:** Expected number of packets to complete the album

## Scripts

### 1. `run.py` - Classic Dynamic Programming Solution

Solves the original problem using backward dynamic programming with memoization.

**Usage:**
```bash
echo "N A B" | python run.py
```

**Example:**
```bash
echo "2 0 1" | python run.py
# Output: 4.0
```

**Algorithm:**
- State: `cache[i]` = expected packets needed when i stickers remain
- Base case: `cache[0] = 0` (album complete)
- Recurrence: considers all possible packet sizes and their probabilities

### 2. `compute_discounted_rewards.py` - MDP with Matrix Inversion

Formulates the problem as an MDP with discount factor γ=0.5 and unit cost per packet. Solves using direct matrix inversion method.

**Usage:**
```bash
python compute_discounted_rewards.py N A B
```

**Example:**
```bash
python compute_discounted_rewards.py 2 0 1
```

**Algorithm:**
- Models states as Markov chain (0=complete, 1 to N=stickers remaining)
- Constructs transition matrix P and reward vector R
- Solves: J̄ = (I - γP)⁻¹R̄ where γ=0.5

**Output:**
- Transition matrix P
- Expected discounted costs J̄ for each state
- Final discounted cost to complete album

### 3. `compute_value_iteration.py` - MDP with Value Iteration

Solves the same MDP formulation using iterative value iteration method with convergence threshold ε=1e-5.

**Usage:**
```bash
python compute_value_iteration.py N A B
```

**Example:**
```bash
python compute_value_iteration.py 2 0 1
```

**Algorithm:**
- Iteratively updates: J⁽ⁿ⁺¹⁾[i] = 1 + γ × Σ P(k) × J⁽ⁿ⁾[i-k]
- Converges when max|J⁽ⁿ⁺¹⁾ - J⁽ⁿ⁾| < 1e-5
- Automatically compares results with matrix inversion method

**Output:**
- Number of iterations until convergence
- Expected discounted costs from value iteration
- Detailed comparison with matrix inversion results

## Key Differences Between Approaches

| Aspect | Classic DP | MDP (Matrix/Iteration) |
|--------|------------|------------------------|
| Objective | Minimize expected packets | Minimize discounted cost |
| Discount | γ = 1 (no discount) | γ = 0.5 |
| Method | Backward recursion | Forward/matrix methods |
| Result | Higher values | Lower values (due to discount) |

## Example Results

For N=2, A=0, B=1:
- **Classic DP:** 4.0 expected packets
- **MDP (γ=0.5):** 1.778 expected discounted cost

The MDP formulation shows how discounting future costs significantly reduces the total expected cost, as packets purchased later are worth less than immediate packets.

## Test Cases

Try these examples to understand the algorithms:

```bash
# Simple cases
python compute_value_iteration.py 2 0 1
python compute_value_iteration.py 3 0 1
python compute_value_iteration.py 3 1 2
python compute_value_iteration.py 4 1 2

# Compare with classic DP
echo "2 0 1" | python run.py
echo "3 1 2" | python run.py
```

Both MDP methods should produce identical results, demonstrating the equivalence of matrix inversion and value iteration approaches.