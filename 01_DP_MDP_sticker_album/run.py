def compute_expectation(i, N, A, B, cache):
    if cache[i] != None:
        return cache[i]

    if i >= N:
        return 0
    sum_ = 0
    upper_bound = min(B, N-i-1) + 1
    if A == 0:
        for k in range(1, upper_bound):
            sum_ += cache[i + k] if cache[i + k] != None else compute_expectation(i+k, N, A, B, cache)
        value = (B+1)/B * (1 + sum_ / (B + 1))
    else:
        for k in range(A, upper_bound):
            sum_ += cache[i + k] if cache[i + k] != None else compute_expectation(i+k, N, A, B, cache)
        value = 1 + sum_ / (B - A + 1)
        
    # print("Adding to cache: {} - {}".format(i, value))
    cache[i] = value
    return value

def solve(N, A, B):
    cache = [None]*(N+1)
    # Compute in decreasing order for more efficient cache hits
    for i in range(N, -1, -1):
        compute_expectation(i, N, A, B, cache)
    print(cache[0])


if __name__ == "__main__":
    N, A, B = [int(s) for s in input().strip().split()]
    solve(N, A, B)
