def solve(N, A, B):
    cache = [None]*(N+1)
    cache[N] = 0
    lower_bound = 1 if A == 0 else A
    factor_zero_1 = (B+1)/B
    factor_zero_2 = (B + 1)
    factor_nonzero = (B - A + 1)
    for i in range(N-1, -1, -1):
        upper_bound = min(B, N-i) + 1
        sum_ = sum(cache[i + lower_bound: i+upper_bound])
        if A == 0:
            value = factor_zero_1 * (1 + sum_ / factor_zero_2)
        else:
            value = 1 + sum_ / factor_nonzero
        cache[i] = value

    print(cache[0])


if __name__ == "__main__":
    N, A, B = [int(s) for s in input().strip().split()]
    solve(N, A, B)
