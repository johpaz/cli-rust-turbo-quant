import sys
import math

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    h = int(data[1])

    min_h = 2 * n - 1          # decreasing order
    max_h = n * n              # increasing order

    if h < min_h or h > max_h:
        print("impossible")
        return

    # extra height over the minimal one
    K = h - min_h

    # try all possible numbers L of the initial decreasing segment
    for L in range(0, n):          # L can be 0 … n‑1
        m = n - L - 1               # number of possible small indices
        if m < 0:
            continue
        T = K + L                   # required sum of odd numbers for S'
        if T > m * m:               # not enough odd numbers
            continue

        d = m * m - T               # non‑negative
        sqrt_d = math.isqrt(d)

        # lower bound for r : ceil(m - sqrt(m^2 - T)) = m - floor(sqrt(d))
        r_low = m - sqrt_d
        if r_low < 0:
            r_low = 0
        r_up = math.isqrt(T)        # from r^2 <= T

        if r_low > r_up:
            continue

        # choose the smallest r in the interval with correct parity
        r = r_low
        if (T + r) & 1:
            r += 1
        if r > r_up:
            continue

        # sum of indices we need
        sum_i_target = (T + r) // 2

        # build S' – r distinct indices from 1..m whose sum is sum_i_target
        # start with smallest r numbers
        S_prime = list(range(1, r + 1))
        cur_sum = r * (r + 1) // 2
        extra = sum_i_target - cur_sum

        # increase numbers from the end while we still have extra
        i = r - 1
        while extra > 0 and i >= 0:
            # maximal value we can put at position i
            max_val = m - (r - 1 - i)
            cur = S_prime[i]
            inc = min(max_val - cur, extra)
            S_prime[i] += inc
            extra -= inc
            i -= 1
        # extra must be zero now
        # ----------------------------------------------------
        # Build the final ordering
        # large segment : n, n-1, ..., n-L+1
        large = list(range(n, n - L, -1))   # decreasing

        # chain : S' (in increasing order) then the biggest remaining cup (n-L)
        chain = S_prime + [n - L]

        # leftovers : all numbers 1..m not in S', placed decreasing
        S_set = set(S_prime)
        leftovers = [i for i in range(m, 0, -1) if i not in S_set]

        order_indices = large + chain + leftovers
        # convert to heights
        order_heights = [2 * idx - 1 for idx in order_indices]
        print(" ".join(map(str, order_heights)))
        return

    # should never be reached because the existence is guaranteed for
    # all h in [2n-1 , n^2], but keep the fallback
    print("impossible")


if __name__ == "__main__":
    solve()

