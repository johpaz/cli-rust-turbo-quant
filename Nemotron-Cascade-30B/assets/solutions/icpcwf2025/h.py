import sys
import math

INF_NEG = -10 ** 9          # sufficiently small for DP values


def count_digits(num):
    """return list cnt[0..9] of decimal digits of num (0 -> cnt[0]=1)"""
    cnt = [0] * 10
    if num == 0:
        cnt[0] = 1
        return cnt
    while num:
        cnt[num % 10] += 1
        num //= 10
    return cnt


def dp_tight_max(digits, target_set, g):
    """
    digit DP with upper bound given by digits list (most significant first)
    target_set – set of digits that count (for digit 6 it is {6,9})
    return maximal count of those digits among numbers ≤ bound,
           remainder 0 (multiple of g).  Return None if no such number.
    """
    L = len(digits)
    dp = [[INF_NEG] * g for _ in range(2)]          # dp[tight][rem]
    # first digit, cannot be zero
    max_digit = digits[0]
    for d in range(1, max_digit + 1):
        rem = d % g
        cnt = 1 if d in target_set else 0
        tight = 1 if d == max_digit else 0
        if cnt > dp[tight][rem]:
            dp[tight][rem] = cnt

    # remaining positions
    for pos in range(1, L):
        ndp = [[INF_NEG] * g for _ in range(2)]
        max_digit = digits[pos]
        for tight in (0, 1):
            limit = max_digit if tight else 9
            for rem in range(g):
                cur = dp[tight][rem]
                if cur < 0:
                    continue
                for dig in range(limit + 1):
                    new_rem = (rem * 10 + dig) % g
                    new_tight = 1 if (tight == 1 and dig == limit) else 0
                    new_cnt = cur + (1 if dig in target_set else 0)
                    if new_cnt > ndp[new_tight][new_rem]:
                        ndp[new_tight][new_rem] = new_cnt
        dp = ndp

    best = max(dp[0][0], dp[1][0])
    return best if best >= 0 else None


def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    M = int(next(it))
    n = int(next(it))
    p = [int(next(it)) for _ in range(n)]

    g = 0
    for v in p:
        g = math.gcd(g, v)

    max_p = max(p)
    B = max_p * max_p                     # safe conductor bound, ≤ 1 000 000
    if B > M:
        B = M                              # we never need dp beyond M

    # -------------------- reachable sums up to B --------------------
    dp = [False] * (B + 1)
    dp[0] = True
    for coin in p:
        for i in range(coin, B + 1):
            if dp[i - coin]:
                dp[i] = True

    best = [0] * 9        # best[6] will be cnt6+cnt9

    # ----- scores ≤ B that are reachable exactly -----
    limit = min(B, M)
    for x in range(limit + 1):
        if not dp[x]:
            continue
        cnt = count_digits(x)
        for d in range(9):
            if d == 6:
                val = cnt[6] + cnt[9]
            else:
                val = cnt[d]
            if val > best[d]:
                best[d] = val

    # ----- the maximal score M (always reachable) -----
    cntM = count_digits(M)
    for d in range(9):
        if d == 6:
            val = cntM[6] + cntM[9]
        else:
            val = cntM[d]
        if val > best[d]:
            best[d] = val

    # If M ≤ B we are done
    if M <= B:
        out = []
        for d in range(9):
            if best[d] > 0:
                out.append(f"{d} {best[d]}")
        sys.stdout.write("\n".join(out))
        return

    # -------------------- now M > B --------------------
    lenB = len(str(B))
    lenM = len(str(M))
    pow10_lenB = 10 ** lenB

    # ---- numbers with the same number of digits as B and > B ----
    start = B + 1
    end = min(M, pow10_lenB - 1)
    if start <= end:
        for x in range(start, end + 1):
            if x % g != 0:
                continue          # not reachable
            cnt = count_digits(x)
            for d in range(9):
                if d == 6:
                    val = cnt[6] + cnt[9]
                else:
                    val = cnt[d]
                if val > best[d]:
                    best[d] = val

    # ---- numbers with more than lenB digits (lenB+1 … lenM) ----
    # prepare the decimal representation of M for the tight DP
    digitsM = list(map(int, str(M)))

    for d in range(9):
        target = {d}
        if d == 6:
            target = {6, 9}

        # DP for lengths 1 … lenM
        # initialise for length = 1 (first digit cannot be zero)
        dp_len = [INF_NEG] * g
        for first in range(1, 10):
            rem = first % g
            cnt = 1 if first in target else 0
            if cnt > dp_len[rem]:
                dp_len[rem] = cnt

        # process lengths 2 … lenM-1   (non‑tight)
        for L in range(2, lenM):
            new_dp = [INF_NEG] * g
            for rem in range(g):
                cur = dp_len[rem]
                if cur < 0:
                    continue
                for dig in range(10):
                    nrem = (rem * 10 + dig) % g
                    ncnt = cur + (1 if dig in target else 0)
                    if ncnt > new_dp[nrem]:
                        new_dp[nrem] = ncnt
            dp_len = new_dp
            if L >= lenB + 1:
                if dp_len[0] > best[d]:
                    best[d] = dp_len[0]

        # tight DP for the maximal length lenM (only if lenM > lenB)
        if lenM > lenB:
            dp_t = dp_tight_max(digitsM, target, g)
            if dp_t is not None and dp_t > best[d]:
                best[d] = dp_t

    # -------------------- output --------------------
    out_lines = []
    for d in range(9):
        if best[d] > 0:
            out_lines.append(f"{d} {best[d]}")
    sys.stdout.write("\n".join(out_lines))


if __name__ == "__main__":
    solve()

