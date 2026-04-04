import sys
from collections import deque
from array import array

# ------------------------------------------------------------
#  pre‑computation up to the maximal n
# ------------------------------------------------------------
def sieve_spf_and_primes(limit):
    spf = array('I', [0]) * (limit + 1)
    primes = []
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > limit:
                break
            spf[i * p] = p
    return spf, primes

# ------------------------------------------------------------
def solve_one_case(n, spf, parity, primes):
    # ---------- 1. greedy matching (left vertices are those with even Ω) ----------
    match = array('I', [0]) * (n + 1)

    # try to match left vertices in decreasing order
    for v in range(n, 0, -1):
        if parity[v] != 0:          # not a left vertex
            continue
        # try to match v
        ok = False
        if v == 1:
            # match 1 with the first free prime ≤ n
            for p in primes:
                if p > n:
                    break
                if match[p] == 0:
                    match[1] = p
                    match[p] = 1
                    ok = True
                    break
        else:
            # 1) parent
            p = spf[v]
            parent = v // p
            if parent <= n and match[parent] == 0:
                match[v] = parent
                match[parent] = v
                ok = True
            else:
                # 2) children (multiplication by a prime ≤ spf[v])
                for p in primes:
                    if p > spf[v]:
                        break
                    to = v * p
                    if to > n:
                        break
                    if match[to] == 0:
                        match[v] = to
                        match[to] = v
                        ok = True
                        break
            if not ok:
                # 3) division neighbours (different prime divisors)
                x = v
                last = 0
                while x > 1:
                    q = spf[x]
                    if q != last:
                        to = v // q
                        if match[to] == 0:
                            match[v] = to
                            match[to] = v
                            ok = True
                            break
                        last = q
                    x //= q
        if not ok and (v & 1) == 0:          # even and still unmatched
            return f"first {v}"

    # ---------- 2. no unmatched even left vertex ----------
    # free right vertices
    visited = bytearray(n + 1)
    q = deque()
    answer = None
    for v in range(1, n + 1):
        if parity[v] == 1 and match[v] == 0:
            visited[v] = 1
            q.append(v)
            if (v & 1) == 0:          # even free right vertex -> winning start
                answer = v
                break
    if answer is not None:
        return f"first {answer}"

    # ---------- 3. BFS from all free right vertices ----------
    while q and answer is None:
        cur = q.popleft()
        if parity[cur] == 0:                 # left vertex, follow its matched edge
            nxt = match[cur]
            if nxt and not visited[nxt]:
                visited[nxt] = 1
                if (nxt & 1) == 0:
                    answer = nxt
                    break
                q.append(nxt)
        else:                               # right vertex, follow all non‑matching edges
            # division neighbours (prime divisors)
            t = cur
            last = 0
            while t > 1:
                p = spf[t]
                if p != last:
                    u = cur // p
                    if not visited[u] and u != match[cur]:
                        visited[u] = 1
                        if (u & 1) == 0:
                            answer = u
                            break
                        q.append(u)
                    last = p
                t //= p
            if answer is not None:
                break
            # multiplication neighbours (primes p with cur·p ≤ n)
            limit = n // cur
            for p in primes:
                if p > limit:
                    break
                u = cur * p
                if not visited[u] and u != match[cur]:
                    visited[u] = 1
                    if (u & 1) == 0:
                        answer = u
                        break
                    q.append(u)
            # continue loop
    if answer is not None:
        return f"first {answer}"
    else:
        return "second"

# ------------------------------------------------------------
def solve() -> None:
    data = sys.stdin.read().split()
    t = int(data[0])
    ns = list(map(int, data[1:]))
    max_n = max(ns)

    spf, primes = sieve_spf_and_primes(max_n)

    # parity: 0 -> left (even Ω), 1 -> right (odd Ω)
    parity = bytearray(max_n + 1)
    parity[1] = 0
    for i in range(2, max_n + 1):
        parity[i] = parity[i // spf[i]] ^ 1

    out_lines = []
    for n in ns:
        out_lines.append(solve_one_case(n, spf, parity, primes))
    sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
    solve()

