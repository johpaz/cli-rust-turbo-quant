import sys
import math

def solve() -> None:
    data = list(map(int, sys.stdin.read().split()))
    it = iter(data)
    s = next(it)
    r = next(it)
    d = next(it)
    total_nodes = s + r

    # outgoing ducts for each station (1 .. s)
    station_out = [[] for _ in range(s + 1)]

    for _ in range(d):
        i = next(it)
        n = next(it)
        outs = []
        probs = []
        for _ in range(n):
            o = next(it)
            p = next(it)
            outs.append(o)
            probs.append(p / 100.0)
        station_out[i].append((outs, probs))

    # -------------------------------------------------------------
    # DP: for a given vector v (length r) returns (value[1], weight[1])
    def dp(v):
        # v[k] is value for reservoir with index k (0‑based)
        y = [0.0] * (total_nodes + 1)          # value[i]
        w = [None] * (total_nodes + 1)        # weight[i] : list of length r

        for i in range(total_nodes, 0, -1):
            if i > s:                          # reservoir
                idx = i - (s + 1)              # 0 … r-1
                y[i] = v[idx]
                wi = [0.0] * r
                wi[idx] = 1.0
                w[i] = wi
            else:                               # station
                best_val = -1e100
                best_w = None
                for outs, probs in station_out[i]:
                    val = 0.0
                    wi = [0.0] * r
                    for out_node, g in zip(outs, probs):
                        val += g * y[out_node]
                        w_out = w[out_node]
                        for k in range(r):
                            wi[k] += g * w_out[k]
                    if val > best_val + 1e-12:   # choose the best duct
                        best_val = val
                        best_w = wi[:]          # copy
                y[i] = best_val
                w[i] = best_w
        return y[1], w[1]      # source value and its deterministic weight vector

    # -------------------------------------------------------------
    # Linear programme solver for the current set of constraints C
    # (C is a list of weight vectors, each length r)
    def solve_lp(C):
        # r is 1, 2 or 3, global
        if r == 1:
            v = [1.0]
            t = max((wk[0] for wk in C), default=0.0)
            return v, t

        if r == 2:
            best_t = float('inf')
            best_v = None

            # vertices
            for v1 in (0.0, 1.0):
                v = [v1, 1.0 - v1]
                t = max((wk[0] * v[0] + wk[1] * v[1] for wk in C), default=0.0)
                if t < best_t - 1e-12:
                    best_t, best_v = t, v[:]

            m = len(C)
            for i in range(m):
                a_i, b_i = C[i]
                for j in range(i + 1, m):
                    a_j, b_j = C[j]
                    denom = (a_i - a_j) - (b_i - b_j)
                    if abs(denom) < 1e-12:
                        continue
                    v1 = (b_j - b_i) / denom
                    if v1 < -1e-12 or v1 > 1.0 + 1e-12:
                        continue
                    v = [v1, 1.0 - v1]
                    t = max((wk[0] * v[0] + wk[1] * v[1] for wk in C), default=0.0)
                    if t < best_t - 1e-12:
                        best_t, best_v = t, v[:]

            if best_v is None:          # no constraints at all
                best_v = [0.5, 0.5]
                best_t = 0.0
            return best_v, best_t

        # r == 3
        def eval_t(v):
            v0, v1, v2 = v
            mx = 0.0
            for wk in C:
                val = wk[0] * v0 + wk[1] * v1 + wk[2] * v2
                if val > mx:
                    mx = val
            return mx

        best_t = float('inf')
        best_v = None

        # vertices of the simplex
        verts = [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]
        for v in verts:
            t = eval_t(v)
            if t < best_t - 1e-12:
                best_t, best_v = t, v[:]

        m = len(C)
        # intersections on edges (one coordinate = 0)
        for i in range(m):
            a_i, b_i, c_i = C[i]
            for j in range(i + 1, m):
                a_j, b_j, c_j = C[j]

                # v0 = 0  -> v1 = x, v2 = 1-x
                denom = (b_i - c_i) - (b_j - c_j)
                if abs(denom) > 1e-12:
                    x = (c_j - c_i) / denom
                    if -1e-12 <= x <= 1.0 + 1e-12:
                        v = [0.0, x, 1.0 - x]
                        t = eval_t(v)
                        if t < best_t - 1e-12:
                            best_t, best_v = t, v[:]

                # v1 = 0
                denom = (a_i - c_i) - (a_j - c_j)
                if abs(denom) > 1e-12:
                    x = (c_j - c_i) / denom
                    if -1e-12 <= x <= 1.0 + 1e-12:
                        v = [x, 0.0, 1.0 - x]
                        t = eval_t(v)
                        if t < best_t - 1e-12:
                            best_t, best_v = t, v[:]

                # v2 = 0
                denom = (a_i - b_i) - (a_j - b_j)
                if abs(denom) > 1e-12:
                    x = (b_j - b_i) / denom
                    if -1e-12 <= x <= 1.0 + 1e-12:
                        v = [x, 1.0 - x, 0.0]
                        t = eval_t(v)
                        if t < best_t - 1e-12:
                            best_t, best_v = t, v[:]

        # intersections of three constraints (interior point)
        for i in range(m):
            a_i, b_i, c_i = C[i]
            for j in range(i + 1, m):
                a_j, b_j, c_j = C[j]
                for k in range(j + 1, m):
                    a_k, b_k, c_k = C[k]

                    A11 = (a_i - c_i) - (a_j - c_j)
                    A12 = (b_i - c_i) - (b_j - c_j)
                    B1 = c_j - c_i

                    A21 = (a_i - c_i) - (a_k - c_k)
                    A22 = (b_i - c_i) - (b_k - c_k)
                    B2 = c_k - c_i

                    det = A11 * A22 - A12 * A21
                    if abs(det) < 1e-12:
                        continue
                    v1 = (B1 * A22 - B2 * A12) / det
                    v2 = (A11 * B2 - A21 * B1) / det
                    v3 = 1.0 - v1 - v2
                    if v1 >= -1e-12 and v2 >= -1e-12 and v3 >= -1e-12:
                        if v1 + v2 <= 1.0 + 1e-12:
                            v = [v1, v2, v3]
                            t = eval_t(v)
                            if t < best_t - 1e-12:
                                best_t, best_v = t, v[:]

        if best_v is None:          # should not happen, but keep safe
            best_v = [1.0 / r] * r
            best_t = 0.0

        # normalise just in case of tiny numerical errors
        ssum = sum(best_v)
        if abs(ssum - 1.0) > 1e-9:
            best_v = [vi / ssum for vi in best_v]

        return best_v, best_t

    # -------------------------------------------------------------
    # Cutting‑plane loop
    C = []                      # list of active weight vectors
    eps = 1e-9
    for _ in range(30):         # more than enough (r ≤ 3)
        v, t = solve_lp(C)
        y1, w_star = dp(v)
        dot = sum(w_star[i] * v[i] for i in range(r))
        if dot <= t + eps:
            answer = t
            break
        C.append(w_star[:])     # add the newly found violating weight vector

    # output as percentage
    result = answer * 100.0
    # required absolute error ≤ 1e‑6
    print("{:.10f}".format(result))

if __name__ == "__main__":
    solve()

