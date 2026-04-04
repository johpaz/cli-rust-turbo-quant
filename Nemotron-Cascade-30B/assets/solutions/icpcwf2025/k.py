# Your solution code here
import sys
from collections import deque

def solve() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    k = int(next(it))
    t_x = int(next(it))
    t_y = int(next(it))

    total = n + m
    adj = [[] for _ in range(total)]

    for _ in range(k):
        x = int(next(it))
        y = int(next(it))
        d = int(next(it))
        u = x - 1                     # row node
        v = n + (y - 1)               # column node
        adj[u].append((v, d))         # pot[row] - pot[col] = d
        adj[v].append((u, -d))        # pot[col] - pot[row] = -d

    pot = [None] * total            # potential values (p for rows, r for columns)
    comp = [-1] * total             # component id for each node

    comp_has_row = []               # bool per component
    comp_has_col = []               # bool per component
    comp_min_row = []               # min potential among rows in component
    comp_max_col = []               # max potential among columns in component

    INF = 10 ** 30
    comp_cnt = 0

    for node in range(total):
        if pot[node] is None:
            cur = comp_cnt
            comp_cnt += 1
            comp_has_row.append(False)
            comp_has_col.append(False)
            comp_min_row.append(INF)
            comp_max_col.append(-INF)

            dq = deque()
            pot[node] = 0
            comp[node] = cur
            dq.append(node)

            while dq:
                u = dq.popleft()
                if u < n:  # row node
                    comp_has_row[cur] = True
                    if pot[u] < comp_min_row[cur]:
                        comp_min_row[cur] = pot[u]
                else:      # column node
                    comp_has_col[cur] = True
                    if pot[u] > comp_max_col[cur]:
                        comp_max_col[cur] = pot[u]

                for v, w in adj[u]:
                    expected = pot[u] - w  # pot[v] should satisfy pot[u] - pot[v] = w
                    if pot[v] is None:
                        pot[v] = expected
                        comp[v] = cur
                        dq.append(v)
                    else:
                        if pot[v] != expected:
                            print("impossible")
                            return

            # check feasibility within component
            if comp_has_row[cur] and comp_has_col[cur]:
                if comp_min_row[cur] < comp_max_col[cur]:
                    print("impossible")
                    return

    # Determine answer for the treasure location
    row_node = t_x - 1
    col_node = n + (t_y - 1)
    comp_tx = comp[row_node]
    comp_ty = comp[col_node]

    if comp_tx == comp_ty:
        # Same component: depth is fixed
        ans = pot[row_node] - pot[col_node]
        print(ans)
    else:
        # Different components
        L = comp_min_row[comp_tx]       # min row potential in row's component
        delta_i = pot[row_node] - L     # >= 0

        H = comp_max_col[comp_ty]       # max column potential in column's component
        delta_j = H - pot[col_node]     # >= 0

        ans = delta_i + delta_j
        print(ans)


if __name__ == "__main__":
    solve()

