import sys

def solve() -> None:
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))

    N2 = 2 * n                     # vertices: 0..n-1  -> L(i),  n..2n-1 -> R(i)

    parent = list(range(N2))

    # component of each city (left and right)
    comp_a = [i for i in range(n)]          # L(i)
    comp_b = [i + n for i in range(n)]      # R(i)

    # for every component keep the set of cities that have it
    comp_cities = [set() for _ in range(N2)]
    for i in range(n):
        comp_cities[i].add(i)          # L(i)
        comp_cities[i + n].add(i)      # R(i)

    # unordered pair -> number of cities having exactly this pair
    pair_cnt = {}
    for i in range(n):
        a = i
        b = i + n
        if a < b:
            key = (a, b)
        else:
            key = (b, a)
        pair_cnt[key] = 1               # each city has a distinct pair at the start

    # helpers
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def choose2(x: int) -> int:
        return x * (x - 1) // 2

    sum_cnt = 0           # Σ choose2(cnt[comp])
    sum_pair = 0          # Σ choose2(pair_cnt[pair])

    out = []

    for _ in range(m):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        u = a                     # L(a)
        v = b + n                 # R(b)

        ra = find(u)
        rb = find(v)

        if ra == rb:
            ans = sum_cnt - sum_pair
            out.append(str(ans))
            continue

        # always merge the smaller component into the larger one
        if len(comp_cities[ra]) > len(comp_cities[rb]):
            ra, rb = rb, ra

        size_a = len(comp_cities[ra])
        size_b = len(comp_cities[rb])

        # number of cities that have both components ra and rb
        key_ab = (ra, rb) if ra < rb else (rb, ra)
        overlap = pair_cnt.get(key_ab, 0)

        new_size = size_a + size_b - overlap
        sum_cnt += choose2(new_size) - (choose2(size_a) + choose2(size_b))

        # move all cities from the smaller component
        cities_to_move = list(comp_cities[ra])
        for city in cities_to_move:
            a_comp = comp_a[city]
            b_comp = comp_b[city]

            # both components are ra ?
            if a_comp == ra and b_comp == ra:
                a_comp = b_comp = rb
            elif a_comp == ra:
                other = b_comp
                if a_comp != b_comp:
                    key_old = (ra, other) if ra < other else (other, ra)
                    cnt = pair_cnt[key_old]
                    sum_pair -= (cnt - 1)          # choose2(cnt-1) - choose2(cnt)
                    if cnt == 1:
                        del pair_cnt[key_old]
                    else:
                        pair_cnt[key_old] = cnt - 1
                a_comp = rb
            elif b_comp == ra:
                other = a_comp
                if a_comp != b_comp:
                    key_old = (ra, other) if ra < other else (other, ra)
                    cnt = pair_cnt[key_old]
                    sum_pair -= (cnt - 1)
                    if cnt == 1:
                        del pair_cnt[key_old]
                    else:
                        pair_cnt[key_old] = cnt - 1
                b_comp = rb
            else:
                # should not happen
                pass

            comp_a[city] = a_comp
            comp_b[city] = b_comp

            if a_comp != b_comp:
                key_new = (a_comp, b_comp) if a_comp < b_comp else (b_comp, a_comp)
                cnt = pair_cnt.get(key_new, 0)
                sum_pair += cnt                     # choose2(cnt+1) - choose2(cnt) = cnt
                pair_cnt[key_new] = cnt + 1

            comp_cities[rb].add(city)

        comp_cities[ra].clear()
        parent[ra] = rb

        ans = sum_cnt - sum_pair
        out.append(str(ans))

    sys.stdout.write("\n".join(out))

if __name__ == "__main__":
    solve()

