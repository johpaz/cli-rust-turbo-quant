import sys

def solve() -> None:
    input = sys.stdin.readline
    t = int(input())
    for _ in range(t):
        n, m = map(int, input().split())
        # lower bound for each plant
        LB = [1] * (m + 1)                 # 1‑based, default 1 (can be placed at pot 1)
        cat_cnt = [0] * (m + 1)            # number of cats that want to stop at pot p
        pot_counts = [None] * (m + 1)      # for each pot: dict {plant -> how many cats like it}

        for _ in range(n):
            parts = input().split()
            p = int(parts[0])
            k = int(parts[1])
            likes = list(map(int, parts[2:]))
            cat_cnt[p] += 1
            d = pot_counts[p]
            if d is None:
                d = {}
                pot_counts[p] = d
            for typ in likes:
                if LB[typ] < p:
                    LB[typ] = p
                d[typ] = d.get(typ, 0) + 1

        # ---- global lower‑bound feasibility ----
        cnt_by_lb = [0] * (m + 2)
        for typ in range(1, m + 1):
            cnt_by_lb[LB[typ]] += 1
        total = 0
        possible = True
        for p in range(1, m + 1):
            total += cnt_by_lb[p]
            if total < p:
                possible = False
                break
        if not possible:
            print("no")
            continue

        # ---- build intersections for restricted pots ----
        restricted = []          # list of (p, list_of_plants_sorted_by LB descending)
        ok = True
        for p in range(1, m + 1):
            if cat_cnt[p] == 0:
                continue
            d = pot_counts[p]
            inter = []
            need = cat_cnt[p]
            if d:
                for typ, cnt in d.items():
                    if cnt == need and LB[typ] <= p:
                        inter.append(typ)
            # d can be None only when there are cats but no liked plants – impossible
            if not inter:
                ok = False
                break
            inter.sort(key=lambda x: LB[x], reverse=True)   # descending LB
            restricted.append((p, inter))

        if not ok:
            print("no")
            continue

        # ---- match restricted pots (right to left) ----
        used = [False] * (m + 1)          # plant already placed ?
        restricted.sort(key=lambda x: -x[0])   # decreasing p
        ok = True
        for _, plants in restricted:
            placed = False
            for typ in plants:
                if not used[typ]:
                    used[typ] = True
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            print("no")
            continue

        # ---- unrestricted pots ----
        unrestricted = [p for p in range(1, m + 1) if cat_cnt[p] == 0]
        remaining = [typ for typ in range(1, m + 1) if not used[typ]]
        remaining.sort(key=lambda x: LB[x])   # ascending LB

        idx = 0
        for q in unrestricted:
            if idx >= len(remaining) or LB[remaining[idx]] > q:
                ok = False
                break
            idx += 1

        print("yes" if ok else "no")

if __name__ == "__main__":
    solve()

