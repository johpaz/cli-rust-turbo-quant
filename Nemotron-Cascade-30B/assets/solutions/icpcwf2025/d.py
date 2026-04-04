# ---------------  Hornet Rover – minimal hits  -----------------
import sys
import itertools

def solve() -> None:
    input = sys.stdin.readline

    # ----- read grid -------------------------------------------------
    r, c = map(int, input().split())
    grid = []
    start_r = start_c = -1
    for i in range(r):
        line = input().rstrip('\n')
        if len(line) < c:               # safety, should not happen
            line = line.ljust(c)
        grid.append(line)
        if 'S' in line:
            start_r = i
            start_c = line.index('S')

    # ----- read move string -------------------------------------------
    s = input().strip()
    while s == '':
        s = input().strip()
    m = len(s)

    # ----- direction helpers -------------------------------------------
    dir_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]

    # ----- simulate the path, collect masks and target directions -----
    pos_r = [0] * (m + 1)
    pos_c = [0] * (m + 1)
    pos_r[0] = start_r
    pos_c[0] = start_c

    valid_masks = [0] * m          # mask[i] – valid directions before move i
    target_dirs = [0] * m          # target direction index for move i

    for i, ch in enumerate(s):
        target = dir_idx[ch]
        target_dirs[i] = target

        r0 = pos_r[i]
        c0 = pos_c[i]
        mask = 0
        for d in range(4):
            nr = r0 + dr[d]
            nc = c0 + dc[d]
            if 0 <= nr < r and 0 <= nc < c and grid[nr][nc] != '#':
                mask |= (1 << d)
        valid_masks[i] = mask

        # move to the next cell (guaranteed to be flat)
        nr = r0 + dr[target]
        nc = c0 + dc[target]
        pos_r[i + 1] = nr
        pos_c[i + 1] = nc

    # ----- all 24 permutations of the four directions -----------------
    perms = list(itertools.permutations([0, 1, 2, 3]))   # each is a tuple of length 4
    P = len(perms)   # = 24

    # ----- which permutations are compatible with each step ----------
    allowed = [[] for _ in range(m)]          # allowed[i] = list of permutation indices
    for i in range(m):
        mask = valid_masks[i]
        target = target_dirs[i]
        for p_idx, perm in enumerate(perms):
            first = -1
            for d in perm:
                if mask & (1 << d):
                    first = d
                    break
            # target is always valid, so first is never -1 here
            if first == target:
                allowed[i].append(p_idx)

    INF = 10 ** 9

    # ----- dynamic programming -----------------------------------------
    dp_prev = [INF] * P
    for p in allowed[0]:
        dp_prev[p] = 0

    for i in range(1, m):
        # find smallest and second smallest value in dp_prev
        best1 = INF
        best2 = INF
        best1_idx = -1
        for p in range(P):
            val = dp_prev[p]
            if val < best1:
                best2 = best1
                best1 = val
                best1_idx = p
            elif val < best2:
                best2 = val

        dp_curr = [INF] * P
        for p in allowed[i]:
            stay = dp_prev[p]
            if best1_idx != p:
                change = best1 + 1
            else:
                change = best2 + 1
            dp_curr[p] = stay if stay <= change else change
        dp_prev = dp_curr

    answer = min(dp_prev)
    print(answer)


if __name__ == "__main__":
    solve()

