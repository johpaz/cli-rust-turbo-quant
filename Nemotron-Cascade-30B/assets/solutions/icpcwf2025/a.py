#  Skew‑heap reconstruction
#  compatible with the proof above
import sys


def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    left = [0] * (n + 1)
    right = [0] * (n + 1)
    has_parent = [False] * (n + 1)

    for i in range(1, n + 1):
        l = int(next(it))
        r = int(next(it))
        left[i] = l
        right[i] = r
        if l:
            has_parent[l] = True
        if r:
            has_parent[r] = True

    # root must be the only node without a parent
    root = 0
    for i in range(1, n + 1):
        if not has_parent[i]:
            root = i
            break
    if root != 1:
        print("impossible")
        return

    # heap order and no right‑only node
    for i in range(1, n + 1):
        if left[i] and not (i < left[i]):
            print("impossible")
            return
        if right[i] and not (i < right[i]):
            print("impossible")
            return
        if right[i] and left[i] == 0:
            print("impossible")
            return

    # -----------------------------------------------------------------
    # one pop operation, mode = 1 → take larger, mode = 0 → take smaller
    def pop_one(root_idx, mode, L, R):
        stack = []
        cur = root_idx
        while True:
            if R[cur] != 0:
                stack.append(cur)
                cur = L[cur]
                continue

            # right side empty -> two possible candidates
            lc = L[cur]
            leaf = None
            if lc != 0 and L[lc] == 0 and R[lc] == 0:
                leaf = lc

            if leaf is not None:
                # both cur and leaf are candidates
                if (mode == 1 and cur >= leaf) or (mode == 0 and cur <= leaf):
                    popped = cur
                    new_root = lc
                else:
                    popped = leaf
                    new_root = cur
                    # delete the leaf
                    L[cur] = 0
            else:
                popped = cur
                new_root = lc          # may be 0

            # unwind the stored path
            while stack:
                p = stack.pop()
                rp = R[p]               # original right sibling
                L[p] = rp
                R[p] = new_root
                new_root = p
            return popped, new_root

    # -----------------------------------------------------------------
    # smallest insertion permutation  -> maximal pop (mode = 1)
    Lmax = left[:]
    Rmax = right[:]
    cur = 1
    del_seq = []
    for _ in range(n):
        v, cur = pop_one(cur, 1, Lmax, Rmax)
        del_seq.append(v)
    smallest = list(reversed(del_seq))

    # largest insertion permutation  -> minimal pop (mode = 0)
    Lmin = left[:]
    Rmin = right[:]
    cur = 1
    del_seq = []
    for _ in range(n):
        v, cur = pop_one(cur, 0, Lmin, Rmin)
        del_seq.append(v)
    largest = list(reversed(del_seq))

    # output
    print(' '.join(map(str, smallest)))
    print(' '.join(map(str, largest)))


if __name__ == "__main__":
    solve()

