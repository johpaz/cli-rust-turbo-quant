# Solution for the "Imperial Chance & Play Casino" interactive problem.
# The idea is to repeatedly rotate each wheel (except the first) by +1 step.
# If this rotation increases the number of distinct symbols (k), we revert it.
# If it does not increase k (new_k <= old_k), we keep the rotation.
# When a rotation makes k smaller we have eliminated a unique symbol.
# When all wheels have been processed and none of them kept a rotation
# (i.e. every +1 caused an increase and was reverted), we rotate wheel 1
# by +1 to change the configuration – this guarantees progress.
# Each pass uses at most 2·(n‑1)+1 operations, and the number of passes
# needed to reduce k from at most n to 1 is at most n, so the total
# number of operations stays well below the 10000 limit.

import sys

def solve() -> None:
    # read the number of wheels (and symbols per wheel)
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())

    # start the interaction: get the current number of distinct symbols
    cur_k = int(input())          # reads the initial k (k > 1)

    # if the machine already shows a single symbol we are done (should not happen)
    if cur_k <= 1:
        return

    # main loop: keep making progress until all wheels show the same symbol
    while cur_k > 1:
        any_keep = False

        # try to rotate every wheel (except the reference wheel 1) by +1
        for i in range(2, n + 1):
            old_k = cur_k
            # rotate wheel i by +1 step
            print(i, 1)
            new_k = int(input())
            cur_k = new_k
            
            # if we have already won, exit immediately
            if cur_k == 1:
                return

            if new_k <= old_k:
                # the rotation didn't increase the number of distinct symbols;
                # we keep the rotation and remember that we made progress
                any_keep = True
            else:
                # rotation increased k → revert it
                # after the revert cur_k is set to the value after undoing the +1 step
                print(i, -1)
                cur_k = int(input())

        # If in this whole pass we never kept a rotation (all wheels caused an increase)
        # we rotate wheel 1 by +1 to change the configuration; this may look like an
        # increase, but we keep it anyway.  It forces the system to evolve.
        if not any_keep:
            print(1, 1)
            cur_k = int(input())
            if cur_k == 1:
                return

    # When cur_k becomes 1 the judge will stop giving answers; we simply exit.

if __name__ == "__main__":
    solve()
