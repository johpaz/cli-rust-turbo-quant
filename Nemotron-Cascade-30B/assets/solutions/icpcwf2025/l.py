import sys

def solve():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # coordinates of start (xc, yc) and target (xa, ya)
    xc = int(next(it))
    yc = int(next(it))
    xa = int(next(it))
    ya = int(next(it))

    # If we never need to go south, the sun never shines in our eyes.
    if yc <= ya:
        print("0.0")
        return

    low = ya          # lower y (target)
    high = yc         # higher y (start)

    # Collect y‑intervals of rectangles intersected with [low, high]
    intervals = []
    for _ in range(n):
        x1 = int(next(it))
        y1 = int(next(it))
        x2 = int(next(it))
        y2 = int(next(it))
        lo = max(y1, low)
        hi = min(y2, high)
        if lo < hi:               # positive length intersection
            intervals.append((lo, hi))

    # Merge the intervals to find the total covered length in [low, high]
    covered = 0
    if intervals:
        intervals.sort()
        cur_lo, cur_hi = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_hi:       # overlap or just touch
                if e > cur_hi:
                    cur_hi = e
            else:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = s, e
        covered += cur_hi - cur_lo

    total = high - low
    answer = total - covered
    if answer < 0:
        answer = 0.0

    # Output with sufficient precision
    print("{:.10f}".format(answer))

if __name__ == "__main__":
    solve()

