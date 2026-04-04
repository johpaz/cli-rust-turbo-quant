#include <vector>
#include <algorithm>   // std::min, std::max

std::vector<int> construct_range(int M, int /* K */)
{
    // -----------------------------------------------------------------
    // 1) decide the final length N and the maximal allowed height
    // -----------------------------------------------------------------
    int N = std::max(3, M);
    if (N > M)                // this makes N = M when M < 3
        N = M;

    int max_allowed = N - 1;               // heights are in [1 .. max_allowed]

    // radius of the sliding window
    int D = std::min(3500, (N - 1) / 2);
    int twoD = 2 * D;

    // -----------------------------------------------------------------
    // 2) data structures
    // -----------------------------------------------------------------
    std::vector<int> H(N, 0);
    H[0] = 1;
    if (N > 1)
        H[1] = 2;                         // second peak gets a different height

    // versioned counters for the “missing’’ height
    std::vector<int> cnt(max_allowed + 1, 0);
    std::vector<int> last(max_allowed + 1, 0);
    int version = 0;                      // increased for every k
    int total = 0;                        // lower bound on the number of triples

    // -----------------------------------------------------------------
    // 3) greedy construction
    // -----------------------------------------------------------------
    for (int k = 2; k < N; ++k) {
        ++version;
        int i_start = k - twoD;
        if (i_start < 0) i_start = 0;

        int best_cnt = 0;
        int best_missing = 1;             // default if nothing is found

        // scan the recent window [i_start, k)
        for (int i = i_start; i < k; ++i) {
            int hi = H[i];
            int s  = k - i;                // distance i → k
            if (hi >= s) continue;         // cannot be used, skip

            // -------------------------------------------------------------
            // left candidate: j = i + hi
            // -------------------------------------------------------------
            int j = i + hi;                // always < k because hi < s
            int hj = H[j];
            int a = hi;
            int b = s - hi;
            // (hj == a or hj == b or hj == s) and not (a != b && hi == hj)
            if ((hj == a || hj == b || hj == s) &&
                !(a != b && hi == hj)) {
                int missing = 2 * s - hi - hj;
                if (1 <= missing && missing <= max_allowed) {
                    if (last[missing] != version) {
                        cnt[missing] = 1;
                        last[missing] = version;
                    } else {
                        ++cnt[missing];
                    }
                    int cur = cnt[missing];
                    if (cur > best_cnt) {
                        best_cnt   = cur;
                        best_missing = missing;
                    }
                }
            }

            // -------------------------------------------------------------
            // right candidate: j2 = k - hi   (different from the left one)
            // -------------------------------------------------------------
            int j2 = k - hi;
            if (j2 > i && j2 != i + hi) {
                hj = H[j2];
                a = s - hi;
                b = hi;
                if ((hj == a || hj == b || hj == s) &&
                    !(a != b && hi == hj)) {
                    int missing = 2 * s - hi - hj;
                    if (1 <= missing && missing <= max_allowed) {
                        if (last[missing] != version) {
                            cnt[missing] = 1;
                            last[missing] = version;
                        } else {
                            ++cnt[missing];
                        }
                        int cur = cnt[missing];
                        if (cur > best_cnt) {
                            best_cnt   = cur;
                            best_missing = missing;
                        }
                    }
                }
            }
        } // end for i

        // choose the best missing height (or 1 if nothing helped)
        if (best_cnt > 0) {
            H[k] = best_missing;
            total += best_cnt;
        } else {
            H[k] = 1;
        }
    }

    // -----------------------------------------------------------------
    // 4) safety clamp (normally already satisfied)
    // -----------------------------------------------------------------
    for (int &v : H) {
        if (v < 1)          v = 1;
        else if (v > max_allowed) v = max_allowed;
    }

    return H;          // the constructed range
}
