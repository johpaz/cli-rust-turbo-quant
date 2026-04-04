#include <vector>
#include <algorithm>   // for std::min, std::max
#include <cstddef>     // for std::size_t
#include <cassert>

static std::vector<int> greedy_block(int B, int D)
{
    /*  H will hold the heights for the block, 0‑based indices               */
    std::vector<int> H(B, 0);

    const int max_missing = B - 1;                 // allowed values: 1 … max_missing
    const int twoD = 2 * D;

    /* temporary counters, version stamps and the version counter            */
    std::vector<int> cnt(max_missing + 1, 0);
    std::vector<int> last(max_missing + 1, 0);
    int version = 0;

    for (int k = 0; k < B; ++k) {
        ++version;                                 // start a new “window”

        int i_start = k - twoD;
        if (i_start < 0) i_start = 0;

        int best_cnt = 0;
        int best_missing = 1;                      // any legal value if we cannot improve

        for (int i = i_start; i < k; ++i) {
            int hi = H[i];
            int s = k - i;                          // distance i → k
            if (hi >= s) continue;                  // hi cannot be used

            /* ----------- left side: j = i + hi -------------------------- */
            int j = i + hi;
            if (j < k) {
                int hj = H[j];
                int a = hi;
                int b = s - hi;

                bool ok = (hj == a) || (hj == b) || (hj == s);
                ok = ok && !(a != b && hi == hj);  // forbid the symmetric case

                if (ok) {
                    int miss = 2 * s - hi - hj;    // the “missing” value
                    if (miss >= 1 && miss <= max_missing) {
                        if (last[miss] != version) {
                            cnt[miss] = 1;
                            last[miss] = version;
                        } else {
                            ++cnt[miss];
                        }

                        if (cnt[miss] > best_cnt) {
                            best_cnt = cnt[miss];
                            best_missing = miss;
                        }
                    }
                }
            }

            /* ----------- right side: j = k - hi ------------------------- */
            int j2 = k - hi;
            if (j2 > i && j2 != i + hi) {
                int hj = H[j2];
                int a = s - hi;
                int b = hi;

                bool ok = (hj == a) || (hj == b) || (hj == s);
                ok = ok && !(a != b && hi == hj);  // same symmetric check

                if (ok) {
                    int miss = 2 * s - hi - hj;
                    if (miss >= 1 && miss <= max_missing) {
                        if (last[miss] != version) {
                            cnt[miss] = 1;
                            last[miss] = version;
                        } else {
                            ++cnt[miss];
                        }

                        if (cnt[miss] > best_cnt) {
                            best_cnt = cnt[miss];
                            best_missing = miss;
                        }
                    }
                }
            }
        }   // end of inner loop over i

        /* write the height for position k */
        if (best_cnt > 0)
            H[k] = best_missing;
        else
            H[k] = 1;                               // any legal value is fine
    }

    return H;
}

/* ------------------------------------------------------------------ */
/*  Public API – exactly the signature required by the task            */
/* ------------------------------------------------------------------ */
std::vector<int> construct_range(int M, int K)
{
    (void)K;   // avoid unused‑parameter warning

    const int N = std::max(3, M);               // final length of the range
    const int B = std::min(N, 20000);           // block size (greedy works better with a limit)
    const int D = std::min(4000, (B - 1) / 2);   // half‑window size

    std::vector<int> block = greedy_block(B, D);

    const int repeats = N / B;
    const int rest    = N % B;

    std::vector<int> H;
    H.reserve(N);
    for (int r = 0; r < repeats; ++r)
        H.insert(H.end(), block.begin(), block.end());
    H.insert(H.end(), block.begin(), block.begin() + rest);   // the possible tail

    /* Clamp to the allowed interval [1, N‑1] – this is a harmless safety net.   */
    const int max_allowed = N - 1;
    for (int &v : H) {
        if (v < 1) v = 1;
        else if (v > max_allowed) v = max_allowed;
    }

    H.resize(N);   // just in case something went wrong (should already be N)
    return H;
}
