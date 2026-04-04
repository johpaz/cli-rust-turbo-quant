#include <bits/stdc++.h>
using namespace std;

/* ------------------------------------------------------------
   Greedy construction of one block
   ------------------------------------------------------------ */
static pair<vector<int>, long long> build_block(int B, int D)
{
    /* H[0] … H[B‑1] – heights, 1 ≤ H[i] ≤ B‑1 */
    vector<int> H(B, 0);
    H[0] = 1;
    H[1] = 1;                     // any value 1 … B‑1 would work

    const int max_allowed = B - 1;
    int max_missing = 4 * D;
    if (max_missing > max_allowed) max_missing = max_allowed;
    if (max_missing < 1)            max_missing = 1;   // at least 1

    /* versioned counters for “missing” values (size = max_missing+1)   */
    vector<int> cnt(max_missing + 1, 0);
    vector<int> last(max_missing + 1, 0);
    int version = 0;

    long long total = 0;          // total number of triples created in the block
    const int twoD = 2 * D;

    for (int k = 2; k < B; ++k) {
        ++version;                // start a fresh counting round for this k
        int best_cnt = 0;
        int best_missing = 1;     // default – will be overwritten if we find a better one

        /* the distance from the leftmost i we have to examine.
           i must satisfy k‑i ≤ 2·D, i.e. i ≥ k‑2·D                */
        int i_start = k - twoD;
        if (i_start < 0) i_start = 0;

        for (int i = i_start; i < k; ++i) {
            int hi = H[i];
            int s  = k - i;                // distance i → k
            if (hi >= s) continue;         // hi must be smaller than the distance

            /* ----------------------------------------------------
               left candidate : j = i + hi   (the point that would be
               exactly hi units to the right of i)
               ---------------------------------------------------- */
            int j = i + hi;
            if (j < k) {
                int a = hi;
                int b = s - hi;
                int hj = H[j];
                if ( (hj == a || hj == b || hj == s) &&
                     !(a != b && hi == hj) )                // avoid the “a≠b && hi==hj” case
                {
                    int missing = 2 * s - hi - hj;          // how many triples this choice would give
                    if (1 <= missing && missing <= max_missing) {
                        if (last[missing] != version) {
                            cnt[missing] = 1;
                            last[missing] = version;
                        } else {
                            ++cnt[missing];
                        }
                        if (cnt[missing] > best_cnt) {
                            best_cnt = cnt[missing];
                            best_missing = missing;
                        }
                    }
                }
            }

            /* ----------------------------------------------------
               right candidate : j2 = k – hi   (the point that would be
               exactly hi units to the left of k)
               ---------------------------------------------------- */
            int j2 = k - hi;
            if (j2 > i && j2 != i + hi) {      // must be a *different* position
                int a = s - hi;
                int b = hi;
                int hj = H[j2];
                if ( (hj == a || hj == b || hj == s) &&
                     !(a != b && hi == hj) )
                {
                    int missing = 2 * s - hi - hj;
                    if (1 <= missing && missing <= max_missing) {
                        if (last[missing] != version) {
                            cnt[missing] = 1;
                            last[missing] = version;
                        } else {
                            ++cnt[missing];
                        }
                        if (cnt[missing] > best_cnt) {
                            best_cnt = cnt[missing];
                            best_missing = missing;
                        }
                    }
                }
            }
        }   // end for i

        if (best_cnt > 0) {
            H[k] = best_missing;
            total += best_cnt;
        } else {
            H[k] = 1;               // fall‑back – the smallest admissible value
        }
    }   // end for k

    return {H, total};
}

/* ------------------------------------------------------------
   Whole range construction
   ------------------------------------------------------------ */
vector<int> construct_range(int M, int K)
{
    int N = M;
    if (N < 3) N = 3; 

    /* block size – a good compromise between speed and strength */
    int B = min(N, 25000);

    /* window size – large enough to create many triples but not too large */
    int D = (B - 1) / 2;
    if (D > 4000) D = 4000;
    if (D == 0)   D = 1;

    /* Build one block and repeat it to fill the whole range   */
    auto [block, /*unused*/] = build_block(B, D);

    int reps = N / B;
    int rest = N % B;

    vector<int> H;
    H.reserve(N);
    for (int r = 0; r < reps; ++r) H.insert(H.end(), block.begin(), block.end());
    H.insert(H.end(), block.begin(), block.begin() + rest);   // add the leftover part
    H.resize(N);                                              // ensure length exactly N

    /* At this point every H[i] already satisfies
       1 ≤ H[i] ≤ B‑1 ≤ N‑1, i.e. the required bounds.            */
    return H;
}
