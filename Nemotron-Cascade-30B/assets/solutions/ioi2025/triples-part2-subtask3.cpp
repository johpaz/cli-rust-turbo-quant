#include <algorithm>
#include <cassert>
#include <random>
#include <utility>
#include <vector>
#include <cstdint>
#include <unordered_map>

/* ------------------------------------------------------------------ */
/* Helper that implements the “greedy builder”.                         */
/* It returns both the constructed height vector H and the total T.    */
/* ------------------------------------------------------------------ */
static std::pair<std::vector<int>, long long>
greedy_builder(int N,
               int h0,
               int h1,
               const std::vector<int>& useful_vals,
               std::mt19937& rng)
{
    std::vector<int> H(N, 0);
    H[0] = h0;
    H[1] = h1;

    const int max_missing = N - 1;                // allowed values are 1 … max_missing
    std::vector<int> cnt(max_missing + 1, 0);
    std::vector<int> last(max_missing + 1, 0);

    long long total_T = 0;                        // may be larger than int
    const int D = (N - 1) / 2;
    const int twoD = 2 * D;

    int version = 0;                              // increased for every k

    for (int k = 2; k < N; ++k) {
        ++version;                                // new “run” for this k

        int best_cnt = 0;                         // best counter for this k
        int best_missing = 1;                     // associated missing value

        int i_start = k - twoD;
        if (i_start < 0) i_start = 0;

        for (int i = i_start; i < k; ++i) {
            int hi = H[i];
            int s = k - i;                        // s = k-i > 0

            if (hi >= s) continue;                // no candidate possible

            /* ---------- left distance candidate j = i + hi ---------- */
            int j = i + hi;
            if (j < k) {
                int a = hi;
                int b = s - hi;                    // the two distances that must appear together
                int hj = H[j];
                if (hj == a || hj == b || hj == s) {
                    if (! (a != b && hi == hj)) { // the “not (a!=b && hi==hj)” guard
                        int missing = 2 * s - hi - hj;
                        if (1 <= missing && missing <= max_missing) {
                            if (last[missing] != version) {
                                cnt[missing] = 1;
                                last[missing] = version;
                            } else {
                                ++cnt[missing];
                            }
                            int cur = cnt[missing];
                            if (cur > best_cnt) {
                                best_cnt = cur;
                                best_missing = missing;
                            }
                        }
                    }
                }
            }

            /* ---------- right distance candidate j = k - hi ---------- */
            j = k - hi;                            // right‑hand candidate, must be > i and not the same as left one
            if (j > i && j != i + hi) {
                int a = s - hi;
                int b = hi;
                int hj = H[j];
                if (hj == a || hj == b || hj == s) {
                    if (! (a != b && hi == hj)) {
                        int missing = 2 * s - hi - hj;
                        if (1 <= missing && missing <= max_missing) {
                            if (last[missing] != version) {
                                cnt[missing] = 1;
                                last[missing] = version;
                            } else {
                                ++cnt[missing];
                            }
                            int cur = cnt[missing];
                            if (cur > best_cnt) {
                                best_cnt = cur;
                                best_missing = missing;
                            }
                        }
                    }
                }
            }
        }   // end for i

        if (best_cnt > 0) {
            H[k] = best_missing;
            total_T += best_cnt;
        } else {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(useful_vals.size()) - 1);
            H[k] = useful_vals[dist(rng)];
        }
    }

    return {std::move(H), total_T};
}

/* ------------------------------------------------------------------ */
/* Main routine requested by the problem statement.                     */
/* ------------------------------------------------------------------ */
std::vector<int> construct_range(int M, int K)
{
    /* ---- 1. decide N ------------------------------------------------ */
    int N = std::max(3, M);
    if (N > M) N = M;
    // after this: 3 <= N <= M (or N = M when M < 3)

    const int max_h = N - 1;               // allowed heights are 1 … max_h

    /* ---- 2. useful fallback values ---------------------------------- */
    std::vector<int> useful_vals;
    int limit = std::min(N, 10); 
    for (int v = 1; v <= limit; ++v) useful_vals.push_back(v);
    if (max_h >= 1 && std::find(useful_vals.begin(), useful_vals.end(), max_h) == useful_vals.end())
        useful_vals.push_back(max_h);

    /* ---- 3. deterministic seeds -------------------------------------- */
    std::vector<std::pair<int, int>> seeds = {
        {1, 1}, {1, 2}, {2, 1}, {2, 2}
    };
    if (max_h >= 1) seeds.emplace_back(max_h, max_h);
    if (max_h >= 2) seeds.emplace_back(max_h - 1, max_h - 1);
    seeds.emplace_back(max_h, 1);
    seeds.emplace_back(1, max_h);

    /* ---- 4. a few random seeds (fixed RNG) -------------------------- */
    std::mt19937 rnd(123456);
    std::uniform_int_distribution<int> dist01(1, max_h);
    for (int i = 0; i < 3; ++i) {
        int h0 = dist01(rnd);
        int h1 = dist01(rnd);
        seeds.emplace_back(h0, h1);
    }

    /* ---- 5. deduplicate ---------------------------------------------- */
    std::vector<std::pair<int, int>> uniq_seeds;
    {
        std::vector<bool> seen((max_h + 1) * (max_h + 1), false); // cheap, because max_h is at most a few thousands
        for (auto &p : seeds) {
            int h0 = std::max(1, std::min(max_h, p.first));
            int h1 = std::max(1, std::min(max_h, p.second));
            int idx = (h0 - 1) * (max_h + 1) + (h1 - 1);
            if (!seen[idx]) {
                seen[idx] = true;
                uniq_seeds.emplace_back(h0, h1);
            }
        }
    }
    seeds.swap(uniq_seeds);

    /* ---- 6. try every seed, keep the best one ----------------------- */
    std::vector<int> best_H;
    long long best_T = -1;

    std::mt19937 rng(123456);              // same RNG for the fallback values
    for (const auto &seed : seeds) {
        int h0 = std::max(1, std::min(max_h, seed.first));
        int h1 = std::max(1, std::min(max_h, seed.second));

        auto [H, T] = greedy_builder(N, h0, h1, useful_vals, rng);
        if (T > best_T) {
            best_T = T;
            best_H = H;                    // copy
            if (best_T >= K) break;        // early exit when we already have enough
        }
    }

    /* ---- 7. final clamping (safety) ---------------------------------- */
    for (int &v : best_H) {
        if (v < 1) v = 1;
        else if (v > max_h) v = max_h;
    }

    if (static_cast<int>(best_H.size()) > N)
        best_H.resize(N);
    else if (static_cast<int>(best_H.size()) < N)
        best_H.resize(N, 1);               // should not happen, but just in case

    return best_H;
}
