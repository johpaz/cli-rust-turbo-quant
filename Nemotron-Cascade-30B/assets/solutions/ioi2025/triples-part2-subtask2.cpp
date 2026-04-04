#include <bits/stdc++.h>

using namespace std;

long long count_triples(vector<int> H)
{
    const int N = static_cast<int>(H.size());

    long long cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0,
              cnt5 = 0, cnt6 = 0, equal_cnt = 0;

    /* ---------- assignment 1 ---------- */
    for (int i = 0; i < N; ++i) {
        int j = i + H[i];
        if (j >= N) continue;
        int b = H[j];
        int k = j + b;
        if (k >= N) continue;
        if (H[k] == H[i] + b) ++cnt1;
    }

    /* ---------- assignment 2 ---------- */
    for (int i = 0; i < N; ++i) {
        int j = i + H[i];
        if (j >= N) continue;
        int b = H[j] - H[i];
        if (b <= 0) continue;
        int k = j + b;
        if (k >= N) continue;
        if (H[k] == b) ++cnt2;
    }

    /* ---------- assignment 3 ---------- */
    for (int j = 0; j < N; ++j) {
        int i = j - H[j];
        if (i < 0) continue;
        int b = H[i];
        int k = j + b;
        if (k >= N) continue;
        if (H[k] == H[j] + b) ++cnt3;
    }

    /* ---------- assignment 5 ---------- */
    for (int j = 0; j < N; ++j) {
        int i = j - H[j];
        if (i < 0) continue;
        int b = H[i] - H[j];
        if (b <= 0) continue;
        int k = j + b;
        if (k >= N) continue;
        if (H[k] == b) ++cnt5;
    }

    /* ---------- assignment 6 ---------- */
    for (int j = 0; j < N; ++j) {
        int k = j + H[j];
        if (k >= N) continue;
        int a = H[k];
        int i = j - a;
        if (i < 0) continue;
        if (H[i] == a + H[j]) ++cnt6;
    }

    /* ---------- assignment 4 (group i+H[i] = k-H[k]) ---------- */
    // groups[T] = list of k such that k - H[k] == T   (T = “key”)
    unordered_map<int, vector<int>> groups;
    groups.reserve(N * 2);
    for (int k = 0; k < N; ++k) {
        int T = k - H[k];
        groups[T].push_back(k);            // k grows, so each vector stays sorted
    }

    for (int i = 0; i < N; ++i) {
        int T = i + H[i];
        auto it = groups.find(T);
        if (it == groups.end()) continue;   // no k with the needed key

        const vector<int>& ks = it->second;
        // first k that is > i
        auto posIter = upper_bound(ks.begin(), ks.end(), i);
        int H_i = H[i];

        for (auto kIt = posIter; kIt != ks.end(); ++kIt) {
            int k = *kIt;
            int j = k - H_i;
            if (j >= N) break;               // larger k → larger j, so we can stop
            // i < j < k holds automatically for the chosen ks
            if (H[j] == k - i) ++cnt4;
        }
    }

    long long total = cnt1 + cnt2 + cnt3 + cnt4 + cnt5 + cnt6;

    /* ---------- triples with equal distances (a = b) ----------
       These triples were counted twice (once in the main total and
       once in the equal‑distance special case), therefore we subtract
       them once at the end.                                         */
    for (int i = 0; i < N; ++i) {
        int a = H[i];

        // case 1 : distance = a   (i → j → k, with j = i+a, k = i+2a)
        int j = i + a;
        if (j < N) {
            int k = i + 2 * a;
            if (k < N) {
                bool ok = (H[j] == a && H[k] == 2 * a) ||
                          (H[j] == 2 * a && H[k] == a);
                if (ok) ++equal_cnt;
            }
        }

        // case 2 : distance = 2a   (i → j (a) → k (2a), but we treat a = H[i]/2)
        if (a % 2 == 0) {
            int a2 = a / 2;
            if (a2 > 0) {
                j = i + a2;
                int k = i + a;                     // i + 2*a2 = i + a
                if (j < N && k < N) {
                    if (H[j] == a2 && H[k] == a2) ++equal_cnt;
                }
            }
        }
    }

    return total - equal_cnt;
}



/* -------------------------------------------------------------
   2.  greedy_builder – builds a candidate array from (h0,h1)
   ------------------------------------------------------------- */
static vector<int> greedy_builder(int N, int h0, int h1,
                                  int D, int max_h)
{
    vector<int> H(N, 1);
    H[0] = h0;
    H[1] = h1;

    vector<int> cnt(max_h + 1, 0);
    vector<int> last(max_h + 1, 0);
    int version = 0;
    const int twoD = 2 * D;

    for (int k = 2; k < N; ++k) {
        ++version;
        int best_cnt = 0;
        int best_missing = 1;
        int i_start = k - twoD;
        if (i_start < 0) i_start = 0;

        for (int i = i_start; i < k; ++i) {
            int hi = H[i];
            int s = k - i;
            if (hi >= s) continue;

            // ----- left candidate j = i + hi -----
            int j = i + hi;
            if (j < k) {
                int a = hi;
                int b = s - hi;
                int hj = H[j];
                bool ok = (hj == a || hj == b || hj == s) &&
                          !(a != b && hi == hj);
                if (ok) {
                    int missing = 2 * s - hi - hj;
                    if (1 <= missing && missing <= max_h) {
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

            // ----- right candidate j2 = k - hi -----
            int j2 = k - hi;
            if (j2 > i && j2 != i + hi) {
                int a = s - hi;
                int b = hi;
                int hj = H[j2];
                bool ok = (hj == a || hj == b || hj == s) &&
                          !(a != b && hi == hj);
                if (ok) {
                    int missing = 2 * s - hi - hj;
                    if (1 <= missing && missing <= max_h) {
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
        }

        H[k] = (best_cnt > 0) ? best_missing : 1;
    }
    return H;
}

/* -------------------------------------------------------------
   3.  construct_range – the whole construction routine
   ------------------------------------------------------------- */
vector<int> construct_range(int M, int K)
{
    const int N = max(3, M);                 // we always use exactly N = M (M>=3 in tests)
    const int max_h = N - 1;
    int D = (N - 1) / 2;
    if (D == 0) D = 1;

    long long best_T = -1;
    vector<int> best_H;

    /* ----- helper: evaluate a candidate and keep the best one ----- */
    auto evaluate = [&](const vector<int>& H) {
        long long T = count_triples(H);
        if (T > best_T) {
            best_T = T;
            best_H = H;
            if (best_T >= K) throw runtime_error("enough triples");
        }
    };

    /* ----- deterministic patterns ----- */
    auto add_pattern = [&](const vector<int>& pat) {
        int repeats = N / static_cast<int>(pat.size()) + 1;
        vector<int> H;
        H.reserve(N);
        for (int i = 0; i < repeats; ++i) H.insert(H.end(), pat.begin(), pat.end());
        H.resize(N);
        evaluate(H);
    };

    add_pattern({1, 1, 2});
    add_pattern({1, 2, 1});
    add_pattern({2, 1, 1});

    // a2 pattern – 4 on positions whose block‑index %3 ==0
    vector<int> H(N);
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) {                     // even index
            int pos = i / 2;
            H[i] = (pos % 3 == 0) ? 4 : 2;
        } else {                               // odd index
            int pos = (i - 1) / 2;
            H[i] = (pos % 3 == 0) ? 4 : 2;
        }
    }
    evaluate(H);

    /* ----- start pairs for the greedy builder ----- */
    unordered_set<long long> start_pairs;     // encode pair (h0,h1) in a 64‑bit key
    auto encode = [&](int a, int b) -> long long { return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b); };

    int limit = min(20, max_h);
    for (int h0 = 1; h0 <= limit; ++h0)
        for (int h1 = 1; h1 <= limit; ++h1)
            start_pairs.insert(encode(h0, h1));

    start_pairs.insert(encode(max_h, max_h));
    start_pairs.insert(encode(max_h, 1));
    start_pairs.insert(encode(1, max_h));
    if (max_h >= 2) {
        start_pairs.insert(encode(max_h - 1, max_h - 1));
        start_pairs.insert(encode(max_h - 2, max_h - 2));
    }

    mt19937_64 rnd(123456789ULL);
    for (int i = 0; i < 30; ++i) {
        int h0 = uniform_int_distribution<int>(1, max_h)(rnd);
        int h1 = uniform_int_distribution<int>(1, max_h)(rnd);
        start_pairs.insert(encode(h0, h1));
    }

    vector<pair<int,int>> start_list;
    start_list.reserve(start_pairs.size());
    for (auto key : start_pairs) {
        int h0 = static_cast<int>(key >> 32);
        int h1 = static_cast<int>(key & 0xffffffff);
        start_list.emplace_back(h0, h1);
    }
    shuffle(start_list.begin(), start_list.end(),
            mt19937_64(987654321ULL));

    /* ----- greedy builder for every start pair ----- */
    try {
        for (auto [h0, h1] : start_list) {
            int h0c = max(1, min(max_h, h0));
            int h1c = max(1, min(max_h, h1));
            vector<int> cand = greedy_builder(N, h0c, h1c, D, max_h);
            evaluate(cand);
        }
    } catch (const runtime_error&) {
        // we already hit K – just keep the current best
    }

    /* ----- hill climbing if needed ----- */
    if (best_T < K) {
        vector<int> cur_H = best_H;
        long long cur_T = best_T;

        // values we are allowed to change to (small numbers + the maximal one)
        vector<int> cand_vals;
        int small_limit = min(N, 30);
        for (int v = 1; v <= small_limit; ++v) cand_vals.push_back(v);
        if (max_h > small_limit) cand_vals.push_back(max_h);
        sort(cand_vals.begin(), cand_vals.end());
        cand_vals.erase(unique(cand_vals.begin(), cand_vals.end()), cand_vals.end());

        mt19937_64 rnd_hc(987654321ULL);
        uniform_int_distribution<int> idx_dist(0, N - 1);
        uniform_int_distribution<int> val_dist(0, static_cast<int>(cand_vals.size()) - 1);

        for (int step = 0; step < 20000 && best_T < K; ++step) {
            int i = idx_dist(rnd_hc);
            int new_val = cand_vals[val_dist(rnd_hc)];
            if (new_val == cur_H[i]) continue;

            int old = cur_H[i];
            cur_H[i] = new_val;
            long long new_T = count_triples(cur_H);
            if (new_T > cur_T) {
                cur_T = new_T;
                if (new_T > best_T) {
                    best_T = new_T;
                    best_H = cur_H;
                    if (best_T >= K) break;
                }
            } else {
                cur_H[i] = old;   // revert
            }
        }
    }

    /* ----- final safety clamp ----- */
    vector<int> out_H(N);
    for (int i = 0; i < N; ++i) {
        int v = best_H[i];
        if (v < 1) v = 1;
        if (v > max_h) v = max_h;
        out_H[i] = v;
    }
    out_H.resize(N);
    return out_H;
}

