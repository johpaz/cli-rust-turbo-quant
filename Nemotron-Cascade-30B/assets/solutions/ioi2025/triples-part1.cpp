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

