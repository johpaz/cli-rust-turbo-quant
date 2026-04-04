#include <bits/stdc++.h>
using namespace std;

struct Booster {
    long long price;   // P[i]
    int       typ;     // T[i]   (2..4)
    int       idx;     // original coupon index
    double    a;       // P[i] * T[i] / (T[i]-1)   for sorting
};

using ll = long long;
const ll INF64 = numeric_limits<long long>::max();

/*---------------------------------------------------------------*/
/*  Main function required by the statement                        */
vector<int> max_coupons(int A,
                        vector<int> P,
                        vector<int> T)
{
    const int N = (int)P.size();

    /* ----------- 1. split into boosters and cheap coupons ----------- */
    vector<Booster> boosters;
    boosters.reserve(N);
    vector<pair<ll,int>> cheap;          // (price , idx)
    cheap.reserve(N);

    for (int i = 0; i < N; ++i) {
        if (T[i] == 1) {
            cheap.emplace_back(P[i], i);
        } else {
            Booster b;
            b.price = P[i];
            b.typ   = T[i];
            b.idx   = i;
            b.a     = (double)P[i] * (double)T[i] / (double)(T[i] - 1);
            boosters.push_back(b);
        }
    }

    /* -------------------- 2. sort the two groups ------------------- */
    sort(boosters.begin(), boosters.end(),
         [](const Booster& x, const Booster& y){ return x.a < y.a; });

    sort(cheap.begin(), cheap.end(),
         [](const pair<ll,int>& x, const pair<ll,int>& y){ return x.first < y.first; });

    int C = (int)cheap.size();
    vector<ll> cheapPref(C + 1, 0);           // cheapPref[0] = 0
    vector<int> cheapIdx(C);
    for (int i = 0; i < C; ++i) {
        cheapPref[i+1] = cheapPref[i] + cheap[i].first;
        cheapIdx[i]    = cheap[i].second;
    }

    /* --------------------- 3. DP over boosters --------------------- */
    // DP vectors – state id is its position in the vectors
    vector<ll>  tokens;   // tokens left after the state
    vector<int> cnts;     // number of boosters taken in the state
    vector<int> prev;     // previous state id
    vector<int> idxs;     // coupon index of the last taken booster

    tokens.push_back((ll)A);   // state 0 – start
    cnts  .push_back(0);
    prev  .push_back(-1);
    idxs  .push_back(-1);

    vector<int> frontier;      // ids of non‑dominated states
    frontier.push_back(0);

    for (const Booster& b : boosters) {
        vector<int> added;            // newly created states (ids)
        added.reserve(frontier.size());

        for (int sid : frontier) {
            if (tokens[sid] >= b.price) {
                // (tokens[sid] - price) * typ   – do it in 128‑bit to avoid overflow
                __int128 tmp = (__int128)(tokens[sid] - b.price) * (ll)b.typ;
                ll newTok = (tmp > INF64) ? INF64 : (ll)tmp;

                tokens.push_back(newTok);
                cnts  .push_back(cnts[sid] + 1);
                prev  .push_back(sid);
                idxs  .push_back(b.idx);
                added.push_back((int)tokens.size() - 1);
            }
        }

        // ---- merge old and new states, keep only non‑dominated ----
        vector<int> cand = frontier;
        cand.insert(cand.end(), added.begin(), added.end());

        sort(cand.begin(), cand.end(),
             [&](int a, int b) {
                 if (tokens[a] != tokens[b]) return tokens[a] > tokens[b];
                 return cnts[a] > cnts[b];
             });

        vector<int> newFrontier;
        int bestCnt = -1;
        for (int sid : cand) {
            if (cnts[sid] > bestCnt) {
                newFrontier.push_back(sid);
                bestCnt = cnts[sid];
            }
        }
        frontier.swap(newFrontier);
    }

    /* -------------------- 4. Choose the best final state -------------------- */
    long long bestTotal = -1;
    int bestState = -1;
    int bestCheap  = 0;               // how many cheap coupons we can buy

    for (int sid : frontier) {
        ll curTok = tokens[sid];
        // number of cheap coupons affordable with curTok tokens
        int cheapCnt = (int)(upper_bound(cheapPref.begin(), cheapPref.end(), curTok)
                             - cheapPref.begin()) - 1;
        long long total = (long long)cnts[sid] + cheapCnt;
        if (total > bestTotal) {
            bestTotal = total;
            bestState = sid;
            bestCheap = cheapCnt;
        }
    }

    /* -------------------- 5. Reconstruct the answer -------------------- */
    vector<int> answer;
    // boosters (taken in reverse order)
    for (int sid = bestState; sid != 0; sid = prev[sid])
        answer.push_back(idxs[sid]);
    reverse(answer.begin(), answer.end());

    // cheap coupons – they are already sorted by price
    for (int i = 0; i < bestCheap; ++i)
        answer.push_back(cheapIdx[i]);

    return answer;
}
