#include <bits/stdc++.h>
using namespace std;

/* -----------------------------------------------------------------
   Global state that is shared between successive calls of
   send_message for the *same* test case.
   It is re‑initialised each time the first node (i == 1) of a new
   test case is processed.
   ----------------------------------------------------------------- */
namespace {
    // total number of vertices of the current test case
    static int  total_nodes = 0;
    // binary‑lifting depth
    static int  LOG = 0;
    static vector<int>            depth;               // depth[v]
    static vector<vector<int>>    up;                  // up[k][v] = 2^k‑th ancestor

    // current diameter (a,b) and its length
    static int  a = 0, b = 0, diam = 0;

    // auxiliary values that are saved at specific steps
    static int  a2 = -1;          // a after node N‑2
    static int  b2 = -1;          // b after node N‑2
    static int  b_at_N3 = -1;     // b after node N‑3
}

/* -----------------------------------------------------------------
   Helper functions working on the global structures
   ----------------------------------------------------------------- */
static void reinit(int N)
{
    total_nodes = N;
    LOG = 0;
    while ((1 << LOG) <= N) ++LOG;   // floor(log2(N)) + 1
    ++LOG;                           // a little safety margin

    depth.assign(N, 0);                       // depth[0] = 0 already
    up.assign(LOG, vector<int>(N, 0));        // all ancestors of the root = 0

    a = b = diam = 0;
    a2 = b2 = -1;
    b_at_N3 = -1;
}

/* LCA by binary lifting */
static int lca(int u, int v)
{
    if (depth[u] < depth[v]) swap(u, v);
    int diff = depth[u] - depth[v];
    for (int k = 0; diff; ++k) {
        if (diff & 1) u = up[k][u];
        diff >>= 1;
    }
    if (u == v) return u;
    for (int k = LOG - 1; k >= 0; --k) {
        if (up[k][u] != up[k][v]) {
            u = up[k][u];
            v = up[k][v];
        }
    }
    return up[0][u];
}

/* Distance between two vertices */
static int dist(int u, int v)
{
    int w = lca(u, v);
    return depth[u] + depth[v] - 2 * depth[w];
}

/* -----------------------------------------------------------------
   send_message – called while the tree is being built (sites 1 … N‑1)
   Returns 0 (no message) or an integer that will be sent to the museum.
   ----------------------------------------------------------------- */
int send_message(int N, int i, int Pi)          // C‑link for clarity
{
    // first call of a new test case → initialise the global data
    if (i == 1) {
        reinit(N);
    }

    /* ---- insert the new leaf i (parent = Pi) ---- */
    int d = depth[Pi] + 1;
    depth[i] = d;
    up[0][i] = Pi;
    for (int k = 1; k < LOG; ++k) {
        int anc = up[k - 1][i];
        up[k][i] = up[k - 1][anc];
    }

    /* ---- maintain the current diameter (a,b) ---- */
    int da = dist(i, a);
    int db = dist(i, b);
    if (da > diam) {
        b   = i;
        diam = da;
    } else if (db > diam) {
        a   = i;
        diam = db;
    }

    /* ---- send the three tiny messages (only at i = N‑3, N‑2, N‑1) ---- */
    if (total_nodes >= 4 && i == total_nodes - 3 && i >= 1) {
        /* i == N‑3 : store b after node N‑3 (store +1 because 0 would be ambiguous) */
        b_at_N3 = b;
        return b + 1;                // 1 ≤ value ≤ 20000
    }
    else if (i == total_nodes - 2 && i >= 1) {
        /* i == N‑2 : store a and the whole diameter after this step */
        a2 = a;
        b2 = b;
        return a + 1;                // a+1 (still ≤ 20000)
    }
    else if (i == total_nodes - 1 && i >= 1) {
        /* i == N‑1 : compute the flag (1 … 5) */
        int a_final = a;
        int b_final = b;
        int a2_local = a2;
        int b2_local = b2;
        int b0 = b_at_N3;            // b after node N‑3 (may be outdated)

        int flag = 1;                // default (fallback)

        if (a_final == a2_local && b_final == b2_local) {
            flag = (b2_local == total_nodes - 2) ? 2 : 1;
        }
        else if (a_final == a2_local && b_final == total_nodes - 1) {
            flag = 3;                // b replaced at the last insertion
        }
        else if (a_final == total_nodes - 1 && b_final == b2_local) {
            flag = (b2_local == total_nodes - 2) ? 5 : 4;
        }
        else {
            flag = 1;                // should never happen
        }

        return flag;                 // already in [1,5]
    }

    return 0;                        // all other vertices: no message
}

/* -----------------------------------------------------------------
   longest_path – decodes the three messages produced by send_message
   and returns a pair of vertices that realise the tree diameter.
   ----------------------------------------------------------------- */
pair<int,int> longest_path(vector<int> S)
{
    size_t N = S.size();

    if (N <= 1) return {0, 0};
    if (N == 2) return {0, 1};
    if (N == 3) return {1, 2};

    // N >= 4 : three stored values are at positions N‑3, N‑2, N‑1
    int a_msg = S[N - 2];          // a after node N‑2   (+1)
    int b_msg = S[N - 3];          // b after node N‑3   (+1)
    int flag  = S[N - 1];          // 1 … 5

    int a = a_msg - 1;             // a after node N‑2
    int b0 = b_msg - 1;            // b after node N‑3

    int u, v;
    switch (flag) {
        case 1:                    // both old endpoints survive
            u = a;
            v = b0;
            break;
        case 2:                    // b changed at N‑2, a survived
            u = a;
            v = static_cast<int>(N) - 2;
            break;
        case 3:                    // b replaced at N‑1, a survived
            u = a;
            v = static_cast<int>(N) - 1;
            break;
        case 4:                    // a replaced at N‑1, b survived (b unchanged at N‑2)
            u = b0;
            v = static_cast<int>(N) - 1;
            break;
        case 5:                    // a replaced at N‑1, b changed at N‑2
            u = static_cast<int>(N) - 2;
            v = static_cast<int>(N) - 1;
            break;
        default:                   // safety fallback – should never be hit
            u = a;
            v = b0;
    }
    return {u, v};
}
