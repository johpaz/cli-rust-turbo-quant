#include <bits/stdc++.h>
using namespace std;

/* ----------------------------------------------------------------------
   create_map
   ---------------------------------------------------------------------- */
vector<vector<int>> create_map(int N, int M,
                               vector<int> A, vector<int> B)
{
    /* --------------------------------------------------------------
       0‑based adjacency matrix and list of edges
       -------------------------------------------------------------- */
    vector<vector<char>> adj(N, vector<char>(N, 0));
    vector<pair<int,int>> edges;
    edges.reserve(M);

    for (int k = 0; k < M; ++k) {
        int u = A[k] - 1;                 // to 0‑based
        int v = B[k] - 1;
        adj[u][v] = adj[v][u] = 1;
        edges.emplace_back(u, v);
    }
    const int M_edges = static_cast<int>(edges.size());

    /* --------------------------------------------------------------
       edge_id[u][v]  (only for u < v)  →  index of the edge in `edges`
       -------------------------------------------------------------- */
    vector<vector<int>> edge_id(N, vector<int>(N, -1));
    for (int idx = 0; idx < M_edges; ++idx) {
        int u = edges[idx].first;
        int v = edges[idx].second;
        if (u > v) swap(u, v);
        edge_id[u][v] = idx;
    }

    /* --------------------------------------------------------------
       neighbour_mask[i] – bit mask of colours that may be placed
       next to a cell whose colour is i (loops + the 4‑neighbours)
       -------------------------------------------------------------- */
    vector<uint64_t> neighbour_mask(N, 0);
    for (int i = 0; i < N; ++i) {
        uint64_t mask = 1ULL << i;                // the colour itself
        for (int j = 0; j < N; ++j)
            if (adj[i][j]) mask |= (1ULL << j);
        neighbour_mask[i] = mask;
    }
    const uint64_t all_mask = (N == 64) ? ~0ULL : ((1ULL << N) - 1ULL);

    /* --------------------------------------------------------------
       board size K (smallest board that could possibly work)
       -------------------------------------------------------------- */
    int K0 = static_cast<int>(sqrt((double)N));
    while (K0 * K0 < N) ++K0;                     // ceil(sqrt(N))

    int Kedge = 0;
    while (2 * Kedge * (Kedge - 1) < M_edges) ++Kedge;

    int K = max(K0, Kedge);
    K = max(K, 2 * N);                            // a bit of spare space
    if (K > 240) K = 240;                         // safety cap

    /* --------------------------------------------------------------
       random generator – used for the “several random attempts”
       -------------------------------------------------------------- */
    std::mt19937 rng(
        static_cast<unsigned>(chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count()));

    const int MAX_ATTEMPTS = 500;

    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        /* ----- fresh board for this attempt ---------------------- */
        vector<vector<int>> grid(K, vector<int>(K, -1));
        uint64_t used_colours = 0;                // bit mask of colours already used
        vector<char> covered(M_edges, 0);
        int covered_cnt = 0;
        bool ok = true;

        /* ----- greedy filling of the K×K board ------------------- */
        for (int i = 0; i < K && ok; ++i) {
            for (int j = 0; j < K; ++j) {
                uint64_t allowed = all_mask;      // colours allowed by neighbours

                if (i > 0) {
                    int top = grid[i-1][j];
                    allowed &= neighbour_mask[top];
                }
                if (j > 0) {
                    int left = grid[i][j-1];
                    allowed &= neighbour_mask[left];
                }

                if (allowed == 0) {               // dead end → restart whole board
                    ok = false;
                    break;
                }

                uint64_t missing = all_mask & ~used_colours;
                uint64_t missing_allowed = allowed & missing;

                /* ----- pick the colour with the best gain ---------- */
                int best_gain = -1;
                vector<int> candidates;

                uint64_t mask = allowed;
                while (mask) {
                    uint64_t lsb = mask & -mask;
                    int c = __builtin_ctzll(lsb);   // colour index (0‑based)
                    mask ^= lsb;

                    int gain = 0;

                    /* edges that would become covered if we use colour c */
                    if (i > 0) {
                        int nt = grid[i-1][j];
                        if (nt != c && adj[nt][c]) {
                            int u = min(nt, c), v = max(nt, c);
                            int eid = edge_id[u][v];
                            if (eid != -1 && !covered[eid]) ++gain;
                        }
                    }
                    if (j > 0) {
                        int nl = grid[i][j-1];
                        if (nl != c && adj[nl][c]) {
                            int u = min(nl, c), v = max(nl, c);
                            int eid = edge_id[u][v];
                            if (eid != -1 && !covered[eid]) ++gain;
                        }
                    }

                    /* huge bonus for a colour that has never been used */
                    if (missing_allowed && ( (1ULL << c) & missing_allowed ))
                        gain += 1000;

                    if (gain > best_gain) {
                        best_gain = gain;
                        candidates.clear();
                        candidates.push_back(c);
                    } else if (gain == best_gain) {
                        candidates.push_back(c);
                    }
                }   // end while(mask)

                /* ----- random tie‑break among the best candidates --- */
                uniform_int_distribution<int> dist(0, (int)candidates.size() - 1);
                int chosen = candidates[dist(rng)];

                /* ----- store the colour -------------------------------- */
                grid[i][j] = chosen;
                used_colours |= (1ULL << chosen);

                /* ----- mark newly created edges ------------------------ */
                if (i > 0) {
                    int nt = grid[i-1][j];
                    if (nt != chosen && adj[nt][chosen]) {
                        int u = min(nt, chosen), v = max(nt, chosen);
                        int eid = edge_id[u][v];
                        if (eid != -1 && !covered[eid]) {
                            covered[eid] = 1;
                            ++covered_cnt;
                        }
                    }
                }
                if (j > 0) {
                    int nl = grid[i][j-1];
                    if (nl != chosen && adj[nl][chosen]) {
                        int u = min(nl, chosen), v = max(nl, chosen);
                        int eid = edge_id[u][v];
                        if (eid != -1 && !covered[eid]) {
                            covered[eid] = 1;
                            ++covered_cnt;
                        }
                    }
                }
            }   // end for j
        }       // end for i

        if (!ok) continue;
        if (used_colours != all_mask) continue;        // not all colours used
        if (covered_cnt != M_edges) continue;          // some edges still uncovered

        /* ----- SUCCESS – convert to 1‑based output ---------------- */
        vector<vector<int>> answer(K, vector<int>(K));
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                answer[i][j] = grid[i][j] + 1;          // back to 1‑based
        return answer;
    }   // end for attempts

    /* --------------------------------------------------------------
       Very unlikely fallback – deterministic construction
       -------------------------------------------------------------- */
    int Kfallback = max(2 * N, 2);
    Kfallback = min(Kfallback, 240);
    vector<vector<int>> grid(Kfallback, vector<int>(Kfallback, 1));
    int limit = min(N, Kfallback);
    for (int i = 0; i < limit; ++i)
        grid[i][i] = i + 1;                     // put distinct colours on the diagonal

    return grid;
}
