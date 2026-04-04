#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <vector>

/* ----------------------------------------------------------------------
   Helper type – a triple together with the *sorted* list of its three
   distances (d1 , d2 , d3).  The distances are stored already sorted so
   we can compare them directly with the sorted heights of the same
   triple.
   ---------------------------------------------------------------------- */
struct Triple
{
    int i{}, j{}, k{};                     // indices i<j<k
    std::array<int, 3> ds{};              // sorted distances (d1,d2,d3)

    Triple(int a, int b, int c) : i(a), j(b), k(c)
    {
        int d1 = j - i;
        int d2 = k - j;
        int d3 = k - i;
        ds = { d1, d2, d3 };
        std::sort(ds.begin(), ds.end());   // make it sorted (ds[0] ≤ ds[1] ≤ ds[2])
    }
};

/* ----------------------------------------------------------------------
   Count how many triples are *mythical* for a given height vector H.
   A triple is mythical iff the three heights (sorted) are exactly the
   three distances (also sorted) of that triple.
   ---------------------------------------------------------------------- */
static int countTriples(const std::vector<int>& H,
                        const std::vector<Triple>& triples)
{
    int cnt = 0;
    for (const auto& tr : triples)
    {
        std::array<int,3> hs = { H[tr.i], H[tr.j], H[tr.k] };
        std::sort(hs.begin(), hs.end());

        if (hs == tr.ds) ++cnt;
    }
    return cnt;
}

std::vector<int> construct_range(int M, int K)
{
    /* ---- 1. pre‑compute all triples for N = max(3,M) ----------------- */
    const int N = std::max(3, M);
    std::vector<Triple> triples;
    triples.reserve(N * (N-1) * (N-2) / 6);          // upper bound for #triples

    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            for (int k = j + 1; k < N; ++k)
                triples.emplace_back(i, j, k);

    /* ---- 2. random search ------------------------------------------------ */
    std::mt19937 rng{ std::random_device{}() };

    // distribution for the initial random heights: 1 … N‑1 (inclusive)
    std::uniform_int_distribution<int> initDist(1, N - 1);

    // same distribution we will use when we need a random element of `ds`
    // (the distances themselves are already in the range 1 … N‑1)
    // No extra work required – we simply shuffle the three already‑present
    // numbers.

    int best_cnt = -1;
    std::vector<int> best_H;

    const int max_restarts = 5;
    const int max_moves    = 20000;      // moves per restart

    for (int restart = 0; restart < max_restarts; ++restart)
    {
        /* ---- a) random initial configuration --------------------------- */
        std::vector<int> cur_H(N);
        for (int i = 0; i < N; ++i)
            cur_H[i] = initDist(rng);

        int cur_cnt = countTriples(cur_H, triples);

        if (cur_cnt > best_cnt)
        {
            best_cnt = cur_cnt;
            best_H   = cur_H;
        }
        if (cur_cnt >= K)
            return cur_H;                 // we already hit the target

        /* ---- b) local hill‑climbing / random‑move phase ----------------- */
        std::vector<int> cur = cur_H;      // we will mutate this copy

        for (int move = 0; move < max_moves; ++move)
        {
            // pick a random triple
            const Triple& tr = triples[ std::uniform_int_distribution<int>(0, triples.size() - 1)(rng) ];

            // remember the old three heights (so we can revert)
            int old_i = cur[tr.i];
            int old_j = cur[tr.j];
            int old_k = cur[tr.k];

            // permute the three distances of this triple
            std::array<int,3> dlist = tr.ds;   // copy
            std::shuffle(dlist.begin(), dlist.end(), rng);

            // apply the move
            cur[tr.i] = dlist[0];
            cur[tr.j] = dlist[1];
            cur[tr.k] = dlist[2];

            int new_cnt = countTriples(cur, triples);

            // accept the move only if it does not make the global count worse
            if (new_cnt >= cur_cnt)
            {
                cur_cnt = new_cnt;
                if (cur_cnt > best_cnt)
                {
                    best_cnt = cur_cnt;
                    best_H   = cur;
                }
                if (cur_cnt >= K)
                    return cur;          // success – we reached K
            }
            else
            {
                // revert the move (the triple must be mythical again)
                cur[tr.i] = old_i;
                cur[tr.j] = old_j;
                cur[tr.k] = old_k;
            }
        }   // end of one restart's move loop
    }   // end of all restarts

    // never reached K – return the best we ever found (it is still a legal vector)
    return best_H;
}
