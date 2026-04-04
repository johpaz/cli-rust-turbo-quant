// ------------------------------------------------------------
//  Dinosaur migration – research team (send_message) and Museum
// ------------------------------------------------------------

#include <vector>
#include <utility>   // std::pair
#include <algorithm> // std::max

// -----------------------------------------------------------------
//  Constants and global data that survive across calls of send_message
// -----------------------------------------------------------------
static const int _BLOCK = 100;          // block size for the encoding

// current test case size (0 means “no test case started yet”)
static int _curr_N = -1;

// depth[v] for the vertices that have already been visited
static std::vector<int> _depth;

// deepest distance seen so far and the *largest* vertex that attains it
static int _max_depth = -1;
static int _max_node  = -1;

int send_message(int N, int i, int Pi)
{
    // (re)initialise the data structures when we get a new test case
    if (N != _curr_N) {
        _curr_N   = N;
        _depth.assign(N, 0);   // depth[0] stays 0, others are filled later
        _max_depth = -1;
        _max_node  = -1;
    }

    // depth of the current vertex
    int d = _depth[Pi] + 1;
    _depth[i] = d;

    // keep the deepest vertex seen so far (largest index on ties)
    if (d > _max_depth || (d == _max_depth && i > _max_node)) {
        _max_depth = d;
        _max_node  = i;
    }

    // decide what (if anything) to send from site i
    if (i == N - 2) {                       // second‑last site – send block part
        int block = _max_node / _BLOCK;     // integer division, always >= 0
        return block + 1;                   // 1 … 100
    }
    else if (i == N - 1) {                  // last site – send remainder part
        int rem = _max_node % _BLOCK;       // remainder in [0, 99]
        return rem + 1;                     // 1 … 100
    }
    else {
        return 0;                           // no message
    }
}

std::pair<int,int> longest_path(std::vector<int> S)
{
    const int N = static_cast<int>(S.size());

    if (N == 1) {
        return {0, 0};
    }

    // The two last entries are the encoded answer.
    int block_val = S[N - 2];
    int rem_val   = S[N - 1];

    // Convert back to the original block index (0‑based) and remainder.
    int block = (block_val > 0) ? (block_val - 1) : 0;   // 0 … 99
    int rem   = (rem_val   > 0) ? (rem_val   - 1) : 0;   // 0 … 99

    // Re‑assemble the index of the farthest vertex.
    int idx = block * _BLOCK + rem;

    if (idx < 0)            idx = 0;
    else if (idx >= N)      idx = N - 1;

    // Site 0 is always an endpoint of a diameter (as required by the
    // original research problem).  The second endpoint is `idx`.
    return {0, idx};
}
