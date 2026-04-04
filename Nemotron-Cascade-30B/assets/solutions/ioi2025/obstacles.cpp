#include <bits/stdc++.h>
using namespace std;

/* ------------------------------------------------------------------ */
/*  Disjoint Set Union (Union–Find)                                   */
/* ------------------------------------------------------------------ */
struct DSU {
    vector<int> parent, sz;
    DSU(int n = 0) { init(n); }

    void init(int n) {
        parent.resize(n);
        sz.assign(n, 1);
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void union_sets(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) swap(a, b);
        parent[b] = a;
        sz[a] += sz[b];
    }
};

/* ------------------------------------------------------------------ */
/*  Segment tree for range maximum (static)                           */
/* ------------------------------------------------------------------ */
struct MaxSegTree {
    int n;          // original size
    int size;       // power of two >= n
    vector<int> tree;   // 2*size elements

    MaxSegTree() : n(0), size(0) {}

    explicit MaxSegTree(const vector<int>& data) {
        build(data);
    }

    void build(const vector<int>& data) {
        n = static_cast<int>(data.size());
        size = 1;
        while (size < n) size <<= 1;
        tree.assign(2 * size, 0);
        for (int i = 0; i < n; ++i) tree[size + i] = data[i];
        for (int i = size - 1; i > 0; --i)
            tree[i] = max(tree[i << 1], tree[i << 1 | 1]);
    }

    // inclusive query [l, r]; returns -1 for empty interval
    int query(int l, int r) const {
        if (l > r) return -1;
        l += size; r += size;
        int res = -1;
        while (l <= r) {
            if (l & 1)  res = max(res, tree[l++]);
            if (!(r & 1)) res = max(res, tree[r--]);
            l >>= 1; r >>= 1;
        }
        return res;
    }
};

/* ------------------------------------------------------------------ */
/*  Segment tree that stores min / max *active* column indices         */
/* ------------------------------------------------------------------ */
struct ActiveSegTree {
    int n;                 // number of columns
    int size;              // power of two >= n
    vector<int> max_tree; // maximum active index inside the node
    vector<int> min_tree; // minimum active index inside the node

    explicit ActiveSegTree(int n_) : n(n_) {
        size = 1;
        while (size < n) size <<= 1;
        max_tree.assign(2 * size, -1);
        min_tree.assign(2 * size, n);
    }

    // mark column idx as active
    void activate(int idx) {
        int i = idx + size;
        max_tree[i] = idx;
        min_tree[i] = idx;
        for (i >>= 1; i; i >>= 1) {
            int left  = i << 1;
            int right = left | 1;
            max_tree[i] = max(max_tree[left], max_tree[right]);
            min_tree[i] = min(min_tree[left], min_tree[right]);
        }
    }

    // maximum active index in [l, r]; -1 if none
    int query_max(int l, int r) const {
        if (l > r) return -1;
        l += size; r += size;
        int res = -1;
        while (l <= r) {
            if (l & 1)  res = max(res, max_tree[l++]);
            if (!(r & 1)) res = max(res, max_tree[r--]);
            l >>= 1; r >>= 1;
        }
        return res;
    }

    // minimum active index in [l, r]; n if none
    int query_min(int l, int r) const {
        if (l > r) return n;
        l += size; r += size;
        int res = n;
        while (l <= r) {
            if (l & 1)  res = min(res, min_tree[l++]);
            if (!(r & 1)) res = min(res, min_tree[r--]);
            l >>= 1; r >>= 1;
        }
        return res;
    }
};

int N = 0, M = 0;
vector<int> pref_max;   // prefix maximum of T, length N
vector<int> comp;       // component id for each column
DSU* dsu = nullptr;     // created inside initialize

/* ------------------------------------------------------------------ */
/*  Interface required by the statement                                 */
/* ------------------------------------------------------------------ */
void initialize(vector<int> T, vector<int> H)
{
    N = static_cast<int>(T.size());
    M = static_cast<int>(H.size());

    /* 1. prefix maximum of temperatures */
    pref_max.assign(N, 0);
    pref_max[0] = T[0];
    for (int i = 1; i < N; ++i)
        pref_max[i] = max(pref_max[i-1], T[i]);

    /* 2. first blocked row for every column */
    vector<int> col_by_h(M);
    iota(col_by_h.begin(), col_by_h.end(), 0);
    sort(col_by_h.begin(), col_by_h.end(),
         [&](int a, int b){ return H[a] > H[b]; });

    vector<int> first_block(M, N);
    int ptr = 0;
    for (int i = 0; i < N; ++i) {
        int ti = T[i];
        while (ptr < M && H[col_by_h[ptr]] >= ti) {
            int col = col_by_h[ptr];
            first_block[col] = i;
            ++ptr;
        }
    }

    vector<int> Lcol(M);
    for (int j = 0; j < M; ++j) {
        if (first_block[j] == N) Lcol[j] = N - 1;
        else                     Lcol[j] = first_block[j] - 1;
    }

    /* 3. V[j] = max temperature in rows 0 … Lcol[j] */
    vector<int> V(M);
    for (int j = 0; j < M; ++j) {
        if (Lcol[j] >= 0) V[j] = pref_max[Lcol[j]];
        else               V[j] = -1;          // column can never be reached
    }

    /* 4. segment tree for range maximum of humidity (static) */
    MaxSegTree seg_H(H);

    /* 5. DSU over columns */
    dsu = new DSU(M);

    /* 6. process columns in decreasing V */
    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(),
         [&](int a, int b){ return V[a] > V[b]; });

    ActiveSegTree active_seg(M);   // no column active at the beginning

    for (int col : order) {
        int vcol = V[col];
        active_seg.activate(col);               // make the column active

        // ---- left neighbour ----
        int left = -1;
        if (col > 0) left = active_seg.query_max(0, col - 1);
        if (left != -1) {
            int max_h = seg_H.query(left, col);
            if (max_h < vcol) dsu->union_sets(col, left);
        }

        // ---- right neighbour ----
        int right = M;
        if (col + 1 < M) right = active_seg.query_min(col + 1, M - 1);
        if (right != M) {
            int max_h = seg_H.query(col, right);
            if (max_h < vcol) dsu->union_sets(col, right);
        }
    }

    /* 7. store the component id for every column */
    comp.resize(M);
    for (int i = 0; i < M; ++i) comp[i] = dsu->find(i);
}

bool can_reach(int L, int R, int S, int D)
{
    (void)L; (void)R;          // unused, required by the signature
    return comp[S] == comp[D];
}
