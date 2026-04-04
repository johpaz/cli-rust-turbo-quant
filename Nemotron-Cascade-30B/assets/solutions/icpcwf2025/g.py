import sys, math, heapq

def solve():
    data = sys.stdin.read().split()
    it = iter(data)
    t = int(next(it))
    out_lines = []
    for _ in range(t):
        w = int(next(it)); l = int(next(it))
        n = int(next(it)); m = int(next(it))
        vertices = []
        for i in range(n):
            xi = float(next(it)); yi = float(next(it)); zi = float(next(it))
            vertices.append((xi, yi, zi))
        # read triangles (0‑based)
        tri_vertices = []
        for i in range(m):
            a = int(next(it)) - 1
            b = int(next(it)) - 1
            c = int(next(it)) - 1
            tri_vertices.append((a, b, c))

        # Build edge structure
        edge_id = {}
        edge_vertices = []          # list of (u, v) sorted
        edge_triangles = []         # list of list of triangle ids
        tri_edges = [None] * m      # edge ids for each triangle in order (0,1,2)
        for ti in range(m):
            a, b, c = tri_vertices[ti]
            verts = [a, b, c]
            edges = []
            for i_edge in range(3):
                u = verts[i_edge]
                v = verts[(i_edge + 1) % 3]
                key = (u, v) if u < v else (v, u)
                eid = edge_id.get(key)
                if eid is None:
                    eid = len(edge_vertices)
                    edge_id[key] = eid
                    edge_vertices.append(key)
                    edge_triangles.append([])
                edge_triangles[eid].append(ti)
                edges.append(eid)
            tri_edges[ti] = edges

        E = len(edge_vertices)

        # max z for each edge
        edge_max_z = [0.0] * E
        for eid, (u, v) in enumerate(edge_vertices):
            zu = vertices[u][2]
            zv = vertices[v][2]
            edge_max_z[eid] = max(zu, zv)

        # find left and right border vertices (x==0 and x==w)
        left_verts = [i for i, (x, y, z) in enumerate(vertices) if x == 0.0]
        right_verts = [i for i, (x, y, z) in enumerate(vertices) if x == w]
        # sort by y
        left_verts.sort(key=lambda idx: vertices[idx][1])
        right_verts.sort(key=lambda idx: vertices[idx][1])
        vL0, vL1 = left_verts[0], left_verts[1]
        vR0, vR1 = right_verts[0], right_verts[1]

        left_edge_key = (vL0, vL1) if vL0 < vL1 else (vL1, vL0)
        right_edge_key = (vR0, vR1) if vR0 < vR1 else (vR1, vR0)
        left_edge_id = edge_id[left_edge_key]
        right_edge_id = edge_id[right_edge_key]

        # triangles adjacent to borders
        left_tri_id = edge_triangles[left_edge_id][0]   # only one triangle
        right_tri_id = edge_triangles[right_edge_id][0]

        # precompute triangle max heights
        max_z_tri = [0.0] * m
        for ti in range(m):
            a, b, c = tri_vertices[ti]
            h = vertices[a][2]
            h = max(h, vertices[b][2])
            h = max(h, vertices[c][2])
            max_z_tri[ti] = h

        # border elevation intervals
        zL0 = vertices[vL0][2]
        zL1 = vertices[vL1][2]
        left_low = min(zL0, zL1)
        left_high = max(zL0, zL1)

        zR0 = vertices[vR0][2]
        zR1 = vertices[vR1][2]
        right_low = min(zR0, zR1)
        right_high = max(zR0, zR1)

        Z_low = max(left_low, right_low)
        Z_high = min(left_high, right_high)

        # if no overlapping elevation interval
        if Z_low > Z_high + 1e-12:
            out_lines.append("impossible")
            continue

        # pre‑list of interior edges (those shared by two triangles)
        interior_edges = []
        for eid, tris in enumerate(edge_triangles):
            if len(tris) == 2:
                t1, t2 = tris
                interior_edges.append((eid, t1, t2, edge_max_z[eid]))
            # border edges have length 1 and will be handled separately

        # DSU implementation
        class DSU:
            __slots__ = ('parent', 'rank')
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
            def find(self, x):
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x
            def union(self, x, y):
                xr = self.find(x)
                yr = self.find(y)
                if xr == yr:
                    return
                if self.rank[xr] < self.rank[yr]:
                    self.parent[xr] = yr
                elif self.rank[xr] > self.rank[yr]:
                    self.parent[yr] = xr
                else:
                    self.parent[yr] = xr
                    self.rank[xr] += 1

        max_z_edge_left = max(zL0, zL1)
        max_z_edge_right = max(zR0, zR1)

        # connectivity test for a given Z
        def is_connected(Z):
            # DSU for triangles + two border nodes
            dsu = DSU(m + 2)
            left_node = m
            right_node = m + 1
            # interior edges
            for eid, t1, t2, max_z_e in interior_edges:
                if max_z_e + 1e-12 >= Z and max_z_tri[t1] + 1e-12 >= Z and max_z_tri[t2] + 1e-12 >= Z:
                    dsu.union(t1, t2)
            # left border
            if max_z_edge_left + 1e-12 >= Z and max_z_tri[left_tri_id] + 1e-12 >= Z:
                dsu.union(left_tri_id, left_node)
            # right border
            if max_z_edge_right + 1e-12 >= Z and max_z_tri[right_tri_id] + 1e-12 >= Z:
                dsu.union(right_tri_id, right_node)
            return dsu.find(left_node) == dsu.find(right_node)

        # check if any Z in interval works (start at Z_low)
        if not is_connected(Z_low):
            out_lines.append("impossible")
            continue

        # binary search for maximal feasible Z
        lo = Z_low
        hi = Z_high
        for _ in range(55):
            mid = (lo + hi) * 0.5
            if is_connected(mid):
                lo = mid
            else:
                hi = mid
        Z_opt = lo

        # Build graph for Dijkstra at Z_opt
        adj = [[] for _ in range(E)]
        tol = 1e-12
        for ti in range(m):
            a, b, c = tri_vertices[ti]
            h0 = vertices[a][2]
            h1 = vertices[b][2]
            h2 = vertices[c][2]
            edges = tri_edges[ti]  # [e01, e12, e20]
            # compute active edges (product <= 0)
            active = []
            pts = []  # intersection points (x, y) for each active edge
            # edge 0 (a,b)
            d01 = (h0 - Z_opt) * (h1 - Z_opt)
            if d01 <= tol:
                e0 = edges[0]
                u, v = edge_vertices[e0]
                hu = vertices[u][2]
                hv = vertices[v][2]
                # interpolation factor
                tpar = (Z_opt - hu) / (hv - hu)
                x = vertices[u][0] + tpar * (vertices[v][0] - vertices[u][0])
                y = vertices[u][1] + tpar * (vertices[v][1] - vertices[u][1])
                active.append(e0)
                pts.append((x, y))
            # edge 1 (b,c)
            d12 = (h1 - Z_opt) * (h2 - Z_opt)
            if d12 <= tol:
                e1 = edges[1]
                u, v = edge_vertices[e1]
                hu = vertices[u][2]
                hv = vertices[v][2]
                tpar = (Z_opt - hu) / (hv - hu)
                x = vertices[u][0] + tpar * (vertices[v][0] - vertices[u][0])
                y = vertices[u][1] + tpar * (vertices[v][1] - vertices[u][1])
                active.append(e1)
                pts.append((x, y))
            # edge 2 (c,a)
            d20 = (h2 - Z_opt) * (h0 - Z_opt)
            if d20 <= tol:
                e2 = edges[2]
                u, v = edge_vertices[e2]
                hu = vertices[u][2]
                hv = vertices[v][2]
                tpar = (Z_opt - hu) / (hv - hu)
                x = vertices[u][0] + tpar * (vertices[v][0] - vertices[u][0])
                y = vertices[u][1] + tpar * (vertices[v][1] - vertices[u][1])
                active.append(e2)
                pts.append((x, y))

            # add edges between each pair of active edges
            k = len(active)
            if k >= 2:
                for i in range(k):
                    ei = active[i]
                    xi, yi = pts[i]
                    for j in range(i + 1, k):
                        ej = active[j]
                        xj, yj = pts[j]
                        wlen = math.hypot(xi - xj, yi - yj)
                        adj[ei].append((ej, wlen))
                        adj[ej].append((ei, wlen))

        # Dijkstra from left_edge_id to right_edge_id
        INF = 1e100
        dist = [INF] * E
        dist[left_edge_id] = 0.0
        heap = [(0.0, left_edge_id)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u] + 1e-12:
                continue
            if u == right_edge_id:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd + 1e-12 < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        answer = dist[right_edge_id]
        if answer >= INF / 2:
            out_lines.append("impossible")
        else:
            out_lines.append("{:.9f}".format(answer))

    sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
    solve()

