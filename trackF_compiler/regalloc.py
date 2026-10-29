"""Register allocation as an optimization problem.

Interference graph: one node per variable; an edge (u, v) when u and v are live at the same time (so they cannot
share a register). With K registers:

  variables  x[v, r] in {0,1}  (v gets register r),   s[v] in {0,1}  (v is spilled to memory)
  minimize   sum_v cost[v] * s[v]          cost = (uses + defs) * 10^(loop depth)  ~ extra loads/stores executed
  subject to sum_r x[v, r] + s[v] = 1             for every v
             x[u, r] + x[v, r] <= 1               for every edge (u, v) and register r

Solved with (a) greedy graph colouring (Chaitin-Briggs simplify/select heuristic) and (b) an exact ILP (PuLP/CBC).
Simplification: spilling is modelled as a cost only (we do not rewrite the code with loads/stores and re-allocate).
"""
import time
from collections import defaultdict

import pulp

from ir import liveness
from passes import _loops


def interference_graph(code):
    live_out = liveness(code)
    nodes, edges = set(), set()
    for i, ins in enumerate(code):
        nodes |= set(ins.uses())
        if ins.dst is not None:
            nodes.add(ins.dst)
            for v in live_out[i]:
                # dst is written while v is still needed later -> they must be in different registers.
                # (For 'd = mov s', d and s hold the same value, so they may share: no edge.)
                if v != ins.dst and not (ins.op == "mov" and v == ins.a):
                    edges.add(tuple(sorted((ins.dst, v))))
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return sorted(nodes), sorted(edges), adj


def spill_costs(code):
    depth = [0] * len(code)
    for head, end in _loops(code):
        for i in range(head, end + 1):
            depth[i] += 1
    cost = defaultdict(float)
    for i, ins in enumerate(code):
        for v in ins.uses() + ([ins.dst] if ins.dst is not None else []):
            cost[v] += 10 ** depth[i]
    return cost


def greedy_coloring(nodes, adj, K, cost):
    """Simplify: repeatedly remove a node with fewer than K neighbours left (it can always be coloured later).
    If none exists, remove the node with the lowest cost/degree as a *potential* spill (optimistic, Briggs).
    Select: pop nodes and give each the lowest register not used by its coloured neighbours; if none -> spill."""
    remaining = set(nodes)
    deg = {v: len(adj[v]) for v in nodes}
    stack = []
    while remaining:
        easy = [v for v in remaining if deg[v] < K]
        v = min(easy, key=lambda u: deg[u]) if easy else min(remaining, key=lambda u: cost[u] / (deg[u] + 1e-9))
        stack.append(v)
        remaining.remove(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1
    color, spilled = {}, set()
    for v in reversed(stack):
        used = {color[u] for u in adj[v] if u in color}
        free = [r for r in range(K) if r not in used]
        if free:
            color[v] = free[0]
        else:
            spilled.add(v)
    return color, spilled


def ilp_allocation(nodes, edges, K, cost):
    m = pulp.LpProblem("regalloc", pulp.LpMinimize)
    x = {(v, r): pulp.LpVariable(f"x_{i}_{r}", cat="Binary") for i, v in enumerate(nodes) for r in range(K)}
    s = {v: pulp.LpVariable(f"s_{i}", cat="Binary") for i, v in enumerate(nodes)}
    m += pulp.lpSum(cost[v] * s[v] for v in nodes)
    for v in nodes:
        m += pulp.lpSum(x[v, r] for r in range(K)) + s[v] == 1
    for u, v in edges:
        for r in range(K):
            m += x[u, r] + x[v, r] <= 1
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    color = {v: r for (v, r), var in x.items() if var.value() > 0.5}
    spilled = {v for v in nodes if s[v].value() > 0.5}
    return color, spilled, pulp.LpStatus[m.status]


def check(color, spilled, nodes, edges):
    """Valid allocation: every variable has exactly one register or is spilled; neighbours never share."""
    assert set(color) | spilled == set(nodes) and not (set(color) & spilled)
    for u, v in edges:
        assert not (u in color and v in color and color[u] == color[v]), (u, v)
    return True


def allocate(code, K):
    nodes, edges, adj = interference_graph(code)
    cost = spill_costs(code)
    t0 = time.perf_counter()
    gc, gs = greedy_coloring(nodes, adj, K, cost)
    tg = time.perf_counter() - t0
    t0 = time.perf_counter()
    ic, isp, status = ilp_allocation(nodes, edges, K, cost)
    ti = time.perf_counter() - t0
    check(gc, gs, nodes, edges)
    check(ic, isp, nodes, edges)
    return {"vars": len(nodes), "interference_edges": len(edges), "K": K,
            "greedy_spilled": len(gs), "greedy_spill_cost": sum(cost[v] for v in gs), "greedy_ms": 1000 * tg,
            "ilp_spilled": len(isp), "ilp_spill_cost": sum(cost[v] for v in isp), "ilp_ms": 1000 * ti, "ilp_status": status}
