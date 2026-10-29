"""0/1 knapsack solved four ways: greedy, dynamic programming, Branch-and-Bound, ILP.

Problem: choose x_i in {0,1} to maximize sum v_i x_i subject to sum w_i x_i <= C.
"""
import itertools

import numpy as np
import pulp


def make_instance(n, seed):
    """Weakly correlated instance (value ~ weight +- 10): harder than random for greedy and B&B.
    Capacity = half the total weight."""
    rng = np.random.default_rng(seed)
    w = rng.integers(10, 101, n)
    v = np.clip(w + rng.integers(-10, 11, n), 1, None)
    return v.astype(int), w.astype(int), int(w.sum() // 2)


def check(v, w, C, x):
    """Independent feasibility check; returns the objective value."""
    x = np.asarray(x)
    assert set(np.unique(x)) <= {0, 1}, "x must be binary"
    assert w @ x <= C, "capacity violated"
    return int(v @ x)


def brute_force(v, w, C):
    """Try all 2^n subsets. Only for small n; used as ground truth in tests."""
    best, best_x = 0, np.zeros(len(v), int)
    for bits in itertools.product([0, 1], repeat=len(v)):
        x = np.array(bits)
        if w @ x <= C and v @ x > best:
            best, best_x = int(v @ x), x
    return best, best_x


def greedy(v, w, C):
    """Take items in decreasing value/weight ratio while they fit. O(n log n), no optimality guarantee."""
    x = np.zeros(len(v), int)
    cap = C
    for i in np.argsort(-v / w, kind="stable"):
        if w[i] <= cap:
            x[i] = 1
            cap -= w[i]
    return int(v @ x), x


def dp(v, w, C):
    """Dynamic programming over capacity. best[i][c] = best value using items < i with capacity c.
    O(n*C) time: exact, but pseudo-polynomial (cost grows with C, not just n)."""
    n = len(v)
    best = np.zeros((n + 1, C + 1), dtype=np.int64)
    for i in range(n):
        best[i + 1] = best[i]
        wi = w[i]
        # either skip item i, or take it and use the best for capacity c - w_i
        best[i + 1, wi:] = np.maximum(best[i, wi:], best[i, :C + 1 - wi] + v[i])
    # walk back through the table to recover which items were taken
    x = np.zeros(n, int)
    c = C
    for i in range(n, 0, -1):
        if best[i, c] != best[i - 1, c]:
            x[i - 1] = 1
            c -= w[i - 1]
    return int(best[n, C]), x


def branch_and_bound(v, w, C):
    """Depth-first Branch-and-Bound.
    Items are sorted by ratio. At each node we branch on 'take item i' / 'skip item i'.
    Bound: the fractional-knapsack value of the remaining items (an upper bound on any completion).
    A node is pruned when its bound cannot beat the best solution found so far (the incumbent).
    Returns (value, x, nodes_explored)."""
    n = len(v)
    order = np.argsort(-v / w, kind="stable")
    vs, ws = v[order].astype(float), w[order].astype(float)

    def bound(i, cap, val):
        # fill greedily with remaining items, taking a fraction of the first one that does not fit
        for j in range(i, n):
            if ws[j] <= cap:
                cap -= ws[j]
                val += vs[j]
            else:
                return val + vs[j] * cap / ws[j]
        return val

    best_val, best_x = greedy(v, w, C)  # start from the greedy incumbent
    best_val = float(best_val)
    best_sorted = best_x[order].copy()
    nodes = 0
    stack = [(0, C, 0.0, [])]  # (next item, remaining capacity, value so far, choices)
    while stack:
        i, cap, val, choice = stack.pop()
        nodes += 1
        if i == n:
            if val > best_val:
                best_val, best_sorted = val, np.array(choice)
            continue
        if bound(i, cap, val) <= best_val:
            continue  # prune: this subtree cannot improve on the incumbent
        # push 'skip' first so 'take' is explored first (DFS finds good solutions early)
        stack.append((i + 1, cap, val, choice + [0]))
        if ws[i] <= cap:
            stack.append((i + 1, cap - ws[i], val + vs[i], choice + [1]))
    x = np.zeros(n, int)
    x[order] = best_sorted
    return int(round(best_val)), x, nodes


def ilp(v, w, C):
    """Integer Linear Program solved by CBC (via PuLP), which itself uses LP-based branch-and-cut."""
    n = len(v)
    m = pulp.LpProblem("knapsack", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]
    m += pulp.lpSum(int(v[i]) * x[i] for i in range(n))
    m += pulp.lpSum(int(w[i]) * x[i] for i in range(n)) <= C
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    xs = np.array([int(round(xi.value())) for xi in x])
    return int(v @ xs), xs
