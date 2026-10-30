"""0/1 knapsack: greedy, DP, own Branch-and-Bound, ILP (PuLP)."""
import time
import numpy as np
import pulp


def make_instance(n, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(10, 101, size=n)
    v = w + rng.integers(-10, 31, size=n)  # correlated values
    v = np.maximum(v, 1)
    cap = int(w.sum() // 2)
    return [int(x) for x in v], [int(x) for x in w], cap


def greedy(v, w, cap):
    order = sorted(range(len(v)), key=lambda i: v[i] / w[i], reverse=True)
    tot_w = tot_v = 0
    for i in order:
        if tot_w + w[i] <= cap:
            tot_w += w[i]
            tot_v += v[i]
    return tot_v


def dp(v, w, cap):
    best = np.zeros(cap + 1, dtype=np.int64)
    for vi, wi in zip(v, w):
        if wi <= cap:
            best[wi:] = np.maximum(best[wi:], best[:cap + 1 - wi] + vi)
    return int(best[cap])


def branch_and_bound(v, w, cap):
    """DFS B&B with fractional (Dantzig) upper bound. Returns (value, nodes)."""
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i] / w[i], reverse=True)
    V = [v[i] for i in order]
    W = [w[i] for i in order]
    best = [greedy(v, w, cap)]
    nodes = [0]

    def bound(k, cw, cv):
        rem = cap - cw
        b = cv
        for j in range(k, n):
            if W[j] <= rem:
                rem -= W[j]
                b += V[j]
            else:
                b += V[j] * rem / W[j]
                break
        return b

    def rec(k, cw, cv):
        nodes[0] += 1
        if cv > best[0]:
            best[0] = cv
        if k == n:
            return
        if bound(k, cw, cv) <= best[0] + 1e-9:
            return
        if cw + W[k] <= cap:
            rec(k + 1, cw + W[k], cv + V[k])
        rec(k + 1, cw, cv)

    rec(0, 0, 0)
    return best[0], nodes[0]


def ilp(v, w, cap):
    n = len(v)
    prob = pulp.LpProblem("knap", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]
    prob += pulp.lpSum(v[i] * x[i] for i in range(n))
    prob += pulp.lpSum(w[i] * x[i] for i in range(n)) <= cap
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[prob.status] == "Optimal"
    return int(round(pulp.value(prob.objective)))


def brute_force(v, w, cap):
    n = len(v)
    best = 0
    for mask in range(1 << n):
        tw = tv = 0
        for i in range(n):
            if mask >> i & 1:
                tw += w[i]
                tv += v[i]
        if tw <= cap and tv > best:
            best = tv
    return best
