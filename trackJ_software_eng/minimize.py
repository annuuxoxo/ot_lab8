"""Test-suite minimization as weighted set cover.

x_t in {0,1}: keep test t.   minimize  sum_t time_t * x_t
subject to  sum_{t covers l} x_t >= 1   for every line l covered by the full suite.

Solvers: greedy (best new-lines-per-second; ln(n)+1 approximation for weighted set cover),
ILP (PuLP/CBC, exact), and a random baseline.

Shared setup costs (optional `setup`): some tests need a heavy module (numpy, pandas) whose import is paid
once per run, by whichever selected test imports it first. Then the cost is not a sum over tests:
    minimize  sum_t base_t * x_t + sum_m import_m * z_m
    s.t.      coverage as above,   z_m >= x_t  for every module m that test t needs
(a fixed-charge set cover). `needs[t]` = set of modules, `setup[m]` = import cost.
"""
import random

import pulp


def requirements(cover):
    """All lines covered by at least one test. cover: {test: set(lines)}."""
    out = set()
    for lines in cover.values():
        out |= lines
    return out


def covered_by(selection, cover):
    out = set()
    for t in selection:
        out |= cover[t]
    return out


def suite_cost(selection, cost, needs=None, setup=None):
    """Model cost of a suite: per-test base costs plus each needed module's import once."""
    total = sum(cost[t] for t in selection)
    if setup:
        mods = set().union(*[needs.get(t, set()) for t in selection]) if selection else set()
        total += sum(setup[m] for m in mods)
    return total


def greedy(cover, cost, needs=None, setup=None):
    """Repeatedly take the test with the most still-uncovered lines per second of *marginal* cost
    (its base cost plus imports not yet paid). Then drop tests made redundant (most expensive first)."""
    needs, setup = needs or {}, setup or {}
    need = requirements(cover)
    chosen, paid = [], set()

    def marginal(t):
        return cost[t] + sum(setup[m] for m in needs.get(t, set()) - paid if m in setup)

    while need:
        t = max((t for t in cover if cover[t] & need),
                key=lambda t: (len(cover[t] & need) / max(marginal(t), 1e-9), -marginal(t), t))
        chosen.append(t)
        paid |= needs.get(t, set())
        need -= cover[t]
    full = requirements(cover)
    changed = True
    while changed:
        changed = False
        for t in sorted(chosen, key=lambda t: -(suite_cost(chosen, cost, needs, setup)
                                                - suite_cost([u for u in chosen if u != t], cost, needs, setup))):
            rest = [u for u in chosen if u != t]
            if covered_by(rest, cover) >= full:
                chosen, changed = rest, True
                break
    return chosen


def ilp(cover, cost, time_limit: float = 120.0, needs=None, setup=None):
    """Exact weighted set cover. Lines covered by the same set of tests are merged first
    (identical constraints), which shrinks the model without changing the optimum."""
    tests = sorted(cover)
    by_line = {}
    for t in tests:
        for line in cover[t]:
            by_line.setdefault(line, []).append(t)
    rows = {frozenset(ts) for ts in by_line.values()}
    m = pulp.LpProblem("test_suite_minimization", pulp.LpMinimize)
    x = {t: pulp.LpVariable(f"x_{i}", cat="Binary") for i, t in enumerate(tests)}
    needs, setup = needs or {}, setup or {}
    z = {mod: pulp.LpVariable(f"z_{i}", cat="Binary") for i, mod in enumerate(sorted(setup))}
    m += pulp.lpSum(cost[t] * x[t] for t in tests) + pulp.lpSum(setup[mod] * z[mod] for mod in z)
    for t in tests:
        for mod in needs.get(t, set()):
            if mod in z:
                m += z[mod] >= x[t]
    for ts in rows:
        m += pulp.lpSum(x[t] for t in ts) >= 1
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    chosen = [t for t in tests if x[t].value() is not None and x[t].value() > 0.5]
    proven = m.sol_status == pulp.LpSolutionOptimal
    return chosen, {"status": "optimal" if proven else pulp.LpSolution[m.sol_status],
                    "n_vars": len(tests), "n_constraints": len(rows), "n_lines": len(by_line)}


def random_subset(cover, k: int, seed: int):
    """Naive baseline: k tests chosen uniformly at random (same size as the optimized suite)."""
    rng = random.Random(seed)
    return rng.sample(sorted(cover), k)
