"""Track J: set-cover solvers keep full coverage; ILP is optimal (brute force); greedy never beats ILP."""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trackJ_software_eng"))
import minimize  # noqa: E402


def random_instance(seed, n_tests=10, n_lines=25):
    rng = random.Random(seed)
    cover = {f"t{i}": {l for l in range(n_lines) if rng.random() < 0.2} for i in range(n_tests)}
    cost = {t: rng.uniform(0.001, 1.0) for t in cover}
    return cover, cost


def brute_force(cover, cost):
    req = minimize.requirements(cover)
    best = None
    tests = sorted(cover)
    for k in range(len(tests) + 1):
        for sub in itertools.combinations(tests, k):
            if minimize.covered_by(sub, cover) >= req:
                c = sum(cost[t] for t in sub)
                best = c if best is None or c < best else best
    return best


def test_solvers_keep_coverage_and_ilp_is_optimal():
    for seed in range(10):
        cover, cost = random_instance(seed)
        req = minimize.requirements(cover)
        g = minimize.greedy(cover, cost)
        x, info = minimize.ilp(cover, cost)
        assert info["status"] == "optimal"
        assert minimize.covered_by(g, cover) >= req
        assert minimize.covered_by(x, cover) >= req
        opt = brute_force(cover, cost)
        assert abs(sum(cost[t] for t in x) - opt) < 1e-9
        assert sum(cost[t] for t in g) >= opt - 1e-9


def test_greedy_can_be_suboptimal():
    # greedy takes x (5 lines/s), then y; ILP takes z alone (1.0 < 1.1)
    cover = {"x": {1, 2}, "y": {3, 4}, "z": {1, 2, 3, 4}}
    cost = {"x": 0.4, "y": 0.7, "z": 1.0}
    assert sorted(minimize.greedy(cover, cost)) == ["x", "y"]
    assert minimize.ilp(cover, cost)[0] == ["z"]


def test_greedy_removes_redundant_tests():
    # greedy takes a (10 lines/s), then b to reach line 2; b also covers line 1, so a is dropped
    cover = {"a": {1}, "b": {1, 2}}
    cost = {"a": 0.1, "b": 0.3}
    assert minimize.greedy(cover, cost) == ["b"]


def test_tests_covering_nothing_are_never_chosen():
    cover = {"a": {1, 2}, "empty": set()}
    cost = {"a": 1.0, "empty": 0.0}
    assert minimize.greedy(cover, cost) == ["a"]
    assert minimize.ilp(cover, cost)[0] == ["a"]


def test_shared_import_costs_are_paid_once():
    # a and c look cheap but need pandas (0.2 s, paid once); b covers line 1 without pandas for 0.05 s
    cover = {"a": {1}, "b": {1}, "c": {2}, "d": {2}}
    cost = {"a": 0.001, "b": 0.05, "c": 0.001, "d": 0.3}
    needs = {"a": {"pandas"}, "c": {"pandas"}}
    setup = {"pandas": 0.2}
    # without import costs both solvers pick a + c (0.002 s), which really costs 0.202 s
    assert sorted(minimize.ilp(cover, cost)[0]) == ["a", "c"]
    # with them: a + c = 0.202 (pandas once) beats b + d = 0.35 and b + c = 0.251
    x, _ = minimize.ilp(cover, cost, needs=needs, setup=setup)
    assert sorted(x) == ["a", "c"]
    assert abs(minimize.suite_cost(x, cost, needs, setup) - 0.202) < 1e-9
    setup["pandas"] = 0.4      # now b + d (0.35) is cheapest
    x, _ = minimize.ilp(cover, cost, needs=needs, setup=setup)
    assert sorted(x) == ["b", "d"]
    g = minimize.greedy(cover, cost, needs=needs, setup=setup)
    assert minimize.covered_by(g, cover) == {1, 2}
    assert minimize.suite_cost(g, cost, needs, setup) >= minimize.suite_cost(x, cost, needs, setup) - 1e-9
