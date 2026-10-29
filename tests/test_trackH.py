import itertools
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trackH_dbms"))
import index_selection as h  # noqa: E402


def brute(benefit, size, B):
    best, best_set = 0, ()
    keys = list(benefit)
    for r in range(len(keys) + 1):
        for S in itertools.combinations(keys, r):
            if sum(size[k] for k in S) <= B and sum(benefit[k] for k in S) > best:
                best, best_set = sum(benefit[k] for k in S), S
    return best


@pytest.mark.parametrize("seed", range(10))
def test_dp_and_ilp_are_optimal_greedy_is_feasible(seed):
    rng = np.random.default_rng(seed)
    keys = [f"i{k}" for k in range(9)]
    benefit = {k: float(rng.uniform(-50, 200)) for k in keys}  # includes harmful (negative) candidates
    size = {k: int(rng.integers(1, 40)) for k in keys}
    B = int(rng.integers(10, 120))
    opt = brute(benefit, size, B)
    for fn in (h.dp, h.ilp):
        S = fn(benefit, size, B)
        assert sum(size[k] for k in S) <= B
        assert np.isclose(sum(benefit[k] for k in S), opt)
    G = h.greedy(benefit, size, B)
    assert sum(size[k] for k in G) <= B and sum(benefit[k] for k in G) <= opt + 1e-9
    assert all(benefit[k] > 0 for k in G)


def test_result_hash_ignores_row_order_and_float_noise():
    assert h.result_hash([(1, 2.0000000001), (0, "a")]) == h.result_hash([(0, "a"), (1, 2.0)])
    assert h.result_hash([(1, 2.0)]) != h.result_hash([(1, 2.1)])


def test_timeout_cap_interrupts_slow_query():
    con = sqlite3.connect(":memory:")
    old = h.CAP
    h.CAP = 0.2
    try:
        rows, t = h.timed(con, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c")
        assert rows is None and t == 0.2
    finally:
        h.CAP = old


@pytest.mark.skipif(not os.path.exists(h.DB), reason="TPC-H database not generated (run trackH_dbms/data.py)")
def test_tpch_row_counts_sf01():
    from data import row_counts
    c = row_counts()
    assert c["lineitem"] == 600572 and c["orders"] == 150000 and c["customer"] == 15000
