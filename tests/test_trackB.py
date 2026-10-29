import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trackB_discrete"))
import knapsack as ks  # noqa: E402
import qubo as qb  # noqa: E402


@pytest.mark.parametrize("seed", range(8))
def test_exact_knapsack_methods_match_brute_force(seed):
    v, w, C = ks.make_instance(12, seed)
    opt, _ = ks.brute_force(v, w, C)
    for val, x in [ks.dp(v, w, C), ks.branch_and_bound(v, w, C)[:2], ks.ilp(v, w, C)]:
        assert val == opt
        assert ks.check(v, w, C, x) == opt


@pytest.mark.parametrize("seed", range(8))
def test_greedy_is_feasible_and_never_beats_optimum(seed):
    v, w, C = ks.make_instance(12, seed)
    opt, _ = ks.brute_force(v, w, C)
    g, x = ks.greedy(v, w, C)
    assert ks.check(v, w, C, x) == g <= opt


def test_bnb_prunes():
    v, w, C = ks.make_instance(40, 0)
    _, _, nodes = ks.branch_and_bound(v, w, C)
    assert nodes < 2 ** 20  # far fewer than the 2^40 leaves of full enumeration


def test_maxcut_qubo_equals_minus_cut_and_ising():
    G = qb.make_graph(7, seed=3)
    Q = qb.maxcut_qubo(G)
    h, J, off = qb.qubo_to_ising(Q)
    for bits in itertools.product([0, 1], repeat=7):
        x = np.array(bits)
        assert np.isclose(qb.qubo_energy(Q, x), -qb.cut_value(G, x))
        assert np.isclose(qb.ising_energy(h, J, off, 1 - 2 * x), qb.qubo_energy(Q, x))


def test_penalty_enforces_cardinality_when_large_enough():
    rng = np.random.default_rng(0)
    rel = rng.uniform(0, 1, 8)
    red = np.abs(rng.uniform(0, 0.2, (8, 8)))
    Q, const = qb.feature_selection_qubo(rel, red, k=3, P=5.0)
    _, x = qb.brute_force_qubo(Q)
    assert x.sum() == 3
    # with no penalty the solver ignores the constraint
    Q0, _ = qb.feature_selection_qubo(rel, red, k=3, P=0.0)
    _, x0 = qb.brute_force_qubo(Q0)
    assert x0.sum() != 3


def test_simulated_annealing_finds_small_maxcut_optimum():
    G = qb.make_graph(8, seed=8)
    Q = qb.maxcut_qubo(G)
    e_bf, _ = qb.brute_force_qubo(Q)
    e_sa, _ = qb.simulated_annealing(Q, num_reads=50, seed=0)
    assert np.isclose(e_sa, e_bf)
