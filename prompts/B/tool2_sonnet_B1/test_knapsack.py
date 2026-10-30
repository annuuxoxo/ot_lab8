import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pytest
from knapsack import *


@pytest.mark.parametrize("n", [5, 10, 12, 15])
@pytest.mark.parametrize("seed", range(5))
def test_exact_match_brute_force(n, seed):
    v, w, cap = make_instance(n, seed)
    bf = brute_force(v, w, cap)
    assert dp(v, w, cap) == bf
    assert branch_and_bound(v, w, cap)[0] == bf
    assert ilp(v, w, cap) == bf
    assert greedy(v, w, cap) <= bf
