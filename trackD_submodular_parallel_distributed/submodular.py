"""Track D part 1: submodular maximization.

Max coverage: choose k of m sets to cover as many elements as possible, f(S) = |union of sets in S|.
f is monotone and submodular (diminishing returns), so greedy is guaranteed to reach >= (1 - 1/e) of the optimum
under a cardinality constraint (Nemhauser, Wolsey, Fisher 1978). We TEST those assumptions instead of assuming them,
and show an objective that violates them, where greedy can be arbitrarily bad.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import save_results  # noqa: E402

TRACK = "trackD_submodular_parallel_distributed"


def make_coverage(m=20, universe=200, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 40, m)
    return [frozenset(rng.choice(universe, s, replace=False).tolist()) for s in sizes]


def coverage(sets, S):
    return len(frozenset().union(*[sets[i] for i in S])) if S else 0


def with_pair_bonus(sets, pairs, bonus):
    """Coverage + bonus for each 'complementary' pair chosen together -> NOT submodular (the pair is worth more
    together than the sum of its parts: increasing returns)."""
    def f(S):
        s = set(S)
        return coverage(sets, S) + bonus * sum(1 for a, b in pairs if a in s and b in s)
    return f


def test_assumptions(f, m, samples=2000, seed=0):
    """Empirical check on random (A subset of B, e not in B):
    monotone: f(B) >= f(A);   diminishing returns: f(A+e) - f(A) >= f(B+e) - f(B)."""
    rng = np.random.default_rng(seed)
    mono_viol = dr_viol = 0
    for _ in range(samples):
        B = set(rng.choice(m, rng.integers(1, m - 1), replace=False).tolist())
        A = set(x for x in B if rng.random() < 0.5)
        e = int(rng.choice([x for x in range(m) if x not in B]))
        fA, fB = f(sorted(A)), f(sorted(B))
        mono_viol += fB < fA
        dr_viol += (f(sorted(A | {e})) - fA) < (f(sorted(B | {e})) - fB)
    return mono_viol / samples, dr_viol / samples


def greedy(f, m, k):
    """Add the element with the largest marginal gain, k times. O(k*m) evaluations of f."""
    S = []
    for _ in range(k):
        best = max((i for i in range(m) if i not in S), key=lambda i: f(S + [i]))
        S.append(best)
    return S


def brute_force(f, m, k):
    return max(itertools.combinations(range(m), k), key=lambda S: f(list(S)))


def experiment(m=20, ks=(2, 3, 4, 5, 6)):
    sets = make_coverage(m)
    cov = lambda S: coverage(sets, S)  # noqa: E731
    rng = np.random.default_rng(1)
    pairs = [tuple(rng.choice(m, 2, replace=False)) for _ in range(6)]
    bonus = with_pair_bonus(sets, pairs, bonus=40)

    rows = []
    for name, f in [("coverage (submodular)", cov), ("coverage + pair bonus (NOT submodular)", bonus)]:
        mono, dr = test_assumptions(f, m)
        for k in ks:
            t0 = time.perf_counter()
            g = greedy(f, m, k)
            tg = time.perf_counter() - t0
            t0 = time.perf_counter()
            opt = brute_force(f, m, k)
            tb = time.perf_counter() - t0
            rows.append({"objective": name, "k": k, "monotone_violation_rate": mono, "diminishing_returns_violation_rate": dr,
                         "greedy_value": f(g), "optimum": f(list(opt)), "ratio": f(g) / f(list(opt)),
                         "bound_1_minus_1_over_e": 1 - 1 / np.e, "greedy_ms": 1000 * tg, "brute_force_ms": 1000 * tb})

    # Adversarial non-submodular instance: items 0 and 1 are worthless alone but worth 100 together; item 2 is worth 1.
    def adv(S):
        s = set(S)
        return (100 if {0, 1} <= s else 0) + (1 if 2 in s else 0)
    g = greedy(adv, 3, 2)
    rows.append({"objective": "adversarial pair (NOT submodular)", "k": 2, "greedy_value": adv(g),
                 "optimum": adv([0, 1]), "ratio": adv(g) / adv([0, 1]), "bound_1_minus_1_over_e": 1 - 1 / np.e})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "d1_submodular")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(experiment().to_string(index=False))
