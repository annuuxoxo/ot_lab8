# Track B — Discrete optimization

Run: `python trackB_discrete/run.py` (≈ 15 s) · Tests: `pytest tests/test_trackB.py`
All numbers below come from `results/trackB_discrete/*.csv`.

## Objective
Compare exact methods and heuristics on combinatorial problems (greedy, dynamic programming, Branch-and-Bound, ILP),
and formulate problems as QUBO / Ising models with penalty terms — the classical baseline for Track C.

## Formulations
**0/1 knapsack** — variables x_i ∈ {0,1}; maximize Σ v_i x_i s.t. Σ w_i x_i ≤ C.
Weakly correlated instances (v_i ≈ w_i ± 10, harder than random), C = half the total weight; n = 10, 50, 100, 200, 3 seeded instances each.

**Feature selection** — choose a subset S of the 13 wine features with |S| = k to maximize 5-fold CV accuracy of logistic regression.

**Max-Cut as QUBO** — x_i ∈ {0,1} = side of node i. Edge cut iff x_i + x_j − 2x_i x_j = 1, so
minimize xᵀQx with Q_ii = −deg(i), Q_ij = 2 per edge ⇒ xᵀQx = −cut. Ising via x = (1 − s)/2 (`qubo.qubo_to_ising`).

**Constrained feature selection as QUBO** — minimize −Σ rel_i x_i + Σ red_ij x_i x_j + P(Σ x_i − k)², with rel = |corr(feature, label)|,
red = |corr(feature_i, feature_j)|, k = 3. The penalty expands to P(1 − 2k) on each diagonal entry and 2P off-diagonal.

## Methods
| Method | Type | Cost | File |
|---|---|---|---|
| Greedy (value/weight) | heuristic | O(n log n) | `knapsack.greedy` |
| Dynamic programming | exact | O(n·C) — pseudo-polynomial | `knapsack.dp` |
| Branch-and-Bound (own; fractional-knapsack bound, greedy incumbent) | exact | exponential worst case | `knapsack.branch_and_bound` |
| ILP (PuLP + CBC) | exact | branch-and-cut | `knapsack.ilp` |
| Brute force / simulated annealing (dwave-neal) | exact / heuristic | 2ⁿ / tunable | `qubo.py` |

## Results

**Knapsack** (`b1_knapsack_summary.csv`, `b1_knapsack.png`)

| n | Greedy gap (mean / max) | DP time | B&B time (nodes) | ILP time |
|---|---|---|---|---|
| 10 | 0.91% / 2.74% | 0.03 ms | 0.05 ms (45) | 27 ms |
| 50 | 0.23% / 0.47% | 0.14 ms | 0.43 ms (703) | 22 ms |
| 100 | 0.12% / 0.36% | 0.42 ms | 0.72 ms (710) | 20 ms |
| 200 | 0.10% / 0.16% | 1.44 ms | 1.80 ms (762) | 24 ms |

DP, B&B and ILP always return the same optimum (checked independently for feasibility). Greedy is near-optimal and
gets relatively better as n grows. B&B explores ~760 nodes instead of 2²⁰⁰ leaves thanks to the bound. ILP time is
dominated by launching the CBC solver (~20 ms constant), so it is slowest here but the most general method. DP is fast
because C is small (~5,500); its cost grows with C, not just n.

**Feature selection** (`b1_feature_selection.csv`)

| k | Exhaustive (subsets, acc) | Greedy forward (subsets, acc) |
|---|---|---|
| 2 | 78, 0.916 | 25, 0.916 |
| 3 | 286, **0.950** | 36, 0.939 ✗ suboptimal |
| 4 | 715, 0.972 | 46, 0.972 |

Greedy is 6–16× cheaper but misses the best subset at k = 3 (it commits to feature 0 early; the optimum {6, 9, 12} does not contain it).

**Max-Cut QUBO** (`b2_maxcut.csv`; graphs + optima saved to `maxcut_graphs.json` for Track C)

| n | edges | optimal cut | brute force | SA approx. ratio (10 seeds) |
|---|---|---|---|---|
| 6 | 7 | 6 | 0.09 ms | 1.00 ± 0 |
| 8 | 19 | 14 | 0.37 ms | 1.00 ± 0 |
| 10 | 23 | 17 | 1.5 ms | 1.00 ± 0 |
| 12 | 36 | 25 | 6.1 ms | 1.00 ± 0 |

QUBO energy = −cut and Ising energy = QUBO energy verified on all 2⁶ bitstrings. Brute force grows as 2ⁿ (×4 per +2 nodes).

**Penalty weight** (`b2_feature_qubo_penalty.csv`, SA with 5 reads × 20 seeds)

| P | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.2 | 1 | 5 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| optimum feasible (k = 3)? | ✗ (picks 2) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SA success rate | 0.90 | 0.50 | 0.45 | 0.50 | 0.85 | 0.30 | 0.20 | 0.00 | 0.05 |

With no penalty the constraint is ignored. Any P ≥ 0.005 makes the optimum feasible, but **a large penalty makes the
problem harder for the annealer** (success falls to 0–5% at P ≥ 5): the penalty dominates the energy scale, so the
objective differences become tiny by comparison and the landscape is full of steep walls. P should be "just large enough".

## Correctness checks (`tests/test_trackB.py`, all passing)
DP, B&B and ILP equal brute force on 8 random instances; greedy is feasible and never beats the optimum; B&B prunes;
QUBO = −cut = Ising on all bitstrings; a large enough penalty enforces the cardinality constraint; SA finds the 8-node optimum.

## Limitations
Small instances (chosen so brute force can verify everything); ILP timing includes solver start-up; SA results depend
on its read budget.
