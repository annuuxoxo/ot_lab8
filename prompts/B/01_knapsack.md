# B1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 131d1cf

## Original prompt
```
In trackB_discrete/: solve 0/1 knapsack with greedy (value/weight), dynamic programming, my own Branch-and-Bound
(fractional upper bound), and ILP via PuLP. Random seeded instances, n = 10, 50, 100, 200, 3 per size.
Report objective, optimality gap, runtime and B&B nodes; plot runtime vs n.
Tests: for n ≤ 15 the exact methods match brute force.
```

## AI response (excerpt)
Generated `knapsack.py` (greedy, DP with table walk-back, DFS Branch-and-Bound with fractional bound and greedy incumbent,
ILP via PuLP, brute force, independent `check()`), `b_experiments.py` (knapsack sizes 10-200, feature selection greedy vs
exhaustive on wine), `tests/test_trackB.py`.

## Revised prompt / corrections
- PuLP 4.2 broke the ILP code (new API, no bundled solver) -> pinned pulp==2.9.0 (failure log #4).
- Feature-selection evaluation counting rewritten: the first version cleared a shared cache in several places, which
  was correct but hard to explain; now each method counts its own evaluations.

## Verification note
DP, B&B and ILP equal brute force on 8 instances (tests); every solution passes `check()` in the experiment loop.
**Tool comparison: this prompt must also be run in a second AI tool -> docs/tool_comparison.md (to do by the student).**
