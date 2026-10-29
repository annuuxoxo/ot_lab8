# B2

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** _(fill in after commit)_

## Original prompt
```
In trackB_discrete/qubo.py: Max-Cut on seeded random graphs (6, 8, 10 nodes) as a QUBO, converted to Ising with
x = (1-s)/2; verify the energies match for all bitstrings on the 6-node graph. Add one constrained QUBO (pick exactly k
items) with a penalty P*(sum x - k)^2 and show what happens when P is too small vs large enough.
Solve Max-Cut with brute force and simulated annealing (dwave-neal, 10 seeds, mean ± std).
Save the graphs and optimal cuts to results/trackB/ for Track C.
```

## AI response (excerpt)
Generated `qubo.py` (Max-Cut QUBO, general QUBO->Ising conversion, penalty QUBO for pick-exactly-k feature selection,
brute force, dwave-neal simulated annealing) and the QUBO parts of `b_experiments.py`; graphs + optima saved to
`results/trackB_discrete/maxcut_graphs.json` for Track C.

## Revised prompt / corrections
- Penalty experiment extended: wider P range and SA success rate over 20 seeds instead of a single run, after a
  single run showed SA missing the optimum only at P = 1 (one run was not enough evidence).

## Verification note
QUBO energy == -cut == Ising energy on all bitstrings (6- and 7-node graphs, in experiment and tests).
Large P enforces the constraint, P = 0 does not (test).
