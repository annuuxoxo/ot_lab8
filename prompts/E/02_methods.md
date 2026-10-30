# E2

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-30
- **Commit:** e85049f

## Original prompt
```
In trackE_case_study/: implement (a) a first-fit baseline in given order, (b) greedy DSATUR colouring, and (c) an ILP
in PuLP for the formulation in README. Run on 20 and 40 courses (3 seeds). Report objective, slots, back-to-backs,
violations (must be 0) and runtime. Add a strategy-selection table (variable type, size, exact vs heuristic, hardware,
parallelism). Test: an independent checker validates the hard constraints.
```

## AI response (excerpt)
Generated timetabling.py (generator, independent checker working from raw enrolments, first_fit, dsatur, ilp with
warm start and time limit, clique lower bound), e_experiments.py, run.py and tests/test_trackE.py (checker detects 4
planted violation types; ILP = brute force on tiny instances).

## Revised prompt / corrections
- Plain DSATUR came out *worse* than the baseline (more back-to-backs). Added a back-to-back-aware slot choice and a local search (slot swaps + single-course moves), which gave most of the improvement.
- CBC returns status "Optimal" even when it stops at the time limit; switched to `sol_status` so the table says "Solution Found" (not proven) honestly.
- Added a heuristic scaling run (80, 160 courses) where the ILP is no longer practical.

## Verification note
Every reported timetable has 0 violations according to the independent checker. The ILP's period equals the lower bound on all three 40-course seeds. Numbers in the README were re-checked against e_summary.csv; two overstated figures were corrected before committing.
