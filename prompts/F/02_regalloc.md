# F2

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 674ed04

## Original prompt
```
In trackF_compiler/regalloc.py: build the interference graph from liveness on the IR and allocate K = 2, 3 registers
with greedy graph colouring (spill the highest degree first). Report spills per program and verify that no two
interfering variables share a register.
```

## AI response (excerpt)
Generated `regalloc.py`: interference graph from liveness, spill cost = occurrences × 10^loop depth, Chaitin–Briggs greedy colouring, ILP in PuLP, an independent validity check. Added (beyond the prompt) a pass-ordering experiment: exhaustive search over all 720 orders vs greedy construction.

## Revised prompt / corrections
- Pass ordering added as a second formulated compiler decision after discovering the phase-ordering problem in F1.

## Verification note
Both allocations validated for every program and K (no interfering pair shares a register); ILP never worse than greedy (test).
