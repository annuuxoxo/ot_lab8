# F1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 674ed04

## Original prompt
```
In trackF_compiler/: a mini-compiler in Python for a toy language (int vars, + - * /, assignment, if, while, print):
lexer → recursive-descent parser → AST → three-address IR → IR interpreter that counts executed instructions.
Passes (each switchable): constant folding + propagation, dead-code elimination, common sub-expression elimination.
6 example programs. Measure static and executed instruction count, run time and compile time with passes off vs on.
Test: optimized output == unoptimized output for every program. Keep it simple — I must explain it in a viva.
```

## AI response (excerpt)
Generated `frontend.py` (lexer + recursive-descent parser), `ir.py` (three-address IR, codegen, basic blocks, liveness, interpreter with instruction counts), `passes.py` (constant folding/propagation, CSE, strength reduction, DCE, LICM), 8 example programs, `fuzz.py` (random terminating programs for differential testing), `f_experiments.py`.

## Revised prompt / corrections
- Added a 6th pass (copy coalescing) after seeing many 'temp = op; x = mov temp' pairs in the output.
- CSE had a stale-variable bug risk (avail_key not reset each iteration); fixed before it caused a failure.
- Found that all passes gave WORSE code than coalescing alone on factorial (88 vs 76 instructions): copy propagation blocked coalescing. Led to the pass-ordering experiment (see F2) and a new default order (failure log #14).
- Validated the fuzzer itself by planting two bugs; both were caught.

## Verification note
2,000 random programs: 0 output mismatches (all passes, and each pass alone). All 8 examples identical output. Tests pass.
