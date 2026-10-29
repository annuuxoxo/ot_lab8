# Track F — Compiler optimization

Run: `python trackF_compiler/run.py` (≈ 15 s) · Tests: `pytest tests/test_trackF.py`
All numbers from `results/trackF_compiler/*.csv`.

## What was built
A complete mini-compiler in Python for a toy language (integer variables, `+ - * / %`, comparisons, `if/else`, `while`, `print`):

| stage | file | what it does |
|---|---|---|
| lexer + parser | `frontend.py` | regex tokenizer, recursive-descent parser → AST (tuples) |
| IR + codegen | `ir.py` | three-address code (`d = op a b`, `jz`, `jmp`, labels); compiler temporaries start with `%` |
| CFG + liveness | `ir.py` | basic blocks, successors, backward liveness dataflow |
| interpreter | `ir.py` | runs the IR, counts executed instructions and a weighted cost (mul = 3, div/mod = 10, others 1) |
| optimizer | `passes.py` | **6 passes**, applied repeatedly until the code stops changing |
| register allocation | `regalloc.py` | interference graph, greedy colouring vs ILP |

**Passes:** constant folding + constant/copy propagation + algebraic identities · common sub-expression elimination ·
strength reduction (x·2ᵏ → x shl k) · dead-code elimination (unreachable code, useless jumps, dead assignments via
global liveness) · loop-invariant code motion · copy coalescing.

## Results: O0 vs optimized (`f1_O0_vs_opt.csv`, 8 programs in `examples/`)

| program | static instrs | executed instrs | weighted cost | interpreter run time |
|---|---|---|---|---|
| sum_squares | 21 → 13 | 4,508 → 2,407 (**−47%**) | −56% | 0.99 → 0.51 ms |
| nested_invariant | 26 → 21 | 4,667 → 2,588 (−45%) | −54% | 1.03 → 0.55 ms |
| dead_code | 26 → 16 | 373 → 218 (−42%) | −50% | 0.07 → 0.04 ms |
| factorial | 10 → 8 | 100 → 76 (−24%) | −19% | |
| fibonacci | 13 → 11 | 405 → 325 (−20%) | −20% | |
| collatz | 26 → 21 | 8,244 → 6,746 (−18%) | −8% | 1.76 → 1.47 ms |
| gcd | 23 → 18 | 4,159 → 3,456 (−17%) | −8% | |
| primes | 23 → 20 | 31,893 → 28,281 (−11%) | −6% | 6.97 → 6.31 ms |

Compile time: front end 0.02–0.05 ms, optimizer 0.09–0.54 ms per program. Every optimized program prints exactly the same output.

**Pass ablation** (`f1_pass_ablation.csv`, executed instructions with one pass alone): LICM is the big win on loop-heavy
code (nested_invariant 4,667 → 3,229) and coalescing helps everywhere. Constant folding, CSE and strength reduction
give **0 change alone**: they replace work with cheap copies (`d = mov …`, or a shift instead of a multiply, visible in
the weighted cost), and only DCE and coalescing then delete those copies. Passes enable each other, which is why
they run together until nothing changes.

## Compiler decisions formulated as optimization problems

**1. Register allocation** (`regalloc.py`, `f2_regalloc.csv`). Variables x[v,r] ∈ {0,1} (v gets register r), s[v] ∈ {0,1} (spilled).
Minimize Σ cost[v]·s[v], with cost = (uses + defs) × 10^(loop depth). Constraints: each v gets one register or is
spilled; variables live at the same time never share a register. Solved with greedy Chaitin–Briggs colouring and an exact ILP (CBC).

| program (vars, edges) | K = 2: greedy / ILP spill cost | K = 3 | K = 4 | K = 6 |
|---|---|---|---|---|
| nested_invariant (13, 58) | 666 / 666 | 464 / 464 | 323 / 323 | 112 / 112 |
| fibonacci (5, 9) | **71 / 62** | **40 / 31** | 0 / 0 | 0 / 0 |
| primes (9, 24) | 373 / 373 | 142 / 142 | 22 / 22 | 0 / 0 |
| gcd (9, 21) | 293 / 293 | 73 / 73 | 22 / 22 | 0 / 0 |

Greedy matched the ILP optimum in 30 of 32 cases and took about 0.02 ms vs 14–53 ms for the ILP. On fibonacci it
spilled a more expensive variable (cost 71 vs 62). Both allocations pass an independent validity check. (Spilling is
modelled as a cost; the code is not rewritten with loads and stores.)

**2. Pass ordering (phase ordering)** (`f2_pass_ordering.csv`). Decision: a permutation of the 6 passes. Objective:
total executed instructions over the 8 programs. 720 orderings.

| method | order | total executed | orders evaluated |
|---|---|---|---|
| exhaustive search: best | cse > strength > dce > licm > coalesce > const_fold | **44,097** | 720 (10 s) |
| exhaustive search: worst | cse > strength > dce > coalesce > licm > const_fold | 44,697 | 720 |
| original order | const_fold > cse > strength > dce > licm > coalesce | 44,109 | 1 |
| greedy construction | coalesce > licm > dce > const_fold > cse > strength | 44,697 (= worst) | 21 (0.26 s) |

Found because **adding passes made factorial worse** (coalesce alone: 76 instructions; all passes in the original
order: 88). Copy propagation rewrote `print f` into `print %t2`, which kept `%t2` alive and blocked coalescing. Ordering
matters only a little here (just 3 distinct objective values, 1.4% spread), and the best order is now the default.
**Greedy order construction picked the worst order**: `coalesce` looks best on its own, but greedy can't see how the
passes interact later. Exhaustive search is cheap at 720 orders but grows as n!.

## Correctness
- `tests/test_trackF.py` (all passing): every example prints the same output optimized and unoptimized; known outputs
  (C-style division: −7/2 = −3, −7 % 3 = −1; 12! = 479001600); each pass does its job on a small example; register
  allocations are valid and the ILP is never worse than greedy.
- **Differential fuzzing** (`fuzz.py`): random programs with nested loops and ifs, every operator, divisors always ≥ 1.
  0 mismatches in 2,000 programs with all passes together and with each pass alone (500 run in the test suite).
  **Checked that the fuzzer can actually find bugs:** two planted bugs (constant propagation across labels; LICM
  without the operand check) were exposed by 49% and 35% of 500 random programs.
- Known limitation: DCE removes a dead division whose divisor is a variable; if that divisor were 0 at runtime, the
  unoptimized program would crash and the optimized one would not (standard compiler behaviour for "undefined" cases).

## Limitations
The "machine" is the IR interpreter, so speedups are counted instructions plus interpreter time, not native code.
Constant propagation and CSE are local (per basic block) only. The weighted-cost model is a rough cycle estimate.
