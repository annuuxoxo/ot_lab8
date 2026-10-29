# Assignment 8 — Optimize Anything: 5-Day Plan (reduced scope)

Source: `Assignment_8_Optimize_Anything_Vibe_Coding_Challenge.pdf`
Time: **5 days, ~5 h/day**. All 10 tracks (A–J) are compulsory, so none are dropped. Each one is cut to the
**minimum that still satisfies the brief**: objective → baseline → fair comparison → correctness check → one table/plot.

---

## 0. Scope: what we do and what we skip

| Track | Minimum we do | Skipped |
|---|---|---|
| A | All required optimizers on 3 functions (quadratic, Rosenbrock, saddle) + tiny MLP with hand-written backprop | Big sweeps, Rastrigin, extra datasets |
| B | Knapsack: greedy / DP / B&B / ILP. Max-Cut as QUBO solved by brute force + SA | Scheduling, extra problems |
| C | QAOA (p=1,2) on B's Max-Cut, tiny VQE, one PQC trained 3 ways, shots + noise + barren plateau plots | Real hardware |
| D | Submodular greedy, process-pool speedup (Amdahl), FedAvg, MGDA vs GradNorm | Model/pipeline parallel → short simulated demo |
| E | Exam timetabling: baseline vs greedy vs ILP | Metaheuristics, large data |
| F | Python toy compiler with 3 passes (+1 if time) | LLVM, register-allocation ILP (greedy colouring only, if time) |
| G | xv6: one feature (priority or MLFQ scheduler + stats syscall), 3 workloads | COW fork / second feature |
| H | SQLite + TPC-H SF 0.1: index selection, greedy vs ILP vs no-index default | Join-order DP, PostgreSQL |
| I | All taught algorithms on a simulator (DJ + BV hand-written), one noise-model run, quantum kernel SVM vs classical SVM | IBM hardware, VQC |
| J | Test-suite minimization on a small OSS repo: greedy vs ILP vs full suite | Mutation testing |

Label every quantum and parallel result **simulated** or **measured**. In the report, say that the scope was cut for time.

---

## System requirements

**Hardware (your Mac: Apple M4, 10 cores, 16 GB RAM, ~35 GB free, macOS 26). This is enough.**
- RAM: 8 GB minimum, 16 GB comfortable. Keep quantum simulations at ≤ 20 qubits (statevector memory doubles per qubit).
- Disk: ~10 GB needed (venv with torch + qiskit ~4 GB, xv6 toolchain ~1–2 GB, TPC-H data ~1 GB, results). Free up space if you drop below ~15 GB.
- Cores: ≥ 4 for meaningful Track D speedup plots. With 10 cores (4 performance + 6 efficiency on M4), mention the mixed core types when explaining the speedup curve.
- No GPU or quantum hardware needed. Everything runs on CPU and simulators.

**Software**

| Need | Package / tool | Status on your Mac |
|---|---|---|
| Package manager | Homebrew | ✅ installed |
| Version control | git + GitHub account (public repo) | ✅ git installed |
| Python | **3.12** recommended (some quantum/D-Wave packages lag on 3.13) → `brew install python@3.12` | ⚠️ you have 3.13 |
| Python libraries | numpy, scipy, pandas, matplotlib, scikit-learn, torch, pulp, networkx, pytest, coverage | install in venv |
| Quantum | qiskit, qiskit-aer, pennylane, dwave-neal, minorminer, dwave-networkx | install in venv |
| xv6 (Track G) | `brew install qemu riscv64-elf-gcc` (fallback: Docker / OrbStack + Ubuntu) | ❌ not installed |
| Database (Track H) | sqlite3 (built in) + `pip install duckdb` to generate TPC-H data | ✅ sqlite3; duckdb via pip |
| Compiler (Track F) | nothing extra (pure Python) | ✅ |
| AI tools | Claude Code + one more (ChatGPT / Copilot / Gemini) for the tool comparison | — |

**Setup commands**
```bash
brew install python@3.12 qemu riscv64-elf-gcc
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib scikit-learn torch pulp networkx pytest coverage \
            qiskit qiskit-aer pennylane dwave-neal minorminer dwave-networkx duckdb
git clone https://github.com/mit-pdos/xv6-riscv.git trackG_xv6/xv6
cd trackG_xv6/xv6 && make qemu      # quit QEMU with Ctrl-A then X
```

---

## 1. Habits for every day (the vibe-coding marks depend on these)

- Save every prompt + a short response excerpt + a 1-line verification note in `prompts/<track>/`.
- Commit per task: `[AI:claude-code] trackX: ...`. Use `[manual] trackX: ...` for your own fixes.
- Log AI mistakes in `docs/failure_log.md` as soon as they happen (**need ≥3**).
- **Tool comparison:** run prompt B1 in a second AI tool too (ChatGPT / Copilot / Gemini), and note correctness, hallucinations, time taken and prompt iterations in `docs/tool_comparison.md`.
- Add each AI tool you use to the README's AI tools register.

---

## Day 1 — Setup + Track A + Track B (~5 h)

Start the slow installs first (you'll need them on Day 4):
```bash
brew install qemu riscv64-elf-gcc
git clone https://github.com/mit-pdos/xv6-riscv.git trackG_xv6/xv6
```

**S1 — repo skeleton (30 min)**
```
Set up a Python repo for a university optimization assignment with 10 tracks.
Create folders trackA_continuous … trackJ_software_eng (A continuous, B discrete, C quantum_opt,
D submodular_parallel_distributed, E case_study, F compiler, G xv6, H dbms, I quantum_algos_qml, J software_eng),
each with run.py and a README stub (Objective / Formulation / Baseline / Method / Results / Correctness).
Add common/utils.py (set_seed, timer, save_results to results/<track>/*.csv, save_plot), prompts/A…J,
docs/{chapter6_checklist,failure_log,tool_comparison,report}.md, tests/, and requirements.txt with:
numpy scipy pandas matplotlib torch scikit-learn pulp qiskit qiskit-aer pennylane dwave-neal networkx pytest coverage.
README: setup, how to run each track, hardware info, empty AI tools register table. Create a venv and install.
```

**A1 — optimizers (1.5 h)**
```
In trackA_continuous/, implement in NumPy: GD, SGD, mini-batch SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam,
Newton, and Gauss–Newton (fit y = a*exp(b*x) to synthetic data).
Functions: ill-conditioned convex quadratic, Rosenbrock (non-convex), and f = x^2 - y^2 started near the saddle.
SGD variants get Gaussian gradient noise (fixed seed). Same start, iteration budget and tolerance for every method.
Output: loss curves, a results table (final value, iterations, time), and one learning-rate sweep on Rosenbrock for
GD/Momentum/Adam. Tests: finite-difference gradient check; Newton solves the quadratic in one step.
```

**A2 — backprop + curvature (1 h)**
```
In trackA_continuous/mlp.py: a 1-hidden-layer MLP on make_moons with backprop written by hand. Check the gradients
against PyTorch autograd, then train with SGD vs Adam, with and without L2 regularization.
In trackA_continuous/curvature.py: for logistic regression, compute the Hessian, a diagonal approximation, the
empirical Fisher, and a Hessian-vector product via torch double-backward (check HVP == H @ v). Plot log(f - f*) for
the convex vs strongly-convex (L2) cases to show sublinear vs linear convergence.
```

**B1 — knapsack exact vs heuristic (1 h)** ← *also run this one in a second AI tool*
```
In trackB_discrete/: solve 0/1 knapsack with greedy (value/weight), dynamic programming, my own Branch-and-Bound
(fractional upper bound), and ILP via PuLP. Random seeded instances, n = 10, 50, 100, 200, 3 per size.
Report objective, optimality gap, runtime and B&B nodes; plot runtime vs n.
Tests: for n ≤ 15 the exact methods match brute force.
```

**B2 — QUBO (45 min)**
```
In trackB_discrete/qubo.py: Max-Cut on seeded random graphs (6, 8, 10 nodes) as a QUBO, converted to Ising with
x = (1-s)/2; verify the energies match for all bitstrings on the 6-node graph. Add one constrained QUBO (pick exactly k
items) with a penalty P*(sum x - k)^2 and show what happens when P is too small vs large enough.
Solve Max-Cut with brute force and simulated annealing (dwave-neal, 10 seeds, mean ± std).
Save the graphs and optimal cuts to results/trackB/ for Track C.
```

---

## Day 2 — Quantum: Track C + Track I (~5 h)

**C1 — QAOA + VQE + shots/noise (1.5 h)**
```
In trackC_quantum_opt/ (Qiskit + Aer, simulator only):
1. QAOA for Max-Cut on the Track B graphs, p = 1 and 2, COBYLA. Report the approximation ratio vs brute force and
   vs SA, circuit evaluations and time.
2. VQE on a 2-qubit Hamiltonian (e.g. H2 as a Pauli sum) vs exact diagonalization.
3. QAOA p=1 at 100 / 1000 / 10000 shots (5 seeds) and with a fake-backend noise model vs noiseless.
Test: QAOA bitstring energy == classical QUBO energy.
```

**C2 — PQC gradients + barren plateau (1 h)**
```
In trackC_quantum_opt/pqc.py (PennyLane): a small PQC classifier on 2-feature toy data, trained with
(a) parameter-shift written by me (checked against qml.grad), (b) SPSA, (c) QNGOptimizer. Compare loss curves and
circuit evaluations. Then plot gradient variance vs qubit count (2–10 qubits, random circuits, 100 samples) to show
the barren plateau. Add a short note on QUBO embedding limits on annealers: run minorminer on complete graphs K_n
for growing n and report chain lengths.
```

**I1 — quantum algorithms (1.5 h)**
```
In trackI_quantum_algos_qml/ (Qiskit + Aer):
- Hand-written circuits: Deutsch–Jozsa (constant + balanced) and Bernstein–Vazirani. Tests: correct answer with prob 1.
- Also: Grover (3–4 qubits), QFT (check vs numpy FFT), QPE (T gate → phase 1/8), Shor for 15 (textbook circuit is
  fine), and annealing on a small QUBO with dwave-neal (labelled as classical simulation).
- Run BV with a fake-backend noise model vs ideal.
- Grover vs linear search for n = 2..8 qubits: oracle queries, transpiled gate counts, simulator wall time.
  Explain why the simulation time still grows exponentially.
```

**I2 — QML vs classical (1 h)**
```
In trackI_quantum_algos_qml/qml.py: on make_moons (same split, 5 seeds), compare a quantum-kernel SVM (fidelity
kernel, simulator) with an RBF SVM and logistic regression. Table: accuracy mean ± std, training time, parameter
count, circuit evaluations. Write a 1-paragraph analysis of when quantum helped or didn't. A negative result is fine.
```

---

## Day 3 — Track D + Track F (~5 h)

**D1 — submodular + parallel (1.5 h)**
```
In trackD_submodular_parallel_distributed/:
1. Max-coverage objective: test diminishing returns on random samples, compare greedy with brute force for small k,
   and check the (1-1/e) bound. Show one non-submodular objective where greedy fails.
2. CPU-bound tasks of uneven size on multiprocessing with p = 1, 2, 4, max cores: static split vs LPT greedy vs
   dynamic queue (work stealing). Report makespan, speedup and efficiency (3 repeats), fit Amdahl's serial fraction,
   show one Gustafson scaled-size run, and one tiny-task case where parallel is slower (overhead).
   Add a lock-contention vs local-counter micro-benchmark (synchronization/locality).
3. A short simulated demo of data / model / pipeline parallelism on a tiny NumPy MLP (label it as a simulation).
```

**D2 — distributed + multi-task (1 h)**
```
In trackD_submodular_parallel_distributed/fed.py: FedAvg vs centralized logistic regression on make_classification
with 10 clients, local steps E ∈ {1, 5}, IID vs non-IID; plot accuracy vs rounds.
In multitask.py: a two-head model on synthetic data comparing equal weighting, MGDA (2-task closed form) and GradNorm.
Finish with a strategy-selection table for the methods in tracks A–D.
```

**F1 — mini-compiler + passes (2 h)**
```
In trackF_compiler/: a mini-compiler in Python for a toy language (int vars, + - * /, assignment, if, while, print):
lexer → recursive-descent parser → AST → three-address IR → IR interpreter that counts executed instructions.
Passes (each switchable): constant folding + propagation, dead-code elimination, common sub-expression elimination.
6 example programs. Measure static and executed instruction count, run time and compile time with passes off vs on.
Test: optimized output == unoptimized output for every program. Keep it simple — I must explain it in a viva.
```
*(If time: add strength reduction or greedy graph-colouring register allocation as the "formulated decision". The brief requires one compiler decision posed as an optimization problem, so greedy colouring with K registers to minimize spills is the cheapest way to cover it.)*

**F2 — register allocation (30 min)**
```
In trackF_compiler/regalloc.py: build the interference graph from liveness on the IR and allocate K = 2, 3 registers
with greedy graph colouring (spill the highest degree first). Report spills per program and verify that no two
interfering variables share a register.
```

---

## Day 4 — Track G + Track H (~5 h)

**G1 — xv6 scheduler (2.5 h, test after every edit)**
```
In xv6-riscv at trackG_xv6/xv6:
1. Add a syscall getpinfo that returns per-process pid, ticks run, ticks runnable-waiting, creation tick and first-run tick.
2. Add a priority scheduler (or a simple 3-level MLFQ), selectable with SCHED=RR|PRIO in the Makefile so the original
   round-robin remains for comparison.
3. User programs cpubound, iobound and mixed, plus schedbench, which forks a mix and prints turnaround and response time.
Minimal, commented changes; explain the proc.c locking. It must boot and pass `usertests -q`.
```
Then run `schedbench` under both schedulers plus `usertests -q`, save the output to `results/trackG/`, and run `git diff > trackG_xv6/sched.patch`.
*If the toolchain fails:* ask "Give me a Dockerfile to build and run xv6-riscv with qemu on an Apple Silicon Mac."

**H1 — index selection (2.5 h)**
```
In trackH_dbms/ with SQLite: load TPC-H at scale factor 0.1 (e.g. generate with DuckDB's tpch extension, export to
SQLite); document the row counts. Use 6 TPC-H-style queries as the workload.
Pick ~10 candidate indexes from the WHERE/JOIN columns; measure each one's size and its standalone benefit
(median of 5 runs). Index selection under a storage budget = 0/1 knapsack: solve with greedy (benefit/size) and
ILP (PuLP). Build each chosen set and measure the real workload latency vs the no-index default, for 2 budgets.
Save EXPLAIN QUERY PLAN before/after. Test: query result hashes are identical in every configuration.
```

---

## Day 5 — Track E + Track J + finish (~6 h)

**E1 — formulation (20 min)**
```
Formulate exam timetabling for ~40 courses as an optimization problem: courses sharing students can't share a slot,
rooms have capacity, and we minimize slots used plus back-to-back exams. Give decision variables, objective, hard/soft
constraints, assumptions and metrics. Ask me for missing information instead of inventing it.
Propose a seeded synthetic enrolment dataset.
```

**E2 — methods (1.5 h)**
```
In trackE_case_study/: implement (a) a first-fit baseline in given order, (b) greedy DSATUR colouring, and (c) an ILP
in PuLP for the formulation in README. Run on 20 and 40 courses (3 seeds). Report objective, slots, back-to-backs,
violations (must be 0) and runtime. Add a strategy-selection table (variable type, size, exact vs heuristic, hardware,
parallelism). Test: an independent checker validates the hard constraints.
```

**J1 — test-suite minimization (1.5 h)**
```
In trackJ_software_eng/: clone a small public Python project with >1,000 lines and a fast pytest suite (suggest 2,
e.g. tabulate or python-slugify). Profile per-test runtime and per-test line coverage (coverage.py --cov-context=test).
Formulate: x_t ∈ {0,1}, minimize total runtime s.t. every line covered by the full suite stays covered (set cover).
Solve with greedy and ILP; compare with the full suite. Report runtime saved and coverage retained, and briefly note
the maintainability risk.
```

**V1 — review + checklist + report (2 h)**
```
Act as a skeptical reviewer: run every track's run.py and pytest from a clean venv, and list bugs, unfair comparisons,
missing variance for stochastic results, and unsupported speedup claims. Then fill docs/chapter6_checklist.md
(every Chapter 6 method → track, file, one-line result, simulated/measured) and draft docs/report.md with one section
per track plus answers to the 10 final questions in the brief. Use only numbers that exist in results/.
```

**Final manual checks (45 min):**
- [ ] Compare the checklist against your class's Chapter 6 slides and fill any gaps
- [ ] Failure log ≥3 entries, tool comparison done, AI tools register complete
- [ ] Remove secrets from `prompts/` and push to a **public** GitHub repo
- [ ] Short demo (4–5 slides from report.md)
- [ ] Viva: explain one file per track without AI

---

## If you fall behind

Cut in this order: F2 → D1 part 3 (parallelism demos) → C2 embedding note → second budget in H.
**Never cut:** baselines, correctness tests, prompt logs, and the failure log.
