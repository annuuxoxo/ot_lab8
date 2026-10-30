# Chapter 6 Coverage Checklist

| Area | Method / concept | Track | File | Result (one line) | Simulated / measured |
|---|---|---|---|---|---|
| Foundations | ML as optimization (parameters, loss, regularization) | A | `trackA_continuous/mlp.py` | MLP = BCE + L2; λ shrinks ‖W‖ 10.9 → 1.7, best test acc at λ = 0.001 | measured |
| Foundations | Continuous vs discrete optimization | A, B | Track A vs Track B | Same "selection" idea: continuous weights (A) vs binary x_i (B) | measured |
| Foundations | Convex vs non-convex landscapes | A | `functions.py` | Quadratic (convex) vs Rosenbrock / saddle function | measured |
| Foundations | Local/global minima, saddle points | A | `functions.Saddle` | Newton converges to the saddle (0,0); first-order methods reach minima (0, ±1) | measured |
| Continuous | Gradient descent | A | `optimizers.GD` | Quadratic 1859 iters; Rosenbrock not converged in 5000 | measured |
| Continuous | SGD, mini-batch SGD | A | `a1_experiments.stochastic` | Lin. reg., 10 epochs: SGD 1.9e-4, mini-batch 2.4e-4 vs GD 0.25 (5 seeds) | measured |
| Continuous | Momentum | A | `optimizers.Momentum` | Rosenbrock 1277 iters | measured |
| Continuous | Nesterov (NAG) | A | `optimizers.NAG` | Quadratic 246 iters (vs GD 1859) | measured |
| Continuous | AdaGrad | A | `optimizers.AdaGrad` | Quadratic 20 iters; Rosenbrock stalls (step size decays) | measured |
| Continuous | RMSProp | A | `optimizers.RMSProp` | Quadratic 35 iters | measured |
| Continuous | Adam | A | `optimizers.Adam` | Rosenbrock 1230 iters; robust to lr 0.01–0.3 | measured |
| Continuous | Newton | A | `optimizers.Newton` | 1 iter quadratic, 6 Rosenbrock; attracted to saddle | measured |
| Continuous | Gauss–Newton | A | `optimizers.GaussNewton` | Exp-fit in 6 iters (GD 233), matches SciPy | measured |
| Continuous | Backpropagation (separate from optimizer) | A | `mlp.backward` | Matches autograd to 2.8e-17 | measured |
| Continuous | Fisher, diagonal approximations | A | `a2_experiments.curvature_experiment` | Empirical Fisher 26% from H at w*; diag precond κ 10,597 → 1,540 | measured |
| Continuous | Hessian-vector products | A | `a2_experiments.hvp` | Double-backward HVP = H·v to 2e-15 | measured |
| Continuous | Convergence: convex / strongly convex / non-convex | A | `a2_experiments.convergence_experiment`, `a1_*` | Convex: O(1/k) slope −0.84; strongly convex: linear ρ = 0.975; non-convex: saddle/Rosenbrock | measured |
| Continuous | Step-size / optimizer tuning | A | `a1_lr_sweep.csv` | GD on Rosenbrock diverges for lr ≥ 0.01; Adam fine 0.01–0.3 | measured |
| Continuous | Noisy objectives | A | `a1_experiments.stochastic` | Constant-lr SGD stalls at a noise floor; Adam lowest (5e-5) | measured |
| Discrete | Greedy | B | `knapsack.greedy`, feature selection | Knapsack gap ≤ 2.7%; feature selection suboptimal at k = 3 | measured |
| Discrete | Dynamic programming | B | `knapsack.dp` | Exact; 1.4 ms at n = 200 (pseudo-polynomial in C) | measured |
| Discrete | Branch-and-Bound | B | `knapsack.branch_and_bound` | Exact; ~760 nodes at n = 200 | measured |
| Discrete | ILP | B | `knapsack.ilp` (PuLP/CBC) | Exact; ~20 ms (solver start-up dominated) | measured |
| Discrete | Feature selection (combinatorial) | B | `b_experiments.feature_selection_experiment` | Greedy 6–16× cheaper, misses optimum at k = 3 | measured |
| Discrete | QUBO / Ising, binary↔spin, penalty terms | B | `qubo.py` | Max-Cut QUBO = −cut = Ising verified; penalty too large → SA success 0% | measured |
| Foundations | Basic quantum computing concepts, role of optimization in QML | C, I | `trackI_quantum_algos_qml/`, `trackC_quantum_opt/pqc.py` | Superposition/interference (DJ, BV), variational circuits trained classically | simulated |
| Quantum | QAOA | C | `qaoa_vqe.py` | Max-Cut ratio 0.78→0.92 (p = 1→3, n = 6); SA (classical) = 1.00 | simulated |
| Quantum | VQE | C | `qaoa_vqe.vqe_experiment` | H₂ exact to 4e-8 Ha ideal; 28.6 mHa error with noise | simulated |
| Quantum | PQCs / QNNs | C, I | `pqc.classifier` | 12-angle classifier, test acc 0.975 (C), 0.958 (I) | simulated |
| Quantum | Parameter-shift rule | C | `pqc.parameter_shift_grad` | Equals autodiff to 3.5e-16 | simulated |
| Quantum | SPSA | C | `pqc.train_classifier` | Same budget: loss 0.239 vs 0.200 (param-shift); 2 runs/step regardless of #params | simulated |
| Quantum | Quantum natural gradient | C | `pqc.qng_experiment` | 4× fewer steps at equal step size; ties GD once both tuned (6 steps) | simulated |
| Quantum | Barren plateaus | C | `pqc.barren_plateau_experiment` | Deep: Var 0.087 → 0.0022 (2 → 10 qubits); shallow flat ≈ 0.09 | simulated |
| Quantum | Finite-shot measurement cost | C | `qaoa_vqe.shots_experiment` | Std ∝ 1/√shots (0.16 / 0.047 / 0.011) | simulated |
| Quantum | Hardware noise | C, I | `common/quantum.py` noise model | QAOA p > 1 gains erased; Grover n = 5 → 9% success | simulated (noise model) |
| Quantum | Limits of embedding QUBOs on annealers | C | `pqc.embedding_experiment` | K₃₆ does not embed on Chimera C8 (512 qubits); chains up to 13 | simulated (classical embedding) |
| Quantum | Deutsch–Jozsa | I | `algorithms.deutsch_jozsa` (hand-written) | 1 query, 100% ideal | simulated |
| Quantum | Bernstein–Vazirani | I | `algorithms.bernstein_vazirani` (hand-written) | 1 query, 100% ideal, 85–90% noisy | simulated |
| Quantum | Grover | I | `algorithms.grover` (hand-written) | 25 vs 528 queries at N = 1024; simulator 6000× slower than classical | simulated |
| Quantum | Quantum Fourier Transform | I | `algorithms.qft` (hand-written) | = DFT matrix to 3.8e-15 | simulated |
| Quantum | Quantum Phase Estimation | I | `algorithms.qpe` | T gate φ = 1/8 exact; φ = 1/3 needs more bits | simulated |
| Quantum | Shor | I | `algorithms.shor15` | 15 = 3 × 5, 51.6% of shots; 13,328 CX gates | simulated |
| Quantum | Quantum annealing | I | `algorithms.annealing_experiment` | P(opt) 0.04 → 0.99 as anneal time 0.5 → 50 | simulated (exact time evolution) |
| Quantum | Variational classifier | I | `qml_vs_classical.py` | 0.958 ± 0.013 vs RBF SVM 0.962 ± 0.019 | simulated |
| Quantum | Quantum kernels | I | `qml_vs_classical.quantum_kernel` | 0.934 ± 0.021 after tuning feature scale (0.744 untuned) | simulated |
| Submodular | Diminishing returns, testing assumptions | D | `submodular.test_assumptions` | Coverage: 0% violations; pair-bonus objective: 14.3% violations | measured |
| Submodular | Greedy selection + checking the (1−1/e) guarantee | D | `submodular.greedy` | Ratio 1.00 when submodular; 0.61 (k = 3) and 0.01 (adversarial) when not | measured |
| Parallel | Load balancing, makespan | D | `parallel.py` | Static 5.5× vs LPT/dynamic 6.8× at p = 10 (imbalance 1.38 → 1.06) | measured |
| Parallel | Work stealing | D | `parallel.work_stealing` | 6.7× at p = 10, 19 steals | measured |
| Parallel | Locality | D | `parallel.sync_locality_experiment` | Row-wise 12.5 ms vs column-wise 32.4 ms | measured |
| Parallel | Synchronization | D | `parallel.sync_locality_experiment` | Shared lock 69.6 ms vs private counters 13.7 ms | measured |
| Parallel | Communication overhead | D | `parallel.overhead_experiment`, `parallelism_demo.py` | Tiny tasks: 0.37× at 10 threads; floats communicated per scheme | measured / simulated |
| Parallel | Data, model, pipeline parallelism | D | `parallelism_demo.py` | All match the single worker exactly; pipeline bubble 50% → 3% | simulated |
| Parallel | Speedup, efficiency, Amdahl, Gustafson | D | `parallel.py` | Amdahl serial fraction ≈ 0.05; Gustafson scaled speedup 6.96 at p = 10 | measured |
| Strategy selection | Choosing methods by variable type, size, hardware, communication | D, E | Track D README table | Table for all methods in A–D | — |
| Distributed | FedAvg vs centralized | D | `fed.py` | E = 1/5/20 → 34/7/2 rounds; non-IID slower | simulated |
| Multi-task | MGDA | D | `multitask.py` | Raw MGDA ignores the large-gradient task; normalized MGDA best on both tasks | measured (CPU) |
| Multi-task | GradNorm | D | `multitask.py` | Weights → [0.013, 1.987]; task 2 acc 0.970, task 1 MSE 132.5 | measured (CPU) |
| Discrete (applied) | Greedy vs ILP for register allocation | F | `trackF_compiler/regalloc.py` | Greedy optimal in 30/32 cases; ILP exact | measured |
| Discrete (applied) | Exhaustive vs greedy search for pass ordering | F | `f_experiments.pass_ordering` | Greedy picks the worst of 720 orders | measured |
| Discrete (applied) | Greedy vs DP vs ILP for index selection (knapsack) | H | `trackH_dbms/index_selection.py` | DP = ILP always; greedy equal at 10/25% budgets; up to 3.11× speedup at 50% budget | measured |
| Discrete (applied) | Interaction-aware (measured) greedy | H | `index_selection.measured_greedy` | 1.36× vs 1.08× for knapsack methods at 10% budget | measured |
| Scheduling (applied) | MLFQ as an online scheduling heuristic | G | `trackG_xv6/mlfq_scheduler.patch` | Interactive turnaround 33 → 13 ticks; no CPU-throughput cost | measured (QEMU) |
| Discrete (applied, main case study) | First-fit vs DSATUR vs local search vs ILP for exam timetabling | E | `trackE_case_study/timetabling.py` | 40 courses: objective −32% (ILP) / −29% (local search) vs first-fit; ILP period = lower bound on all seeds; plain DSATUR worse than baseline | measured |
| Discrete (applied) | Lower bounds to certify heuristics (clique + room-capacity bound) | E | `timetabling.clique_lower_bound` | Room bound 19–20 slots vs clique 9–11 at 40 courses; rooms are the binding constraint | measured |
| Strategy selection | Method choice for the main case study | E | `trackE_case_study/README.md` §5 | Heuristic + local search at registry scale; ILP for small instances and bounds | — |
| Discrete (applied) | Greedy vs ILP weighted set cover (test-suite minimization) | J | `trackJ_software_eng/minimize.py` | tabulate: 322 → 68 tests, 100% line coverage kept, 2.5× faster wall time; greedy within 0.1% of ILP | measured |
| Discrete (applied) | Fixed-charge set cover (shared setup costs) | J | `minimize.ilp(needs, setup)` | Naive per-test costs under-predicted the suite 2.5×; fixed-charge model within 10% of measured | measured |
