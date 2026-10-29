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
