# Track A — Continuous optimization

Run: `python trackA_continuous/run.py` (≈ 30 s on the M4) · Tests: `pytest tests/test_trackA.py`
All numbers below come from `results/trackA_continuous/*.csv`.

## Objective
Compare the Chapter 6 continuous optimizers (GD, SGD, mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton,
Gauss–Newton) on convex and non-convex objectives, and show backpropagation, curvature approximations, convergence
rates, step-size sensitivity and noisy objectives.

## Formulation
- Decision variables: parameters x ∈ ℝ² (test functions), w ∈ ℝ²⁰ (linear regression), MLP weights, logistic-regression weights.
- Objectives (all hand-coded with analytic gradient and Hessian — `functions.py`, `problems.py`):
  | Objective | Type | Minimum |
  |---|---|---|
  | `0.5 xᵀAx`, A = diag(1, 100) | convex, ill-conditioned (κ = 100) | x = 0 |
  | Rosenbrock | non-convex, curved valley | (1, 1) |
  | `x² + y⁴/4 − y²/2` | non-convex, **saddle at (0,0)**, minima at (0, ±1) | f* = −0.25 |
  | `a·exp(b·t)` fit | nonlinear least squares | SciPy reference |
  | linear regression, n = 2000 | finite sum (noisy sampled gradients) | closed form |
  | MLP / logistic regression | ML loss + L2 regularizer | — |

## Baseline and fairness
Plain gradient descent is the baseline. Every method gets the **same start point, iteration budget (5000), tolerance
(‖∇f‖ < 1e-8) and the same learning-rate grid**; the best lr per method is reported. Stochastic methods: lr tuned on
seed 0, then evaluated on 5 fresh seeds (mean ± std), with an equal budget of sample-gradients per epoch.

## Results

**1. Landscapes** (`a1_landscapes.csv`, `a1_<function>.png`)

| Function | GD | Momentum | NAG | AdaGrad | RMSProp | Adam | Newton |
|---|---|---|---|---|---|---|---|
| Quadratic (iters to converge) | 1859 | 390 | 246 | 20 | 35 | 363 | **1** |
| Rosenbrock (iters; ✗ = not converged in 5000) | ✗ (f = 3.8e-3) | 1277 | 4179 | ✗ (6.9e-5) | ✗ (3.1e-4) | 1230 | **6** |
| Saddle (where it ends) | min (0, 1) | min | min | min | min | min | **saddle (0, 0)** ✗ |

- Ill-conditioning slows GD; momentum/NAG fix most of it; adaptive methods rescale per coordinate (A is diagonal, which favours AdaGrad/RMSProp here).
- **Newton converges in 1 step on the quadratic and 6 on Rosenbrock, but on the saddle function it goes straight to the saddle point** (f = 0, not −0.25), because it heads to any stationary point and does not check that the Hessian is positive definite. First-order methods escape because the tiny y-component grows each step.
- Learning-rate sensitivity: `a1_lr_sweep.png`. GD on Rosenbrock only works in a narrow band (best lr = 1e-3, worse at 3e-3, diverges for lr ≥ 0.01); Adam reaches f < 1e-9 for every lr from 0.01 to 0.3.

**2. Stochastic / noisy gradients** (`a1_stochastic.csv`, linear regression, 10 epochs, 5 seeds)

| Method | updates/epoch | f − f* (mean ± std) |
|---|---|---|
| GD (full batch) | 1 | 0.254 |
| SGD (B = 1) | 2000 | 1.9e-4 ± 4.7e-5 |
| Mini-batch SGD (B = 32) | 63 | 2.4e-4 ± 6.5e-5 |
| Mini-batch Momentum (B = 32) | 63 | 1.7e-4 ± 5.8e-5 |
| Mini-batch Adam (B = 32) | 63 | **5.0e-5 ± 9.6e-6** |

Same number of sample-gradients for all: full-batch GD makes only 10 updates and is far behind. SGD with B = 1 is
competitive per sample but ~25× slower in wall-time than B = 32 (Python loop overhead per update); constant-lr SGD stalls
at a noise floor instead of reaching f*.

**3. Gauss–Newton** (`a1_gauss_newton.csv`): fitting `y = a·exp(b·t)`: Gauss–Newton 6 iterations, Newton 5, GD 233 (best
lr). All reach the SciPy `least_squares` solution (distance < 3e-9). Gauss–Newton needs only first derivatives and its
J ᵀJ is always positive semi-definite.

**4. Backpropagation** (`mlp.py`, `a2_mlp_training.csv`): hand-written backprop matches PyTorch autograd to
**2.8e-17**. Backprop computes the gradient; SGD/Momentum/Adam are separate update rules that consume it.
L2 regularization shrinks the weight norm 10.9 → 8.1 → 2.6 → 1.7 (λ = 0, 0.001, 0.01, 0.1); λ = 0.001 gives the best
test accuracy (0.928), larger λ underfits (0.84, 0.83).

**5. Curvature** (`a2_curvature.csv`, logistic regression, 10 badly scaled features)
- Hessian-vector product by double backward matches H·v to **2e-15** without forming H; diagonal Hessian formula checked against HVPs with unit vectors.
- At the optimum, empirical Fisher differs from the Hessian by 26% (relative Frobenius norm). (At w = 0 they are identical for logistic regression, so comparing there would be misleading.)
- Diagonal preconditioning cuts the Hessian condition number from 10,597 to 1,540, **but plain GD still converged faster** (1e-6 gap after 1304 iters vs ~2800). Reason, verified by projecting the initial error onto the Hessian eigenvectors: the two smallest eigenvalues (0.001, from the dataset's 2 redundant features) have *zero* component in the error, so GD's effective condition number is only 10.6 / 0.036 ≈ 294. The condition number is a worst-case bound, not a prediction.

**6. Convergence rates** (`a2_convergence.png`): GD on separable logistic regression without L2 (convex, not strongly
convex — minimum at infinity) decays sublinearly (log-log slope −0.84, close to O(1/k)); with L2 = 0.01 (strongly
convex) it converges linearly (rate ρ ≈ 0.975 per iteration) down to machine precision in ~800 iterations.

## Correctness checks (`tests/test_trackA.py`, all passing)
Finite-difference checks of every gradient and Hessian; Newton solves the quadratic in exactly 1 step; Newton converges
to the saddle; every first-order method minimizes the quadratic; Gauss–Newton matches SciPy; mini-batch gradient with
the full batch equals the full gradient.

## Limitations
2-D test functions are illustrative only; the learning-rate grid is coarse (factor ~3) for part 1; wall-times are
Python/NumPy timings on one machine and include interpreter overhead.
