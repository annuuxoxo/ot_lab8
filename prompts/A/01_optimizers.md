# A1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** b2f1aa8

## Original prompt
```
In trackA_continuous/, implement in NumPy: GD, SGD, mini-batch SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam,
Newton, and Gauss–Newton (fit y = a*exp(b*x) to synthetic data).
Functions: ill-conditioned convex quadratic, Rosenbrock (non-convex), and f = x^2 - y^2 started near the saddle.
SGD variants get Gaussian gradient noise (fixed seed). Same start, iteration budget and tolerance for every method.
Output: loss curves, a results table (final value, iterations, time), and one learning-rate sweep on Rosenbrock for
GD/Momentum/Adam. Tests: finite-difference gradient check; Newton solves the quadratic in one step.
```

## AI response (excerpt)
Generated `functions.py` (quadratic, Rosenbrock, saddle with analytic grad/Hessian), `optimizers.py` (GD, Momentum,
NAG, AdaGrad, RMSProp, Adam, Newton, Gauss-Newton with one shared `run()` loop), `problems.py` (exp-fit least squares,
finite-sum linear regression), `a1_experiments.py`, `tests/test_trackA.py`.
Design choice made by the AI: instead of adding artificial Gaussian noise to SGD, SGD/mini-batch run on a real finite-sum
problem (sampled gradients are the noise), with an equal sample-gradient budget per epoch.
The saddle function was changed from x^2 - y^2 (unbounded below) to x^2 + y^4/4 - y^2/2 so methods have a minimum to reach.

## Revised prompt / corrections
- Start point of the quadratic moved from (1, 1) to (1.3, 0.7): AdaGrad was "solving" it in 2 steps by accident (failure log #5).
- README claim about the GD lr range corrected after checking `a1_lr_sweep.csv` (said diverges at 3e-3; it diverges at 1e-2).

## Verification note
`pytest tests/test_trackA.py` — all pass (finite-difference grad/Hessian checks, Newton 1 step on quadratic,
Gauss-Newton == SciPy least_squares). Results inspected; saddle plot checked visually (Newton stops at (0,0)).
