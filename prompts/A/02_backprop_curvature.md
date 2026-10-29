# A2

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** b2f1aa8

## Original prompt
```
In trackA_continuous/mlp.py: a 1-hidden-layer MLP on make_moons with backprop written by hand. Check the gradients
against PyTorch autograd, then train with SGD vs Adam, with and without L2 regularization.
In trackA_continuous/curvature.py: for logistic regression, compute the Hessian, a diagonal approximation, the
empirical Fisher, and a Hessian-vector product via torch double-backward (check HVP == H @ v). Plot log(f - f*) for
the convex vs strongly-convex (L2) cases to show sublinear vs linear convergence.
```

## AI response (excerpt)
Generated `mlp.py` (hand-written forward/backward for a tanh-sigmoid MLP, L2 term, SGD/Momentum/Adam on parameter dicts)
and `a2_experiments.py` (autograd gradient check, L2 sweep, Hessian / diagonal / empirical Fisher / HVP on logistic regression,
convex vs strongly-convex convergence plots).

## Revised prompt / corrections
- Fisher vs Hessian comparison moved from w = 0 (where they are identical) to the optimum (failure log #6).
- Strongly-convex rate: fit only above the 1e-10 floor and report rho instead of a meaningless log-log slope.
- Learning-rate grid for preconditioned vs plain GD made finer and identical (failure log #7); iterations 300 -> 5000;
  diagonal Hessian switched to the closed form, checked against HVPs with unit vectors.
- Added lambda = 0.001 to show mild regularization helping test accuracy.
- Surprising result (GD beats diagonal preconditioning) kept, and explained by projecting the initial error onto the
  Hessian eigenvectors.

## Verification note
Backprop vs autograd max difference 2.8e-17; HVP vs H v 2e-15; diagonal formula == HVP diagonal (assert in code).
Convergence plot inspected visually.
