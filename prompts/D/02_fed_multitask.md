# D2

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 5ef2fd7

## Original prompt
```
In trackD_submodular_parallel_distributed/fed.py: FedAvg vs centralized logistic regression on make_classification
with 10 clients, local steps E ∈ {1, 5}, IID vs non-IID; plot accuracy vs rounds.
In multitask.py: a two-head model on synthetic data comparing equal weighting, MGDA (2-task closed form) and GradNorm.
Finish with a strategy-selection table for the methods in tracks A–D.
```

## AI response (excerpt)
Generated `fed.py` (FedAvg with weighted averaging, IID vs label-skewed clients, E local epochs, centralized reference) and `multitask.py` (shared trunk + 2 heads, equal weights, MGDA two-task closed form, GradNorm with learnable weights).

## Revised prompt / corrections
- FedAvg first version saturated at the linear model's ~70% ceiling after 1 round, so no effect of E or non-IID was visible; made classes more separable and lr smaller (failure log #12).
- MGDA bug: the whole loss was scaled by alpha, so with alpha = 0 task 1's head got no gradient at all. Fixed: heads get their own loss gradient, only the shared trunk uses the min-norm combination; added the normalized-gradient variant (failure log #13).

## Verification note
MGDA closed-form alpha checked against a grid search (test); FedAvg client splits disjoint (test). Strategy-selection table written in README.
