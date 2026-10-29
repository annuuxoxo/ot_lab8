# C2

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 772039e

## Original prompt
```
In trackC_quantum_opt/pqc.py (PennyLane): a small PQC classifier on 2-feature toy data, trained with
(a) parameter-shift written by me (checked against qml.grad), (b) SPSA, (c) QNGOptimizer. Compare loss curves and
circuit evaluations. Then plot gradient variance vs qubit count (2–10 qubits, random circuits, 100 samples) to show
the barren plateau. Add a short note on QUBO embedding limits on annealers: run minorminer on complete graphs K_n
for growing n and report chain lengths.
```

## AI response (excerpt)
Generated `trackC_quantum_opt/pqc.py`: data re-uploading PQC classifier (PennyLane, batched), my own parameter-shift gradient and my own SPSA with equal circuit-run budgets, QNG vs GD on the VQE cost, a barren plateau study (random McClean-style circuits, deep vs shallow), and minorminer embeddings of K_n on Chimera / Pegasus.

## Revised prompt / corrections
- Prompt asked for QNG on the classifier; QNG needs a single-circuit cost (the metric tensor depends on the input), so QNG is demonstrated on the VQE cost instead (documented in README).
- QNG vs GD step-size grid widened from 0.05–0.4 to 0.05–1.6 after GD's best step was at the grid edge; QNG's apparent 2× advantage disappeared (failure log #9). Added a circuit-execution counter per step.
- Investigated identical test accuracies of both classifiers: real coincidence on a 40-point test set (per-seed values listed in README).

## Verification note
Parameter-shift vs autodiff max diff 3.5e-16 (test). Barren plateau plot inspected: straight line on semilog for deep circuits.
