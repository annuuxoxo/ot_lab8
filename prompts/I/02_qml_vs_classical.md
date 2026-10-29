# I2

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 9252bf4

## Original prompt
```
In trackI_quantum_algos_qml/qml.py: on make_moons (same split, 5 seeds), compare a quantum-kernel SVM (fidelity
kernel, simulator) with an RBF SVM and logistic regression. Table: accuracy mean ± std, training time, parameter
count, circuit evaluations. Write a 1-paragraph analysis of when quantum helped or didn't. A negative result is fine.
```

## AI response (excerpt)
Generated `trackI_quantum_algos_qml/qml_vs_classical.py`: quantum kernel SVM (fidelity kernel from a hand-written ZZ feature map, statevector), variational classifier reusing the Track C circuit, RBF SVM, logistic regression, MLP; 5 splits; accuracy, train time, parameters and hardware-equivalent circuit runs.

## Revised prompt / corrections
- First version used default hyperparameters: the quantum kernel got 0.744 and the MLP 0.858, both artifacts of the setup. Revised so every model is tuned by 5-fold CV on the training set; quantum kernel feature scale added as a hyperparameter (failure log #10).
- Then two chosen values sat at grid edges (kernel scale 0.25, RBF gamma 10); grids widened until choices were interior.

## Verification note
Kernel matrix symmetric, unit diagonal, PSD (test). Chosen hyperparameters per seed checked in `i2_qml_raw.csv`.
