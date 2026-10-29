"""Track I part 2: quantum machine learning vs classical models on the same data (simulation only).

Quantum models: (a) quantum-kernel SVM (fidelity kernel |<phi(x)|phi(x')>|^2 from a hand-written 2-qubit feature map),
(b) variational quantum classifier = the data re-uploading PQC from Track C, trained with Adam.
Classical models: RBF SVM, logistic regression, small MLP.
Same 5 random train/test splits for every model; report accuracy, training time, parameter count and the number of
circuit runs a real device would need (the simulator computes exact statevectors instead).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trackC_quantum_opt"))
from common.utils import save_results  # noqa: E402
import pqc  # noqa: E402  (variational classifier circuit from Track C)

TRACK = "trackI_quantum_algos_qml"


def feature_map(x, reps=2):
    """ZZ-type feature map on 2 qubits: H, RZ(2 x_i), then an entangling RZZ-like block with angle 2(pi - x0)(pi - x1)."""
    qc = QuantumCircuit(2)
    for _ in range(reps):
        qc.h([0, 1])
        qc.rz(2 * x[0], 0)
        qc.rz(2 * x[1], 1)
        qc.cx(0, 1)
        qc.rz(2 * (np.pi - x[0]) * (np.pi - x[1]), 1)
        qc.cx(0, 1)
    return qc


def quantum_kernel(XA, XB):
    """K_ij = |<phi(a_i)|phi(b_j)>|^2. On hardware each entry is a separate circuit estimated with shots."""
    SA = np.array([Statevector(feature_map(x)).data for x in XA])
    SB = np.array([Statevector(feature_map(x)).data for x in XB])
    return np.abs(SA.conj() @ SB.T) ** 2


def run_split(seed, n=200, vqc_steps=60):
    X, y = make_moons(n, noise=0.2, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    lo, hi = Xtr.min(0), Xtr.max(0)
    scale = lambda Z: (Z - lo) / (hi - lo)  # noqa: E731  -> [0, 1] using training statistics only
    Ztr, Zte = scale(Xtr), scale(Xte)
    rows = []

    # Quantum kernel SVM. Hyperparameters tuned by 5-fold CV on the training set only, like every model below:
    # the feature scale s (features mapped to [0, s*pi] -> controls how fast the kernel decays) and C.
    t0 = time.perf_counter()
    best = None
    for s_ in (0.05, 0.1, 0.25, 0.5, 1.0):
        Ktr = quantum_kernel(Ztr * s_ * np.pi, Ztr * s_ * np.pi)
        gs = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10, 100]}, cv=5).fit(Ktr, ytr)
        if best is None or gs.best_score_ > best[0]:
            best = (gs.best_score_, s_, gs.best_estimator_)
    _, s_best, svc = best
    t_fit = time.perf_counter() - t0
    acc = svc.score(quantum_kernel(Zte * s_best * np.pi, Ztr * s_best * np.pi), yte)
    ntr, nte = len(ytr), len(yte)
    rows.append({"model": "Quantum kernel SVM", "test_acc": acc, "train_time_s": t_fit,
                 "params": f"{len(svc.support_)} support vectors", "tuned": f"scale={s_best}, C={svc.C}",
                 "circuit_runs_on_hardware": ntr * (ntr - 1) // 2 + nte * ntr})

    # Variational quantum classifier (Track C circuit, features in [-pi/2, pi/2], labels +-1), Adam on simulator
    Vtr, Vte = Ztr * np.pi - np.pi / 2, Zte * np.pi - np.pi / 2
    ytr_pm, yte_pm = 2 * ytr - 1, 2 * yte - 1
    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.uniform(-np.pi, np.pi, (pqc.LAYERS, 2, 2)), requires_grad=True)
    opt = qml.AdamOptimizer(0.1)
    cost = lambda t: pnp.mean((pqc.classifier(Vtr[:, 0], Vtr[:, 1], t) - ytr_pm) ** 2)  # noqa: E731
    t0 = time.perf_counter()
    for _ in range(vqc_steps):
        theta = opt.step(cost, theta)
    t_fit = time.perf_counter() - t0
    acc = np.mean(np.sign(pqc.classifier(Vte[:, 0], Vte[:, 1], theta)) == yte_pm)
    n_par = theta.size
    rows.append({"model": "Variational quantum classifier", "test_acc": float(acc), "train_time_s": t_fit,
                 "params": f"{n_par} circuit angles", "tuned": "fixed: Adam lr=0.1, 60 steps (not tuned)",
                 # parameter-shift on hardware: 2 runs per parameter per training point per step
                 "circuit_runs_on_hardware": vqc_steps * 2 * n_par * ntr})

    # Classical baselines (same scaled features)
    for name, model, grid, count in [
        ("RBF SVM", SVC(kernel="rbf"), {"C": [0.1, 1, 10], "gamma": ["scale", 1, 10, 30, 100]},
         lambda m: f"{len(m.support_)} support vectors"),
        ("Logistic regression", LogisticRegression(), {"C": [0.1, 1, 10]},
         lambda m: f"{m.coef_.size + m.intercept_.size} weights"),
        ("MLP (16 hidden)", MLPClassifier((16,), solver="lbfgs", max_iter=5000, random_state=seed),
         {"alpha": [1e-4, 1e-2, 1]},
         lambda m: f"{sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)} weights"),
    ]:
        t0 = time.perf_counter()
        gs = GridSearchCV(model, grid, cv=5).fit(Ztr, ytr)
        m = gs.best_estimator_
        rows.append({"model": name, "test_acc": m.score(Zte, yte), "train_time_s": time.perf_counter() - t0,
                     "params": count(m), "tuned": str(gs.best_params_), "circuit_runs_on_hardware": 0})
    for r in rows:
        r["seed"] = seed
    return rows


def main(seeds=5):
    pd.set_option("display.width", 200)
    raw = pd.DataFrame([r for s in range(seeds) for r in run_split(s)])
    save_results(raw, TRACK, "i2_qml_raw")
    summary = (raw.groupby("model", sort=False)
                  .agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                       train_time_s=("train_time_s", "mean"), params_seed0=("params", "first"),
                       tuned_seed0=("tuned", "first"),
                       circuit_runs_on_hardware=("circuit_runs_on_hardware", "first"))
                  .reset_index())
    save_results(summary, TRACK, "i2_qml_vs_classical")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
