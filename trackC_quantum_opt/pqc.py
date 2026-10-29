"""Track C part 2 (PennyLane, simulation only):
1. A parameterized quantum circuit (PQC) classifier trained with my own parameter-shift gradients vs my own SPSA.
2. Quantum natural gradient (QNG) vs plain gradient descent on a VQE cost (QNG needs a single-circuit cost).
3. Barren plateaus: gradient variance vs number of qubits, deep vs shallow random circuits.
4. Limits of embedding QUBOs on annealing hardware: minorminer embeddings of complete graphs on Chimera / Pegasus.
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import make_moons

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import save_plot, save_results  # noqa: E402

TRACK = "trackC_quantum_opt"
LAYERS = 3
dev2 = qml.device("default.qubit", wires=2)


# ---------------- 1. PQC classifier ----------------

@qml.qnode(dev2)
def classifier(x0, x1, theta):
    """Data re-uploading circuit: each layer encodes the 2 features, then trainable RY/RZ + CNOT.
    x0, x1 may be arrays (a whole batch is simulated at once via broadcasting). Output: <Z on qubit 0> in [-1, 1]."""
    for l in range(LAYERS):
        qml.RY(x0, wires=0)
        qml.RY(x1, wires=1)
        for q in range(2):
            qml.RY(theta[l, q, 0], wires=q)
            qml.RZ(theta[l, q, 1], wires=q)
        qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


def load_data(seed=0):
    X, y = make_moons(120, noise=0.1, random_state=seed)
    X = (X - X.min(0)) / (X.max(0) - X.min(0)) * np.pi - np.pi / 2  # features in [-pi/2, pi/2]
    y = 2 * y - 1  # labels in {-1, +1}
    return X[:80], y[:80], X[80:], y[80:]


def loss(theta, X, y):
    return np.mean((classifier(X[:, 0], X[:, 1], theta) - y) ** 2)


def accuracy(theta, X, y):
    return np.mean(np.sign(classifier(X[:, 0], X[:, 1], theta)) == y)


def parameter_shift_grad(theta, X, y):
    """d loss / d theta_k = mean_i 2 (f_i - y_i) * d f_i / d theta_k, with the exact parameter-shift rule
    d f / d theta_k = [f(theta + pi/2 e_k) - f(theta - pi/2 e_k)] / 2   (valid for Pauli-rotation gates).
    Cost: 2 circuit runs per parameter per data point."""
    f = classifier(X[:, 0], X[:, 1], theta)
    grad = np.zeros_like(theta)
    for idx in np.ndindex(theta.shape):
        shift = np.zeros_like(theta)
        shift[idx] = np.pi / 2
        df = (classifier(X[:, 0], X[:, 1], theta + shift) - classifier(X[:, 0], X[:, 1], theta - shift)) / 2
        grad[idx] = np.mean(2 * (f - y) * df)
    return grad


def train_classifier(seed=0, budget_evals=None):
    Xtr, ytr, Xte, yte = load_data()
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi, np.pi, (LAYERS, 2, 2))
    n_par, n = theta0.size, len(ytr)
    steps_ps = 40
    budget = budget_evals or steps_ps * 2 * n_par * n  # parameter-shift budget, in single-circuit runs
    curves, rows = {}, []

    # (a) gradient descent with parameter-shift gradients
    theta, hist, evals = theta0.copy(), [], 0
    t0 = time.perf_counter()
    for _ in range(steps_ps):
        theta = theta - 0.3 * parameter_shift_grad(theta, Xtr, ytr)
        evals += 2 * n_par * n
        hist.append((evals, loss(theta, Xtr, ytr)))
    curves["Parameter-shift GD"] = hist
    rows.append({"method": "Parameter-shift GD", "circuit_runs": evals, "train_loss": hist[-1][1],
                 "train_acc": accuracy(theta, Xtr, ytr), "test_acc": accuracy(theta, Xte, yte),
                 "time_s": time.perf_counter() - t0})

    # (b) SPSA: perturb ALL parameters at once with a random +-1 vector; 2 loss evaluations per step regardless of
    # the number of parameters. Standard gain sequences a_k = a/(k+1+A)^0.602, c_k = c/(k+1)^0.101.
    theta, hist, evals = theta0.copy(), [], 0
    a, c, A = 0.3, 0.2, 10
    k = 0
    t0 = time.perf_counter()
    while evals + 2 * n <= budget:
        ak, ck = a / (k + 1 + A) ** 0.602, c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], theta.shape)
        g = (loss(theta + ck * delta, Xtr, ytr) - loss(theta - ck * delta, Xtr, ytr)) / (2 * ck) * delta
        theta = theta - ak * g
        evals += 2 * n
        k += 1
        if k % 10 == 0:
            hist.append((evals, loss(theta, Xtr, ytr)))
    curves["SPSA"] = hist
    rows.append({"method": f"SPSA ({k} steps)", "circuit_runs": evals, "train_loss": loss(theta, Xtr, ytr),
                 "train_acc": accuracy(theta, Xtr, ytr), "test_acc": accuracy(theta, Xte, yte),
                 "time_s": time.perf_counter() - t0})
    return pd.DataFrame(rows), curves


def classifier_experiment(seeds=3):
    # correctness: my parameter-shift gradient == PennyLane autodiff gradient
    Xtr, ytr, _, _ = load_data()
    th = np.random.default_rng(42).uniform(-np.pi, np.pi, (LAYERS, 2, 2))
    auto = qml.grad(lambda t: pnp.mean((classifier(Xtr[:, 0], Xtr[:, 1], t) - ytr) ** 2))(pnp.array(th, requires_grad=True))
    ps_err = float(np.abs(parameter_shift_grad(th, Xtr, ytr) - auto).max())
    print(f"max |parameter-shift grad - autodiff grad| = {ps_err:.2e}")

    dfs, all_curves = [], []
    for s in range(seeds):
        df, curves = train_classifier(seed=s)
        df["seed"] = s
        dfs.append(df)
        all_curves.append(curves)
    df = pd.concat(dfs)
    df["method_group"] = df["method"].str.replace(r" \(.*\)", "", regex=True)
    summary = df.groupby("method_group").agg(circuit_runs=("circuit_runs", "mean"),
                                             train_loss_mean=("train_loss", "mean"), train_loss_std=("train_loss", "std"),
                                             test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                                             time_s=("time_s", "mean")).reset_index()
    save_results(df, TRACK, "c2_classifier_raw")
    save_results(summary, TRACK, "c2_classifier")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, curves in enumerate(all_curves):
        for j, (name, h) in enumerate(curves.items()):
            e, l = zip(*h)
            ax.plot(e, l, color=f"C{j}", alpha=0.7, label=name if i == 0 else None)
    ax.set(xlabel="single-circuit runs", ylabel="training MSE", title="PQC classifier: parameter-shift vs SPSA (3 seeds)")
    ax.legend()
    save_plot(fig, TRACK, "c2_classifier")
    plt.close(fig)
    return summary, ps_err


# ---------------- 2. Quantum natural gradient ----------------

H2 = qml.Hamiltonian([-1.052373245772859, 0.39793742484318045, -0.39793742484318045, -0.01128010425623538,
                      0.18093119978423156],
                     [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1),
                      qml.PauliX(0) @ qml.PauliX(1)])


@qml.qnode(dev2)
def vqe_cost(theta):
    for layer in range(3):
        qml.RY(theta[2 * layer], wires=0)
        qml.RY(theta[2 * layer + 1], wires=1)
        if layer < 2:
            qml.CNOT(wires=[0, 1])
    return qml.expval(H2)


def qng_experiment(inits=5, max_steps=200):
    """Same random starts and the same step-size grid for GD and QNG; count steps to chemical accuracy (1.6 mHa).
    QNG preconditions the gradient with the (block-diagonal) Fubini-Study metric tensor of the circuit,
    i.e. it takes steps measured in 'distance between quantum states' rather than in raw parameter space."""
    exact = float(np.linalg.eigvalsh(qml.matrix(H2, wire_order=[0, 1])).min())
    rng = np.random.default_rng(0)
    starts = [rng.uniform(-np.pi, np.pi, 6) for _ in range(inits)]
    rows = []
    for name, make in [("GD", lambda s: qml.GradientDescentOptimizer(s)),
                       ("QNG", lambda s: qml.QNGOptimizer(s, approx="block-diag"))]:
        for step in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6):  # wide grid: GD's best was at the edge of a narrower one
            hits, finals, execs = [], [], []
            for th0 in starts:
                opt, th = make(step), pnp.array(th0, requires_grad=True)
                hit = None
                with qml.Tracker(dev2) as tracker:  # counts circuit executions inside opt.step only
                    for k in range(max_steps):
                        th = opt.step(vqe_cost, th)
                        steps_done = k + 1
                        if not np.isfinite(float(vqe_cost(th))):
                            break
                        if vqe_cost(th) - exact < 1.6e-3:
                            hit = k + 1
                            break
                # subtract the convergence checks (one execution per step) from the count
                execs.append((tracker.totals.get("executions", 0) - steps_done) / steps_done)
                hits.append(hit)
                finals.append(float(vqe_cost(th)) - exact)
            reached = [h for h in hits if h is not None]
            rows.append({"optimizer": name, "stepsize": step, "reached_chem_acc": f"{len(reached)}/{inits}",
                         "median_steps_to_chem_acc": float(np.median(reached)) if reached else None,
                         "circuit_execs_per_step": float(np.mean(execs)),
                         "final_error_mHa_median": 1000 * float(np.median(finals))})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "c2_qng")
    return df


# ---------------- 3. Barren plateaus ----------------

def random_circuit_grad(n, layers, rng):
    """Random hardware-efficient circuit (McClean et al. 2018): RY(pi/4) on all qubits, then per layer a random
    Pauli rotation (RX/RY/RZ) with a random angle on every qubit and a CZ ladder. Cost = <Z0 Z1>.
    Returns dCost/dtheta_0 by the parameter-shift rule."""
    dev = qml.device("default.qubit", wires=n)
    gates = rng.choice([qml.RX, qml.RY, qml.RZ], size=(layers, n))
    theta = rng.uniform(0, 2 * np.pi, (layers, n))

    @qml.qnode(dev)
    def cost(t):
        for q in range(n):
            qml.RY(np.pi / 4, wires=q)
        for l in range(layers):
            for q in range(n):
                gates[l, q](t[l, q], wires=q)
            for q in range(n - 1):
                qml.CZ(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    shift = np.zeros_like(theta)
    shift[0, 0] = np.pi / 2
    return (cost(theta + shift) - cost(theta - shift)) / 2


def barren_plateau_experiment(qubits=(2, 4, 6, 8, 10), samples=100):
    rng = np.random.default_rng(0)
    rows = []
    for n in qubits:
        for label, layers in [("deep (L = 5n)", 5 * n), ("shallow (L = 2)", 2)]:
            g = [random_circuit_grad(n, layers, rng) for _ in range(samples)]
            rows.append({"qubits": n, "circuit": label, "layers": layers, "grad_mean": np.mean(g), "grad_var": np.var(g)})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "c2_barren_plateau")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, g in df.groupby("circuit"):
        ax.semilogy(g["qubits"], g["grad_var"], "o-", label=label)
    ax.set(xlabel="qubits", ylabel="Var[dC/dθ₀] over random parameters",
           title=f"Barren plateau: cost <Z0 Z1>, {samples} random circuits per point")
    ax.legend()
    save_plot(fig, TRACK, "c2_barren_plateau")
    plt.close(fig)
    return df


# ---------------- 4. QUBO embedding on annealing hardware ----------------

def embedding_experiment(sizes=(4, 8, 12, 16, 20, 24, 28, 32, 36, 40)):
    """A dense QUBO (complete graph K_n, e.g. Max-Cut on a dense graph or a penalty term coupling all variables)
    must be minor-embedded: each logical variable becomes a chain of physical qubits."""
    import dwave_networkx as dnx
    import minorminer
    import networkx as nx
    targets = {"Chimera C8 (512 qubits)": dnx.chimera_graph(8), "Pegasus P6 (680 qubits)": dnx.pegasus_graph(6)}
    rows = []
    for name, T in targets.items():
        for n in sizes:
            t0 = time.perf_counter()
            emb = minorminer.find_embedding(nx.complete_graph(n).edges, T.edges, random_seed=0, timeout=20, tries=10)
            ok = bool(emb) and len(emb) == n
            chains = [len(c) for c in emb.values()] if ok else []
            rows.append({"hardware": name, "logical_vars_n": n, "embedded": ok,
                         "physical_qubits": sum(chains) if ok else None,
                         "max_chain": max(chains) if ok else None,
                         "mean_chain": float(np.mean(chains)) if ok else None,
                         "time_s": time.perf_counter() - t0})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "c2_embedding")
    return df


def main():
    pd.set_option("display.width", 200)
    summary, _ = classifier_experiment()
    print("== Classifier ==\n", summary.to_string(index=False))
    print("\n== QNG vs GD ==\n", qng_experiment().to_string(index=False))
    print("\n== Barren plateau ==\n", barren_plateau_experiment().to_string(index=False))
    print("\n== Embedding ==\n", embedding_experiment().to_string(index=False))


if __name__ == "__main__":
    main()
