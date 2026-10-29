"""Track C part 1: QAOA for Max-Cut (Track B graphs), VQE for H2, finite-shot and noise studies.
Everything is SIMULATED (Qiskit Aer); no real quantum hardware is used."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.quantum import BASIS, bits_of, noise_model, run_counts  # noqa: E402
from common.utils import RESULTS, save_plot, save_results  # noqa: E402

TRACK = "trackC_quantum_opt"


# ---------------- QAOA ----------------

def load_graphs():
    with open(os.path.join(RESULTS, "trackB_discrete", "maxcut_graphs.json")) as f:
        data = json.load(f)
    out = []
    for d in data:
        G = nx.Graph()
        G.add_nodes_from(range(d["n"]))
        G.add_edges_from(d["edges"])
        out.append((G, d["opt_cut"], d["sa_ratio_mean"]))
    return out


def cut_table(G):
    """cut(x) for every basis state index (used to turn probabilities into an expected cut)."""
    n = G.number_of_nodes()
    cuts = np.zeros(2 ** n)
    for k in range(2 ** n):
        x = bits_of(k, n)
        cuts[k] = sum(x[i] != x[j] for i, j in G.edges)
    return cuts


def cost_operator(G):
    """Cut operator C = sum_edges (1 - Z_i Z_j)/2; its eigenvalue on |x> is cut(x)."""
    n = G.number_of_nodes()
    terms = [("I" * n, 0.5 * G.number_of_edges())]
    for i, j in G.edges:
        z = ["I"] * n
        z[n - 1 - i] = "Z"  # Qiskit Pauli strings are written with qubit 0 on the right
        z[n - 1 - j] = "Z"
        terms.append(("".join(z), -0.5))
    return SparsePauliOp.from_list(terms).simplify()


def qaoa_circuit(G, gammas, betas, measure=False):
    """|+>^n, then p layers of  exp(-i gamma C)  (RZZ on each edge)  and  exp(-i beta sum X)  (RX on each qubit)."""
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i, j in G.edges:
            qc.rzz(-g, i, j)  # exp(+i g/2 Z_i Z_j) = exp(-i g (1 - Z_i Z_j)/2) up to a global phase
        qc.rx(2 * b, range(n))
    if measure:
        qc.measure_all()
    return qc


def expected_cut(G, params, cuts):
    p = len(params) // 2
    probs = Statevector(qaoa_circuit(G, params[:p], params[p:])).probabilities()
    return float(probs @ cuts), probs


def interp_init(prev):
    """INTERP warm start (Zhou et al. 2020): build depth-(p+1) angles by linearly interpolating the best depth-p
    angles. Gives smooth schedules and avoids the bad local optima that random starts hit at larger p."""
    p = len(prev) // 2

    def stretch(a):
        a = np.concatenate([[0.0], a, [0.0]])
        return np.array([(i / p) * a[i] + ((p - i) / p) * a[i + 1] for i in range(p + 1)])

    return np.concatenate([stretch(prev[:p]), stretch(prev[p:])])


def optimize_qaoa(G, p, cuts, restarts=5, seed=0, objective=None, warm=None):
    """Maximize the expected cut with COBYLA from several random starts (plus an optional warm start).
    Returns best params and the number of objective (circuit) evaluations."""
    rng = np.random.default_rng(seed)
    evals = 0
    best = None

    def neg(params):
        nonlocal evals
        evals += 1
        return -(objective(params) if objective else expected_cut(G, params, cuts)[0])

    starts = [np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)]) for _ in range(restarts)]
    if warm is not None:
        starts.insert(0, warm)
    for x0 in starts:
        r = minimize(neg, x0, method="COBYLA", options={"maxiter": 300 * p, "rhobeg": 0.5})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, evals


def qaoa_experiment(depths=(1, 2, 3)):
    rows, best_params = [], {}
    for G, opt_cut, sa_ratio in load_graphs():
        n = G.number_of_nodes()
        cuts = cut_table(G)
        for p in depths:
            if n >= 12 and p == 3:
                continue  # keep runtime small
            t0 = time.perf_counter()
            warm = interp_init(best_params[(n, p - 1)]) if p > 1 else None
            params, evals = optimize_qaoa(G, p, cuts, warm=warm)
            elapsed = time.perf_counter() - t0
            exp_cut, probs = expected_cut(G, params, cuts)
            best_params[(n, p)] = params
            rows.append({"n": n, "p": p, "opt_cut": opt_cut,
                         "approx_ratio_expected": exp_cut / opt_cut,
                         "prob_optimal_cut": float(probs[cuts == opt_cut].sum()),
                         "sa_ratio_trackB": sa_ratio, "random_guess_ratio": cuts.mean() / opt_cut,
                         "circuit_evals": evals, "cx_after_transpile": transpile(
                             qaoa_circuit(G, params[:p], params[p:]), basis_gates=BASIS).count_ops().get("cx", 0),
                         "time_s": elapsed})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "c1_qaoa")
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in df.groupby("p"):
        ax.plot(g["n"], g["approx_ratio_expected"], "o-", label=f"QAOA p={p} (expected cut / optimum)")
    g1 = df[df.p == 1]
    ax.plot(g1["n"], g1["random_guess_ratio"], "k--", label="random assignment")
    ax.plot(g1["n"], g1["sa_ratio_trackB"], "s:", label="simulated annealing (Track B)")
    ax.set(xlabel="nodes", ylabel="approximation ratio", title="QAOA on Max-Cut (statevector simulation)", ylim=(0.4, 1.05))
    ax.legend(fontsize=8)
    save_plot(fig, TRACK, "c1_qaoa")
    plt.close(fig)
    return df, best_params


# ---------------- Finite shots ----------------

def shots_experiment(best_params, n=8, seeds=5):
    """(a) Variance of the estimated expected cut at fixed parameters; (b) optimizing with a shot-based objective."""
    G, opt_cut, _ = [g for g in load_graphs() if g[0].number_of_nodes() == n][0]
    cuts = cut_table(G)
    params = best_params[(n, 1)]
    exact, probs = expected_cut(G, params, cuts)
    sigma = np.sqrt(probs @ cuts ** 2 - exact ** 2)  # std of the cut value in one measurement
    sim = AerSimulator()
    rows = []
    for shots in (100, 1000, 10000):
        # (a) estimator noise at the optimal parameters
        tqc = transpile(qaoa_circuit(G, params[:1], params[1:], measure=True), sim)
        ests = []
        for s in range(20):
            counts = sim.run(tqc, shots=shots, seed_simulator=s).result().get_counts()
            ests.append(sum(c * cuts[int(b, 2)] for b, c in counts.items()) / shots)
        # (b) optimize using the noisy estimate, then score the found parameters exactly
        finals = []
        for seed in range(seeds):
            counter = {"k": 0}

            def shot_obj(pr, shots=shots, seed=seed, counter=counter):
                counter["k"] += 1
                qc = transpile(qaoa_circuit(G, pr[:1], pr[1:], measure=True), sim)
                c = sim.run(qc, shots=shots, seed_simulator=1000 * seed + counter["k"]).result().get_counts()
                return sum(v * cuts[int(b, 2)] for b, v in c.items()) / shots

            found, _ = optimize_qaoa(G, 1, cuts, restarts=2, seed=seed, objective=shot_obj)
            finals.append(expected_cut(G, found, cuts)[0] / opt_cut)
        rows.append({"shots": shots, "estimate_std_at_opt_params": np.std(ests),
                     "estimate_bias": np.mean(ests) - exact,
                     "theory_std_sigma_over_sqrt_shots": sigma / np.sqrt(shots),
                     "optimized_ratio_mean": np.mean(finals), "optimized_ratio_std": np.std(finals)})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "c1_shots")
    return df


# ---------------- Noise ----------------

def noise_experiment(best_params, n=8, shots=20000):
    """Expected cut at the ideal-optimal parameters, ideal vs noisy simulation, for increasing depth p."""
    G, opt_cut, _ = [g for g in load_graphs() if g[0].number_of_nodes() == n][0]
    cuts = cut_table(G)
    rows = []
    for p in (1, 2, 3):
        params = best_params[(n, p)]
        qc = qaoa_circuit(G, params[:p], params[p:], measure=True)
        out = {}
        for label, noisy in [("ideal", False), ("noisy", True)]:
            counts = run_counts(qc, shots=shots, noisy=noisy, seed=1)
            out[label] = sum(c * cuts[int(b, 2)] for b, c in counts.items()) / shots / opt_cut
        rows.append({"p": p, "cx_gates": transpile(qc, basis_gates=BASIS).count_ops().get("cx", 0),
                     "ratio_ideal": out["ideal"], "ratio_noisy": out["noisy"],
                     "loss_from_noise": out["ideal"] - out["noisy"]})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "c1_noise")
    return df


# ---------------- VQE ----------------

# H2 at bond length 0.735 Angstrom, STO-3G, parity mapping with two-qubit reduction (standard textbook Hamiltonian,
# electronic energy in Hartree, nuclear repulsion not included).
H2 = SparsePauliOp.from_list([("II", -1.052373245772859), ("IZ", 0.39793742484318045),
                              ("ZI", -0.39793742484318045), ("ZZ", -0.01128010425623538),
                              ("XX", 0.18093119978423156)])


def vqe_ansatz(theta):
    """Hardware-efficient ansatz: RY layer, CNOT, RY layer, CNOT, RY layer (6 parameters)."""
    qc = QuantumCircuit(2)
    for layer in range(3):
        qc.ry(theta[2 * layer], 0)
        qc.ry(theta[2 * layer + 1], 1)
        if layer < 2:
            qc.cx(0, 1)
    return qc


def vqe_experiment(restarts=5):
    exact = float(np.linalg.eigvalsh(H2.to_matrix()).min())
    rng = np.random.default_rng(0)
    rows = []
    nm = noise_model()
    dm_sim = AerSimulator(method="density_matrix", noise_model=nm)

    def energy_ideal(t):
        return float(np.real(Statevector(vqe_ansatz(t)).expectation_value(H2)))

    def energy_noisy(t):
        qc = transpile(vqe_ansatz(t), basis_gates=BASIS, optimization_level=0)
        qc.save_density_matrix()
        rho = DensityMatrix(dm_sim.run(qc).result().data()["density_matrix"])
        return float(np.real(rho.expectation_value(H2)))

    for label, fn in [("ideal statevector", energy_ideal), ("noisy (density matrix)", energy_noisy)]:
        best, evals = None, 0
        t0 = time.perf_counter()
        for _ in range(restarts):
            r = minimize(fn, rng.uniform(-np.pi, np.pi, 6), method="COBYLA", options={"maxiter": 400})
            evals += r.nfev
            if best is None or r.fun < best.fun:
                best = r
        rows.append({"setting": label, "vqe_energy_Ha": best.fun, "exact_Ha": exact,
                     "error_mHa": 1000 * (best.fun - exact), "chemical_accuracy_1.6mHa": abs(best.fun - exact) < 1.6e-3,
                     "energy_evals": evals, "time_s": time.perf_counter() - t0})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "c1_vqe")
    return df


def main():
    pd.set_option("display.width", 200)
    qaoa, best = qaoa_experiment()
    print("== QAOA ==\n", qaoa.to_string(index=False))
    print("\n== Shots ==\n", shots_experiment(best).to_string(index=False))
    print("\n== Noise ==\n", noise_experiment(best).to_string(index=False))
    print("\n== VQE ==\n", vqe_experiment().to_string(index=False))


if __name__ == "__main__":
    main()
