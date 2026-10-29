"""Track I part 1: quantum algorithms on a simulator (Qiskit Aer).

Written gate-by-gate (no library algorithm implementations): Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT,
and QPE / Shor built on the hand-written QFT. Quantum annealing: exact simulation of the time evolution of a small
transverse-field Hamiltonian, compared with classical simulated annealing.
All results are SIMULATED; noisy runs use the depolarizing + readout noise model in common/quantum.py.
"""
import os
import sys
import time
from fractions import Fraction
from math import gcd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.quantum import bits_of, gate_counts, run_counts  # noqa: E402
from common.utils import save_plot, save_results  # noqa: E402

TRACK = "trackI_quantum_algos_qml"


def qubit_bits(key, n):
    """Qiskit count key -> string where character i is the value of clbit i (Qiskit keys are reversed)."""
    return key.replace(" ", "")[::-1][:n]


# ---------------- Deutsch-Jozsa ----------------

def dj_oracle(qc, kind, n):
    """Oracle U_f |x>|y> = |x>|y xor f(x)> on inputs 0..n-1 and ancilla n."""
    if kind == "constant_0":
        pass
    elif kind == "constant_1":
        qc.x(n)
    elif kind == "balanced_parity":  # f(x) = x0 xor x1 xor ... : flips y for half the inputs
        for i in range(n):
            qc.cx(i, n)
    elif kind == "balanced_x0":  # f(x) = x0
        qc.cx(0, n)


def deutsch_jozsa(kind, n=3):
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))  # inputs in uniform superposition, ancilla in |->  (phase kickback)
    dj_oracle(qc, kind, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc  # all-zeros output <=> f is constant (one oracle query; classically up to 2^(n-1)+1)


# ---------------- Bernstein-Vazirani ----------------

def bernstein_vazirani(secret):
    """Find s in f(x) = s.x mod 2 with ONE query (classically n queries). secret[i] = bit for qubit i."""
    n = len(secret)
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    for i, b in enumerate(secret):
        if b == "1":
            qc.cx(i, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


# ---------------- Grover ----------------

def mcz(qc, qubits):
    """Multi-controlled Z = H on the last qubit, multi-controlled X, H."""
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])


def grover(n, marked, iterations=None, measure=True):
    """Search N = 2^n items for one marked item with about (pi/4) sqrt(N) oracle queries."""
    N = 2 ** n
    k = iterations if iterations is not None else int(np.floor(np.pi / 4 * np.sqrt(N)))
    qs = list(range(n))
    qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    qc.h(qs)
    zeros = [i for i in qs if not (marked >> i) & 1]
    for _ in range(k):
        # Oracle: flip the phase of |marked>  (X maps |marked> to |11..1>, MCZ flips it, X undoes)
        if zeros:
            qc.x(zeros)
        mcz(qc, qs)
        if zeros:
            qc.x(zeros)
        # Diffuser: reflection about the uniform superposition, 2|s><s| - I
        qc.h(qs)
        qc.x(qs)
        mcz(qc, qs)
        qc.x(qs)
        qc.h(qs)
    if measure:
        qc.measure(qs, qs)
    return qc, k


# ---------------- QFT, QPE, Shor ----------------

def qft(n, inverse=False):
    """Textbook QFT: on qubit j (highest first) a Hadamard, then controlled phases pi/2^(j-q) from the lower qubits,
    then swaps to reverse the order. QFT|x> = (1/sqrt N) sum_y exp(2 pi i x y / N) |y>."""
    qc = QuantumCircuit(n, name="QFT")
    for j in reversed(range(n)):
        qc.h(j)
        for q in range(j):
            qc.cp(np.pi / 2 ** (j - q), q, j)
    for q in range(n // 2):
        qc.swap(q, n - 1 - q)
    return qc.inverse() if inverse else qc


def qpe(phase_gate_angle, t):
    """Estimate phi in U|1> = exp(2 pi i phi)|1> for U = P(angle), using t counting qubits."""
    qc = QuantumCircuit(t + 1, t)
    qc.x(t)  # eigenstate |1> of the phase gate
    qc.h(range(t))
    for j in range(t):
        qc.cp(phase_gate_angle * 2 ** j, j, t)  # controlled-U^(2^j)
    qc.append(qft(t, inverse=True), range(t))
    qc.measure(range(t), range(t))
    return qc


def c_amod15(a, power):
    """Controlled multiplication by a^power mod 15 on 4 work qubits (textbook permutation circuit)."""
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in (2, 13):
            U.swap(2, 3); U.swap(1, 2); U.swap(0, 1)
        if a in (7, 8):
            U.swap(0, 1); U.swap(1, 2); U.swap(2, 3)
        if a in (4, 11):
            U.swap(1, 3); U.swap(0, 2)
        if a in (7, 11, 13):
            for q in range(4):
                U.x(q)
    gate = U.to_gate(label=f"{a}^{power} mod 15")
    return gate.control()


def shor15(a=7, t=8):
    """Order finding for a mod 15: counting register (t qubits) + work register starting in |1>."""
    qc = QuantumCircuit(t + 4, t)
    qc.h(range(t))
    qc.x(t)  # work register = |0001> = 1
    for j in range(t):
        qc.append(c_amod15(a, 2 ** j), [j] + [t + i for i in range(4)])
    qc.append(qft(t, inverse=True), range(t))
    qc.measure(range(t), range(t))
    return qc


def shor_postprocess(counts, a=7, N=15, t=8):
    """Measured y -> phase y/2^t -> continued fractions -> candidate order r -> factors gcd(a^(r/2) +- 1, N)."""
    rows, good = [], 0
    for key, c in counts.items():
        y = int(key, 2)
        phase = y / 2 ** t
        r = Fraction(phase).limit_denominator(N).denominator
        ok = r % 2 == 0 and pow(a, r, N) == 1 and pow(a, r // 2, N) != N - 1
        factors = sorted({gcd(pow(a, r // 2) - 1, N), gcd(pow(a, r // 2) + 1, N)}) if ok else []
        good += c if ok else 0
        rows.append({"measured_y": y, "phase": phase, "count": c, "order_guess_r": r, "factors": factors})
    return pd.DataFrame(rows).sort_values("measured_y"), good / sum(counts.values())


# ---------------- Quantum annealing (exact simulation) ----------------

def annealing_experiment(n=6, times=(0.5, 1, 2, 5, 10, 20, 50), steps=400):
    """H(s) = -(1 - s) sum_i X_i  +  s * H_P,   H_P = -cut(x) (diagonal),   s = t / T.
    Start in |+>^n (ground state of the transverse field), evolve with exp(-i H dt), measure P(optimal cut).
    Adiabatic theorem: success -> 1 when T >> 1 / (min gap)^2."""
    G = nx.gnp_random_graph(n, 0.5, seed=n)  # same graph as Track B's 6-node instance
    N = 2 ** n
    cuts = np.array([sum(bits_of(k, n)[i] != bits_of(k, n)[j] for i, j in G.edges) for k in range(N)])
    HP = np.diag(-cuts.astype(float))
    X = np.array([[0, 1], [1, 0]])
    HX = sum(np.kron(np.kron(np.eye(2 ** (n - 1 - i)), X), np.eye(2 ** i)) for i in range(n))
    optimal = cuts == cuts.max()
    # minimum spectral gap along the path, within the subspace symmetric under flipping all spins (the dynamics
    # never leaves it; Max-Cut's two-fold degenerate optimum x / not-x lives there as a single state)
    flip = np.array([(N - 1) ^ k for k in range(N)])
    sym = np.zeros((N, N))
    for k in range(N):
        sym[k, k] += 0.5
        sym[flip[k], k] += 0.5
    w, v = np.linalg.eigh(sym)
    P = v[:, w > 0.5]  # basis of the symmetric subspace
    gaps = []
    for s in np.linspace(0, 1, 201):
        e = np.linalg.eigvalsh(P.T @ (-(1 - s) * HX + s * HP) @ P)
        gaps.append(e[1] - e[0])
    rows = []
    for T in times:
        psi = np.ones(N, dtype=complex) / np.sqrt(N)
        dt = T / steps
        for k in range(steps):
            s = (k + 0.5) / steps
            psi = expm(-1j * dt * (-(1 - s) * HX + s * HP)) @ psi
        rows.append({"anneal_time_T": T, "P_optimal": float(np.sum(np.abs(psi[optimal]) ** 2))})
    df = pd.DataFrame(rows)
    df["min_gap"] = min(gaps)
    df["random_guess_P"] = optimal.mean()
    # classical simulated annealing (dwave-neal) on the same instance: fraction of single reads that are optimal
    import neal
    Q = {}
    for i, j in G.edges:
        Q[(i, i)] = Q.get((i, i), 0) - 1
        Q[(j, j)] = Q.get((j, j), 0) - 1
        Q[(min(i, j), max(i, j))] = Q.get((min(i, j), max(i, j)), 0) + 2
    res = neal.SimulatedAnnealingSampler().sample_qubo(Q, num_reads=1000, num_sweeps=10, seed=0)
    df["classical_SA_P_optimal_10_sweeps"] = float(np.mean(-res.record.energy == cuts.max()))
    save_results(df, TRACK, "i1_annealing")
    return df


# ---------------- Experiments ----------------

def correctness_and_noise(shots=4000):
    rows = []
    n = 3
    for kind in ["constant_0", "constant_1", "balanced_parity", "balanced_x0"]:
        for noisy in (False, True):
            counts = run_counts(deutsch_jozsa(kind, n), shots, noisy)
            p0 = counts.get("0" * n, 0) / shots
            expected = 1.0 if kind.startswith("constant") else 0.0
            rows.append({"algorithm": "Deutsch-Jozsa", "case": kind, "noisy": noisy,
                         "success_prob": p0 if expected == 1 else 1 - p0})
    for secret in ["1011", "110101"]:
        for noisy in (False, True):
            counts = run_counts(bernstein_vazirani(secret), shots, noisy)
            hits = sum(c for k, c in counts.items() if qubit_bits(k, len(secret)) == secret)
            rows.append({"algorithm": "Bernstein-Vazirani", "case": f"s={secret}", "noisy": noisy,
                         "success_prob": hits / shots})
    for nq in (3, 4, 5):
        marked = 5 % (2 ** nq)
        for noisy in (False, True):
            qc, k = grover(nq, marked)
            counts = run_counts(qc, shots, noisy)
            rows.append({"algorithm": "Grover", "case": f"n={nq}, k={k} iterations", "noisy": noisy,
                         "success_prob": counts.get(format(marked, f"0{nq}b"), 0) / shots})
    for angle, label, t in [(np.pi / 4, "T gate, phi = 1/8", 3), (2 * np.pi / 3, "P(2pi/3), phi = 1/3", 3),
                            (2 * np.pi / 3, "P(2pi/3), phi = 1/3", 6)]:
        counts = run_counts(qpe(angle, t), shots, False)
        best = max(counts, key=counts.get)
        rows.append({"algorithm": "QPE", "case": f"{label}, t={t}", "noisy": False,
                     "success_prob": counts[best] / shots, "estimate": int(best, 2) / 2 ** t})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "i1_algorithms")
    return df


def qft_check(n=4):
    """Hand-written QFT == DFT matrix == Qiskit's QFTGate."""
    N = 2 ** n
    F = np.array([[np.exp(2j * np.pi * x * y / N) for x in range(N)] for y in range(N)]) / np.sqrt(N)
    mine = Operator(qft(n)).data
    lib = Operator(QFTGate(n)).data
    return float(np.abs(mine - F).max()), float(np.abs(mine - lib).max())


def shor_experiment(shots=2000):
    qc = shor15()
    counts = run_counts(qc, shots, False)
    table, success = shor_postprocess(counts)
    save_results(table, TRACK, "i1_shor")
    return table, success, gate_counts(qc.remove_final_measurements(inplace=False))


def grover_vs_linear(qubits=range(2, 11), repeats=200):
    rng = np.random.default_rng(0)
    rows = []
    for n in qubits:
        N = 2 ** n
        marked = int(rng.integers(N))
        qc, k = grover(n, marked, measure=False)
        t0 = time.perf_counter()
        p = float(np.abs(Statevector(qc).data[marked]) ** 2)
        t_q = time.perf_counter() - t0
        gc = gate_counts(qc)
        # classical: unsorted list, check items one by one (each check = one oracle query)
        items = list(range(N))
        t0 = time.perf_counter()
        queries = []
        for _ in range(repeats):
            target = int(rng.integers(N))
            order = rng.permutation(N)
            for q, idx in enumerate(order, 1):
                if items[idx] == target:
                    queries.append(q)
                    break
        t_c = (time.perf_counter() - t0) / repeats
        rows.append({"qubits": n, "N": N, "grover_queries": k, "grover_success_prob": p,
                     "classical_queries_mean": np.mean(queries), "classical_queries_theory": (N + 1) / 2,
                     "cx_gates": gc["cx"], "depth": gc["depth"],
                     "grover_sim_time_ms": 1000 * t_q, "classical_time_ms": 1000 * t_c})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "i1_grover_vs_linear")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.loglog(df.N, df.grover_queries, "o-", label="Grover oracle queries ≈ (π/4)√N")
    a1.loglog(df.N, df.classical_queries_mean, "s-", label="classical queries ≈ N/2 (measured)")
    a1.set(xlabel="N", ylabel="oracle queries", title="Query complexity")
    a2.semilogy(df.qubits, df.grover_sim_time_ms, "o-", label="Grover (statevector simulation)")
    a2.semilogy(df.qubits, df.classical_time_ms, "s-", label="classical linear search (Python)")
    a2.semilogy(df.qubits, df.cx_gates, "^--", label="CX gate count")
    a2.set(xlabel="qubits n (N = 2^n)", ylabel="ms  /  gates", title="Wall-clock time and gate count")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    save_plot(fig, TRACK, "i1_grover_vs_linear")
    plt.close(fig)
    return df


def main():
    pd.set_option("display.width", 200)
    print("== Correctness + noise ==\n", correctness_and_noise().to_string(index=False))
    e_dft, e_lib = qft_check()
    print(f"\nQFT: max |mine - DFT| = {e_dft:.2e}, max |mine - QFTGate| = {e_lib:.2e}")
    table, success, gc = shor_experiment()
    print("\n== Shor (N=15, a=7) ==\n", table.to_string(index=False), f"\nshots giving correct factors: {success:.3f}",
          f"\ncircuit size (transpiled): {gc}")
    print("\n== Grover vs linear ==\n", grover_vs_linear().to_string(index=False))
    print("\n== Annealing ==\n", annealing_experiment().to_string(index=False))


if __name__ == "__main__":
    main()
