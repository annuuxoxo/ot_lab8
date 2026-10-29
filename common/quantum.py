"""Shared quantum helpers for Tracks C and I (Qiskit + Aer, simulation only).

The noise model is a simple, explainable stand-in for real hardware:
depolarizing error on every gate + bit-flip readout error. Default rates are the same order of
magnitude as current superconducting devices (1-qubit ~1e-3, 2-qubit ~1e-2, readout ~2%).
"""
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

BASIS = ["rz", "sx", "x", "cx"]


def noise_model(p1=1e-3, p2=1e-2, readout=0.02):
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def run_counts(qc, shots=4000, noisy=False, seed=0, nm=None):
    """Transpile to the hardware-like basis and sample. Returns {bitstring: count} (Qiskit order: qubit 0 = rightmost)."""
    sim = AerSimulator(noise_model=(nm or noise_model()) if noisy else None, seed_simulator=seed)
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
    return sim.run(tqc, shots=shots).result().get_counts()


def gate_counts(qc):
    """2-qubit gate count and depth after transpiling to the hardware-like basis."""
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
    ops = tqc.count_ops()
    return {"cx": ops.get("cx", 0), "total_gates": sum(ops.values()) - ops.get("measure", 0) - ops.get("barrier", 0),
            "depth": tqc.depth()}


def bits_of(index, n):
    """Qiskit statevector index -> array x with x[i] = value of qubit i (little-endian)."""
    return np.array([(index >> i) & 1 for i in range(n)])
