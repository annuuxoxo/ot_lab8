import os
import sys

import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trackI_quantum_algos_qml"))
import algorithms as alg  # noqa: E402
import qml_vs_classical as qvc  # noqa: E402
from common.quantum import run_counts  # noqa: E402


def test_deutsch_jozsa_ideal():
    for kind in ["constant_0", "constant_1"]:
        assert run_counts(alg.deutsch_jozsa(kind), 500) == {"000": 500}
    for kind in ["balanced_parity", "balanced_x0"]:
        assert "000" not in run_counts(alg.deutsch_jozsa(kind), 500)


def test_bernstein_vazirani_recovers_secret():
    for s in ["1", "1011", "0110101"]:
        counts = run_counts(alg.bernstein_vazirani(s), 200)
        assert len(counts) == 1 and alg.qubit_bits(next(iter(counts)), len(s)) == s


def test_grover_success_matches_theory():
    """With k iterations the success probability is sin^2((2k+1) theta), sin(theta) = 1/sqrt(N)."""
    for n in (2, 3, 4, 5):
        qc, k = alg.grover(n, marked=3, measure=False)
        p = abs(Statevector(qc).data[3]) ** 2
        theta = np.arcsin(1 / np.sqrt(2 ** n))
        assert np.isclose(p, np.sin((2 * k + 1) * theta) ** 2)


def test_qft_equals_dft_matrix():
    e_dft, e_lib = alg.qft_check(3)
    assert e_dft < 1e-12 and e_lib < 1e-12


def test_qpe_t_gate_exact():
    counts = run_counts(alg.qpe(np.pi / 4, 3), 300)
    assert counts == {"001": 300}  # 1/8 = 0.001 in binary


def test_shor_postprocessing_finds_3_and_5():
    table, success = alg.shor_postprocess({"01000000": 1, "11000000": 1})  # y = 64, 192 -> phases 1/4, 3/4
    assert all(f == [3, 5] for f in table["factors"]) and success == 1.0


def test_quantum_kernel_is_valid():
    X = np.random.default_rng(0).uniform(0, np.pi, (6, 2))
    K = qvc.quantum_kernel(X, X)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T) and np.linalg.eigvalsh(K).min() > -1e-10
