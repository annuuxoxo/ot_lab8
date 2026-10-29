import os
import sys

import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trackC_quantum_opt"))
import pqc  # noqa: E402
import qaoa_vqe as qv  # noqa: E402


def small_graph():
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    return G


def test_cost_operator_eigenvalues_are_cut_values():
    """The cut operator is diagonal and its diagonal equals cut(x) for every bitstring = classical QUBO objective."""
    G = small_graph()
    diag = np.real(np.diag(qv.cost_operator(G).to_matrix()))
    assert np.allclose(diag, qv.cut_table(G))


def test_qaoa_expected_cut_matches_operator_expectation():
    G = small_graph()
    params = np.array([0.7, 0.3, 0.4, 0.2])
    exp_cut, _ = qv.expected_cut(G, params, qv.cut_table(G))
    sv = Statevector(qv.qaoa_circuit(G, params[:2], params[2:]))
    assert np.isclose(exp_cut, np.real(sv.expectation_value(qv.cost_operator(G))))


def test_qaoa_at_zero_angles_is_random_guessing():
    G = small_graph()
    cuts = qv.cut_table(G)
    exp_cut, _ = qv.expected_cut(G, np.zeros(2), cuts)
    assert np.isclose(exp_cut, cuts.mean())


def test_interp_warm_start_shape_and_endpoints():
    """INTERP turns p=1 angles (gamma, beta) into p=2 angles; for p=1 both new angles equal the old one."""
    w = qv.interp_init(np.array([0.5, 0.3]))
    assert np.allclose(w, [0.5, 0.5, 0.3, 0.3])


def test_vqe_reaches_exact_ground_energy():
    exact = np.linalg.eigvalsh(qv.H2.to_matrix()).min()
    from scipy.optimize import minimize
    best = min((minimize(lambda t: np.real(Statevector(qv.vqe_ansatz(t)).expectation_value(qv.H2)),
                         np.random.default_rng(s).uniform(-np.pi, np.pi, 6), method="COBYLA",
                         options={"maxiter": 400}).fun for s in range(3)))
    assert abs(best - exact) < 1.6e-3


def test_parameter_shift_matches_autodiff():
    from pennylane import numpy as pnp
    import pennylane as qml
    X, y, _, _ = pqc.load_data()
    th = np.random.default_rng(1).uniform(-np.pi, np.pi, (pqc.LAYERS, 2, 2))
    auto = qml.grad(lambda t: pnp.mean((pqc.classifier(X[:, 0], X[:, 1], t) - y) ** 2))(pnp.array(th, requires_grad=True))
    assert np.allclose(pqc.parameter_shift_grad(th, X, y), auto, atol=1e-10)
