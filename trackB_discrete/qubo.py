"""QUBO and Ising formulations.

QUBO: minimize E(x) = x^T Q x over x in {0,1}^n   (Q upper-triangular; diagonal = linear terms)
Ising: minimize E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset over s in {-1,+1}^n
Conversion uses x = (1 - s) / 2  (so x=0 <-> s=+1, x=1 <-> s=-1).
"""
import itertools

import networkx as nx
import numpy as np


def qubo_energy(Q, x):
    return float(x @ Q @ x)


def ising_energy(h, J, offset, s):
    return float(h @ s + s @ np.triu(J, 1) @ s + offset)


def qubo_to_ising(Q):
    """Substitute x_i = (1 - s_i)/2 into x^T Q x and collect terms.
    Linear term Q_ii x_i            -> Q_ii/2 - Q_ii/2 s_i
    Quadratic term Q_ij x_i x_j     -> Q_ij/4 (1 - s_i - s_j + s_i s_j)"""
    Q = np.triu(Q)
    n = len(Q)
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    for i in range(n):
        offset += Q[i, i] / 2
        h[i] -= Q[i, i] / 2
        for j in range(i + 1, n):
            q = Q[i, j]
            if q:
                offset += q / 4
                h[i] -= q / 4
                h[j] -= q / 4
                J[i, j] += q / 4
    return h, J, offset


def brute_force_qubo(Q):
    n = len(Q)
    best_e, best_x = np.inf, None
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits)
        e = qubo_energy(Q, x)
        if e < best_e:
            best_e, best_x = e, x
    return best_e, best_x


# ---------- Max-Cut ----------

def make_graph(n, seed, p=0.5):
    return nx.gnp_random_graph(n, p, seed=seed)


def cut_value(G, x):
    return sum(1 for i, j in G.edges if x[i] != x[j])


def maxcut_qubo(G):
    """Edge (i,j) is cut iff x_i != x_j, i.e. x_i + x_j - 2 x_i x_j = 1.
    Maximizing the cut = minimizing -sum_edges (x_i + x_j - 2 x_i x_j), so
    Q_ii = -deg(i), Q_ij = +2 for each edge (i<j). Then x^T Q x = -cut(x)."""
    n = G.number_of_nodes()
    Q = np.zeros((n, n))
    for i, j in G.edges:
        i, j = min(i, j), max(i, j)
        Q[i, i] -= 1
        Q[j, j] -= 1
        Q[i, j] += 2
    return Q


# ---------- Constrained feature selection (pick exactly k) ----------

def feature_selection_qubo(relevance, redundancy, k, P):
    """minimize  -sum_i rel_i x_i + sum_{i<j} red_ij x_i x_j + P (sum_i x_i - k)^2
    The penalty (sum x - k)^2 = sum x_i (1 - 2k) + 2 sum_{i<j} x_i x_j + k^2  (using x_i^2 = x_i),
    so it adds P(1-2k) to each diagonal entry, 2P to each off-diagonal one, and a constant P k^2.
    Returns (Q, constant)."""
    n = len(relevance)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -relevance[i] + P * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] = redundancy[i, j] + 2 * P
    return Q, P * k * k


def qubo_dict(Q):
    """Convert matrix to the {(i, j): coeff} dict that dimod/neal samplers expect."""
    n = len(Q)
    return {(i, j): float(Q[i, j]) for i in range(n) for j in range(i, n) if Q[i, j] != 0}


def simulated_annealing(Q, num_reads=100, seed=0):
    """Solve a QUBO with D-Wave's classical simulated annealer (dwave-neal). Returns (energy, x)."""
    import neal
    sampler = neal.SimulatedAnnealingSampler()
    res = sampler.sample_qubo(qubo_dict(Q), num_reads=num_reads, seed=seed)
    best = res.first
    x = np.array([best.sample.get(i, 0) for i in range(len(Q))])
    return float(best.energy), x
