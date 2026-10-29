"""Track B experiments: knapsack (greedy / DP / B&B / ILP), feature selection (greedy vs exhaustive),
and QUBO formulations (Max-Cut, constrained feature selection) solved by brute force and simulated annealing."""
import itertools
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import RESULTS, save_plot, save_results, set_seed  # noqa: E402
import knapsack as ks  # noqa: E402
import qubo as qb  # noqa: E402

TRACK = "trackB_discrete"


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


# ---------------- Knapsack ----------------

def knapsack_experiment(sizes=(10, 50, 100, 200), per_size=3):
    rows = []
    for n in sizes:
        for s in range(per_size):
            v, w, C = ks.make_instance(n, seed=1000 * n + s)
            (opt, x_dp), t_dp = timed(ks.dp, v, w, C)
            (g, x_g), t_g = timed(ks.greedy, v, w, C)
            (b, x_b, nodes), t_b = timed(ks.branch_and_bound, v, w, C)
            (i_val, x_i), t_i = timed(ks.ilp, v, w, C)
            # independent feasibility + objective check for every returned solution
            for val, x in [(opt, x_dp), (g, x_g), (b, x_b), (i_val, x_i)]:
                assert ks.check(v, w, C, x) == val
            for name, val, t, extra in [("Greedy", g, t_g, None), ("DP", opt, t_dp, None),
                                        ("Branch-and-Bound", b, t_b, nodes), ("ILP (CBC)", i_val, t_i, None)]:
                rows.append({"n": n, "instance": s, "capacity": C, "method": name, "value": val,
                             "gap_pct": 100 * (opt - val) / opt, "time_ms": 1000 * t, "bnb_nodes": extra})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "b1_knapsack_raw")
    summary = (df.groupby(["n", "method"], sort=False)
                 .agg(gap_pct_mean=("gap_pct", "mean"), gap_pct_max=("gap_pct", "max"),
                      time_ms_mean=("time_ms", "mean"), time_ms_std=("time_ms", "std"),
                      bnb_nodes_mean=("bnb_nodes", "mean"))
                 .reset_index())
    save_results(summary, TRACK, "b1_knapsack_summary")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in summary.groupby("method", sort=False):
        ax1.loglog(g["n"], g["time_ms_mean"], "o-", label=m)
        ax2.plot(g["n"], g["gap_pct_mean"], "o-", label=m)
    ax1.set(xlabel="n items", ylabel="time (ms, mean of 3)", title="Knapsack runtime")
    ax2.set(xlabel="n items", ylabel="optimality gap (%)", title="Knapsack solution quality")
    ax1.legend()
    ax2.legend()
    save_plot(fig, TRACK, "b1_knapsack")
    plt.close(fig)
    return summary


# ---------------- Feature selection (combinatorial) ----------------

def feature_selection_experiment(ks_=(2, 3, 4)):
    X, y = load_wine(return_X_y=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    n_feat = X.shape[1]

    def cv_acc(subset):
        """Objective: mean 5-fold CV accuracy of the model trained on these features."""
        return cross_val_score(model, X[:, sorted(subset)], y, cv=5).mean()

    rows = []
    for k in ks_:
        # Exhaustive: score every subset of size k -> exact optimum, C(n, k) evaluations
        t0 = time.perf_counter()
        scored = {s: cv_acc(s) for s in itertools.combinations(range(n_feat), k)}
        best = max(scored, key=scored.get)
        rows.append({"k": k, "method": "Exhaustive", "features": list(best), "cv_acc": scored[best],
                     "subsets_evaluated": len(scored), "time_s": time.perf_counter() - t0})

        # Greedy forward: k rounds, each adds the one feature that raises the score most
        t0 = time.perf_counter()
        chosen, evals = [], 0
        for _ in range(k):
            cand = {f: cv_acc(chosen + [f]) for f in range(n_feat) if f not in chosen}
            evals += len(cand)
            chosen.append(max(cand, key=cand.get))
        rows.append({"k": k, "method": "Greedy forward", "features": sorted(chosen), "cv_acc": cv_acc(chosen),
                     "subsets_evaluated": evals, "time_s": time.perf_counter() - t0})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "b1_feature_selection")
    return df


# ---------------- QUBO: Max-Cut ----------------

def maxcut_experiment(sizes=(6, 8, 10, 12), sa_seeds=10):
    rows, saved = [], []
    for n in sizes:
        G = qb.make_graph(n, seed=n)
        Q = qb.maxcut_qubo(G)
        h, J, off = qb.qubo_to_ising(Q)
        if n == 6:  # verify QUBO == Ising == -cut on every bitstring
            for bits in itertools.product([0, 1], repeat=n):
                x = np.array(bits)
                s = 1 - 2 * x
                assert np.isclose(qb.qubo_energy(Q, x), -qb.cut_value(G, x))
                assert np.isclose(qb.ising_energy(h, J, off, s), qb.qubo_energy(Q, x))
        (e_bf, x_bf), t_bf = timed(qb.brute_force_qubo, Q)
        opt_cut = -int(round(e_bf))
        sa_cuts, sa_times = [], []
        for s in range(sa_seeds):
            (e_sa, x_sa), t_sa = timed(qb.simulated_annealing, Q, 50, s)
            sa_cuts.append(qb.cut_value(G, x_sa))
            sa_times.append(t_sa)
        ratios = np.array(sa_cuts) / opt_cut
        rows.append({"n": n, "edges": G.number_of_edges(), "opt_cut": opt_cut, "brute_force_ms": 1000 * t_bf,
                     "sa_ratio_mean": ratios.mean(), "sa_ratio_std": ratios.std(),
                     "sa_ms_mean": 1000 * np.mean(sa_times)})
        saved.append({"n": n, "seed": n, "edges": [list(e) for e in G.edges], "opt_cut": opt_cut,
                      "opt_x": x_bf.tolist(), "sa_ratio_mean": ratios.mean()})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "b2_maxcut")
    # Track C uses these graphs and optima as its classical baseline
    with open(os.path.join(RESULTS, TRACK, "maxcut_graphs.json"), "w") as f:
        json.dump(saved, f, indent=1)
    return df


# ---------------- QUBO: constrained feature selection with penalty ----------------

def penalty_experiment(k=3, penalties=(0.0, 0.005, 0.01, 0.02, 0.05, 0.2, 1.0, 5.0, 20.0), sa_seeds=20):
    X, y = load_wine(return_X_y=True)
    Xs = (X - X.mean(0)) / X.std(0)
    relevance = np.abs([np.corrcoef(Xs[:, i], y)[0, 1] for i in range(X.shape[1])])
    redundancy = np.abs(np.corrcoef(Xs.T))
    rows = []
    for P in penalties:
        Q, const = qb.feature_selection_qubo(relevance, redundancy, k, P)
        e_bf, x_bf = qb.brute_force_qubo(Q)
        # SA with a small budget (5 reads) over many seeds: how often does it find the true optimum?
        hits = [np.isclose(qb.simulated_annealing(Q, 5, seed=s)[0], e_bf) for s in range(sa_seeds)]
        rows.append({"penalty_P": P, "bf_selected": int(x_bf.sum()), "bf_feasible": int(x_bf.sum()) == k,
                     "bf_features": np.flatnonzero(x_bf).tolist(), "bf_objective_wo_penalty": e_bf + const,
                     "sa_success_rate": float(np.mean(hits))})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "b2_feature_qubo_penalty")
    return df


def main():
    set_seed(0)
    pd.set_option("display.width", 200)
    print("== Knapsack ==\n", knapsack_experiment().to_string(index=False))
    print("\n== Feature selection ==\n", feature_selection_experiment().to_string(index=False))
    print("\n== Max-Cut QUBO ==\n", maxcut_experiment().to_string(index=False))
    print("\n== Feature-selection QUBO penalty ==\n", penalty_experiment().to_string(index=False))


if __name__ == "__main__":
    main()
