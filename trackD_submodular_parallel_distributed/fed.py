"""Track D part 4: Federated Averaging (FedAvg) vs centralized training (simulated clients, one machine).

Logistic regression on make_classification, 10 clients. Each round: every client starts from the global model, runs E
local epochs of mini-batch SGD on its own data, and the server averages the client models weighted by data size.
Communication = one model upload + download per client per round.
IID split: random. Non-IID split: label skew (each client holds 95% one class).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import save_plot, save_results  # noqa: E402

TRACK = "trackD_submodular_parallel_distributed"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sgd_epochs(w, X, y, epochs, lr, rng, batch=32):
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            idx = perm[i:i + batch]
            Xb = np.c_[X[idx], np.ones(len(idx))]
            w = w - lr * Xb.T @ (sigmoid(Xb @ w) - y[idx]) / len(idx)
    return w


def accuracy(w, X, y):
    return np.mean((sigmoid(np.c_[X, np.ones(len(y))] @ w) > 0.5) == y)


def split_clients(X, y, k, iid, rng):
    if iid:
        return np.array_split(rng.permutation(len(y)), k)
    # label skew: client i mostly gets class (i % 2); only 5% of its data from the other class
    by_class = {c: list(rng.permutation(np.flatnonzero(y == c))) for c in (0, 1)}
    per = len(y) // k
    clients = []
    for i in range(k):
        main, other = i % 2, 1 - i % 2
        n_main = int(0.95 * per)
        idx = [by_class[main].pop() for _ in range(min(n_main, len(by_class[main])))]
        idx += [by_class[other].pop() for _ in range(min(per - len(idx), len(by_class[other])))]
        clients.append(np.array(idx))
    return clients


def fedavg(clients, X, y, Xte, yte, rounds, local_epochs, lr, seed):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1] + 1)
    sizes = np.array([len(c) for c in clients])
    curve = []
    for _ in range(rounds):
        local = [sgd_epochs(w.copy(), X[c], y[c], local_epochs, lr, rng) for c in clients]
        w = np.sum([wi * n for wi, n in zip(local, sizes)], axis=0) / sizes.sum()
        curve.append(accuracy(w, Xte, yte))
    return np.array(curve)


def main(rounds=40, seeds=3, k=10, lr=0.002):
    # well-separated classes (a linear model can reach ~95%) and a small lr, so convergence takes many rounds
    # (a first version saturated at its ~70% ceiling after one round and showed nothing - failure log #12)
    X, y = make_classification(6000, 20, n_informative=10, class_sep=2.0, flip_y=0.01, random_state=0)
    X, Xte, y, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rows, curves = [], {}
    # centralized reference: same total number of epochs over all the data
    cent = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        w, c = np.zeros(X.shape[1] + 1), []
        for _ in range(rounds):
            w = sgd_epochs(w, X, y, 1, lr, rng)
            c.append(accuracy(w, Xte, yte))
        cent.append(c)
    curves["centralized (1 epoch per 'round')"] = np.array(cent)
    for iid in (True, False):
        for E in (1, 5, 20):
            runs = []
            for s in range(seeds):
                clients = split_clients(X, y, k, iid, np.random.default_rng(100 + s))
                runs.append(fedavg(clients, X, y, Xte, yte, rounds, E, lr, s))
            curves[f"FedAvg {'IID' if iid else 'non-IID'} E={E}"] = np.array(runs)
    for name, c in curves.items():
        target = 0.98 * curves["centralized (1 epoch per 'round')"].mean(0)[-1]
        reach = np.argmax(c.mean(0) >= target) + 1 if (c.mean(0) >= target).any() else None
        rows.append({"setting": name, "acc_after_1_round": c[:, 0].mean(), "acc_after_5_rounds": c[:, 4].mean(),
                     "final_acc_mean": c[:, -1].mean(), "final_acc_std": c[:, -1].std(),
                     "rounds_to_98pct_of_centralized": reach,
                     "client_uploads_total": None if name.startswith("central") else rounds * k})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "d2_fedavg")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, c in curves.items():
        ax.plot(range(1, rounds + 1), c.mean(0), "-" if "non" not in name else "--", label=name)
    ax.set(xlabel="communication round", ylabel="test accuracy", title=f"FedAvg, {k} clients, {seeds} seeds (mean)")
    ax.legend(fontsize=7)
    save_plot(fig, TRACK, "d2_fedavg")
    plt.close(fig)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(main().to_string(index=False))
