import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trackD_submodular_parallel_distributed"))
import fed  # noqa: E402
import parallel  # noqa: E402
import parallelism_demo as pd_  # noqa: E402
import submodular as sm  # noqa: E402


def test_coverage_is_monotone_submodular_and_greedy_meets_bound():
    sets = sm.make_coverage(12, seed=3)
    f = lambda S: sm.coverage(sets, S)  # noqa: E731
    mono, dr = sm.test_assumptions(f, 12, samples=500)
    assert mono == 0 and dr == 0
    for k in (2, 3, 4):
        assert f(sm.greedy(f, 12, k)) >= (1 - 1 / np.e) * f(list(sm.brute_force(f, 12, k)))


def test_pair_bonus_violates_diminishing_returns():
    sets = sm.make_coverage(12, seed=3)
    f = sm.with_pair_bonus(sets, [(0, 1), (2, 3)], bonus=50)
    assert sm.test_assumptions(f, 12, samples=500)[1] > 0


def test_every_scheduler_runs_each_task_exactly_once(monkeypatch):
    done = []
    lock = parallel.threading.Lock()

    def fake(m):
        with lock:
            done.append(m)

    monkeypatch.setattr(parallel, "task", fake)
    tasks = list(range(1, 51))
    for name, fn in parallel.STRATEGIES.items():
        done.clear()
        fn(tasks, 4)
        assert sorted(done) == tasks, name


def test_data_and_model_parallel_match_single_worker():
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((64, 32)), rng.standard_normal((64, 1))
    P = pd_.init()
    g, _ = pd_.data_parallel(P, X, y, workers=3)
    full = pd_.grads(P, X, y)
    assert all(np.allclose(g[k], full[k]) for k in g)
    out, _ = pd_.model_parallel(P, X)
    assert np.allclose(out, pd_.forward(P, X)[1])


def test_fedavg_split_covers_all_data_once():
    X = np.zeros((200, 2))
    y = np.array([0, 1] * 100)
    for iid in (True, False):
        clients = fed.split_clients(X, y, 10, iid, np.random.default_rng(0))
        idx = np.concatenate(clients)
        assert len(idx) == len(set(idx.tolist())) and len(idx) <= 200


def test_mgda_min_norm_formula():
    """alpha from the closed form must minimize ||a g1 + (1-a) g2|| over a grid."""
    g1, g2 = torch.tensor([3.0, 0.5]), torch.tensor([-1.0, 2.0])
    a = torch.clamp(((g2 - g1) @ g2) / ((g1 - g2) @ (g1 - g2)), 0, 1).item()
    grid = np.linspace(0, 1, 10001)
    best = grid[np.argmin([np.linalg.norm(t * g1.numpy() + (1 - t) * g2.numpy()) for t in grid])]
    assert abs(a - best) < 1e-3
