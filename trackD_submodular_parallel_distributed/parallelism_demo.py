"""Track D part 3: data, model and pipeline parallelism on a tiny 2-layer NumPy MLP.

This is a SIMULATION in one process (the 'workers' are just separate pieces of code): it demonstrates what each scheme
splits, verifies it computes exactly the same result as the single-worker version, and counts what must be communicated.
Real speedups for these schemes need multiple GPUs/nodes; the measured CPU speedups are in parallel.py.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import save_results  # noqa: E402

TRACK = "trackD_submodular_parallel_distributed"


def init(d_in=32, h=64, d_out=1, seed=0):
    rng = np.random.default_rng(seed)
    return {"W1": rng.standard_normal((d_in, h)) * 0.1, "W2": rng.standard_normal((h, d_out)) * 0.1}


def forward(P, X):
    H = np.maximum(X @ P["W1"], 0)
    return H, H @ P["W2"]


def grads(P, X, y):
    """Mean squared error gradient for one (sub-)batch."""
    H, out = forward(P, X)
    d_out = 2 * (out - y) / len(X)
    dW2 = H.T @ d_out
    dH = d_out @ P["W2"].T * (H > 0)
    return {"W1": X.T @ dH, "W2": dW2}


def data_parallel(P, X, y, workers=4):
    """Each worker holds a full model copy and a shard of the batch; gradients are averaged (all-reduce),
    weighted by shard size so the result equals the full-batch gradient."""
    shards = np.array_split(np.arange(len(X)), workers)
    g = {k: np.zeros_like(v) for k, v in P.items()}
    for idx in shards:
        gi = grads(P, X[idx], y[idx])
        for k in g:
            g[k] += gi[k] * len(idx) / len(X)
    comm = workers * sum(v.size for v in P.values())  # every worker sends its full gradient each step
    return g, comm


def model_parallel(P, X, workers=2):
    """Worker A holds W1, worker B holds W2. A computes H and sends the activations to B."""
    H = np.maximum(X @ P["W1"], 0)  # on worker A
    out = H @ P["W2"]  # on worker B
    comm = H.size  # activations sent A -> B (and the same size back in the backward pass)
    return out, comm


def pipeline(P, X, micro=8, stages=2):
    """Model parallel + micro-batches: while stage 2 works on micro-batch i, stage 1 starts micro-batch i+1.
    Ideal schedule (GPipe): the 'bubble' (idle) fraction is (stages-1)/(micro+stages-1)."""
    outs = [np.maximum(mb @ P["W1"], 0) @ P["W2"] for mb in np.array_split(X, micro)]
    bubble = (stages - 1) / (micro + stages - 1)
    time_units = micro + stages - 1  # vs micro*stages if the stages ran one after another
    return np.vstack(outs), bubble, time_units


def main():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((256, 32))
    y = rng.standard_normal((256, 1))
    P = init()
    rows = []
    g_full = grads(P, X, y)
    for w in (2, 4, 8):
        g_dp, comm = data_parallel(P, X, y, w)
        err = max(np.abs(g_dp[k] - g_full[k]).max() for k in g_full)
        rows.append({"scheme": f"data parallel, {w} workers", "max_diff_vs_single": err,
                     "floats_communicated_per_step": comm, "note": "full model copy per worker"})
    _, out_full = forward(P, X)
    out_mp, comm = model_parallel(P, X)
    rows.append({"scheme": "model parallel, 2 workers (1 layer each)", "max_diff_vs_single": np.abs(out_mp - out_full).max(),
                 "floats_communicated_per_step": 2 * comm, "note": "activations forward + gradients backward"})
    for m in (1, 4, 8, 32):
        out_pp, bubble, tu = pipeline(P, X, micro=m)
        rows.append({"scheme": f"pipeline, 2 stages, {m} micro-batches", "max_diff_vs_single": np.abs(out_pp - out_full).max(),
                     "floats_communicated_per_step": 2 * comm,
                     "note": f"bubble {bubble:.0%}; {tu} time units vs {2 * m} sequential (speedup {2 * m / tu:.2f}x of max 2x)"})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "d1_parallelism_schemes")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print(main().to_string(index=False))
