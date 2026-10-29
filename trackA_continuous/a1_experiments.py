"""Track A, part 1: compare optimizers on test landscapes, SGD vs mini-batch on a finite sum,
and Gauss-Newton vs Newton vs GD on nonlinear least squares."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import save_plot, save_results, set_seed  # noqa: E402
from functions import ALL  # noqa: E402
from optimizers import FIRST_ORDER, GaussNewton, GD, Adam, Momentum, Newton, run  # noqa: E402
from problems import ExpFit, LinReg  # noqa: E402

TRACK = "trackA_continuous"
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
ITERS, TOL = 5000, 1e-8


def pick_best(runs):
    """Best learning rate = lowest final objective among runs that did not diverge;
    ties broken by fewer iterations. Every method gets the same grid -> same tuning budget."""
    ok = [r for r in runs if not r["diverged"]]
    return min(ok, key=lambda r: (round(r["f"], 12), r["iters"]))


def landscapes():
    rows, sweep_rows = [], []
    for prob in ALL:
        best_runs = {}
        for cls in FIRST_ORDER:
            runs = []
            for lr in LR_GRID:
                r = run(cls(lr), prob, prob.x0, ITERS, TOL)
                r["lr"] = lr
                runs.append(r)
                sweep_rows.append({"function": prob.name, "method": cls.name, "lr": lr,
                                   "final_f": r["f"], "diverged": r["diverged"], "iters": r["iters"]})
            best_runs[cls.name] = pick_best(runs)
        r = run(Newton(), prob, prob.x0, ITERS, TOL)
        r["lr"] = 1.0
        best_runs["Newton"] = r

        for name, r in best_runs.items():
            rows.append({"function": prob.name, "method": name, "best_lr": r["lr"],
                         "final_f": r["f"], "gap_f_minus_fstar": r["f"] - prob.f_star,
                         "iters": r["iters"], "time_ms": 1000 * r["time_s"],
                         "converged": r["converged"],
                         "final_x": np.array2string(r["x"], precision=4)})
        _plot_function(prob, best_runs)

    df = pd.DataFrame(rows)
    save_results(df, TRACK, "a1_landscapes")
    sweep = pd.DataFrame(sweep_rows)
    save_results(sweep, TRACK, "a1_lr_sweep")
    _plot_sweep(sweep)
    return df


def _plot_function(prob, best_runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in best_runs.items():
        gap = np.maximum(r["hist"] - prob.f_star, 1e-16)
        ax1.semilogy(gap, label=f"{name} (lr={r['lr']:g})")
    ax1.set(xlabel="iteration", ylabel="f(x) - f*", title=f"{prob.name}: convergence (best lr per method)")
    ax1.set_xscale("symlog", linthresh=10)
    ax1.legend(fontsize=8)

    # Contour plot with trajectories
    allpts = np.vstack([r["path"] for r in best_runs.values()] + [prob.x0[None, :]])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.3 * (hi - lo) + 0.3
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 200)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 200)
    Z = np.array([[prob.f(np.array([a, b])) for a in xs] for b in ys])
    ax2.contour(xs, ys, Z - Z.min() + 1e-3, levels=np.logspace(-3, np.log10(Z.max() - Z.min() + 1e-3), 25),
                cmap="Greys", linewidths=0.6)
    for name, r in best_runs.items():
        ax2.plot(r["path"][:, 0], r["path"][:, 1], ".-", ms=2, lw=1, label=name)
    ax2.plot(*prob.x0, "k*", ms=12, label="start")
    ax2.set(title=f"{prob.name}: trajectories", xlabel="x1", ylabel="x2")
    ax2.legend(fontsize=7)
    save_plot(fig, TRACK, f"a1_{prob.name}")
    plt.close(fig)


def _plot_sweep(sweep):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (fname, g) in zip(axes, sweep.groupby("function", sort=False)):
        for m, gm in g.groupby("method", sort=False):
            vals = np.where(gm["diverged"], np.nan, gm["final_f"].abs() + 1e-16)
            ax.loglog(gm["lr"], vals, "o-", label=m)
        ax.set(title=f"{fname}: final |f| after {ITERS} iters", xlabel="learning rate")
        ax.legend(fontsize=7)
    fig.suptitle("Learning-rate sensitivity (missing points = diverged)")
    save_plot(fig, TRACK, "a1_lr_sweep")
    plt.close(fig)


class _Batch:
    """Wraps a finite-sum problem so an optimizer's prob.grad(x) returns a mini-batch gradient."""

    def __init__(self, prob, idx):
        self.prob, self.idx = prob, idx

    def grad(self, w):
        return self.prob.grad_batch(w, self.idx)


def _sgd_run(prob, opt, batch, epochs, seed):
    """One run of (mini-batch) SGD. Each epoch visits every sample once in a random order,
    so all methods use the same number of per-sample gradient evaluations (fair budget)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(prob.d)
    opt.reset(prob.d)
    gaps = [prob.f(w) - prob.f_star]
    for _ in range(epochs):
        perm = rng.permutation(prob.n)
        for s in range(0, prob.n, batch):
            w = opt.step(w, _Batch(prob, perm[s:s + batch]))
        gaps.append(prob.f(w) - prob.f_star)
    return np.array(gaps)


def stochastic(epochs=10, seeds=5):
    """GD vs SGD vs mini-batch SGD vs mini-batch Momentum/Adam on noisy (sampled) gradients."""
    prob = LinReg()
    configs = [("GD (full batch)", GD, prob.n), ("SGD (B=1)", GD, 1),
               ("Mini-batch SGD (B=32)", GD, 32), ("Mini-batch Momentum (B=32)", Momentum, 32),
               ("Mini-batch Adam (B=32)", Adam, 32)]
    import time
    rows, curves = [], {}
    for name, cls, B in configs:
        # tune lr on seed 0 only, then evaluate on fresh seeds
        best_lr, best_gap = None, np.inf
        for lr in LR_GRID:
            g = _sgd_run(prob, cls(lr), B, epochs, seed=0)[-1]
            if np.isfinite(g) and g < best_gap:
                best_lr, best_gap = lr, g
        all_gaps, times = [], []
        for s in range(1, seeds + 1):
            t0 = time.perf_counter()
            all_gaps.append(_sgd_run(prob, cls(best_lr), B, epochs, seed=s))
            times.append(time.perf_counter() - t0)
        all_gaps = np.array(all_gaps)
        curves[name] = all_gaps
        rows.append({"method": name, "batch": B, "best_lr": best_lr,
                     "updates_per_epoch": int(np.ceil(prob.n / B)),
                     "final_gap_mean": all_gaps[:, -1].mean(), "final_gap_std": all_gaps[:, -1].std(),
                     "time_ms_mean": 1000 * np.mean(times)})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "a1_stochastic")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in curves.items():
        m, sd = g.mean(0), g.std(0)
        ax.semilogy(m, label=name)
        ax.fill_between(range(len(m)), np.maximum(m - sd, 1e-16), m + sd, alpha=0.2)
    ax.set(xlabel="epoch (= n sample-gradients)", ylabel="f(w) - f*",
           title=f"Linear regression n={prob.n}, d={prob.d}: mean ± std over {seeds} seeds")
    ax.legend(fontsize=8)
    save_plot(fig, TRACK, "a1_stochastic")
    plt.close(fig)
    return df


def gauss_newton():
    prob = ExpFit()
    ref = least_squares(prob.residual, prob.x0, jac=prob.jac, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    gd_runs = [dict(run(GD(lr), prob, prob.x0, 20000, TOL), lr=lr) for lr in LR_GRID]
    results = {"GD": pick_best(gd_runs),
               "Newton": dict(run(Newton(), prob, prob.x0, 100, TOL), lr=1.0),
               "Gauss-Newton": dict(run(GaussNewton(), prob, prob.x0, 100, TOL), lr=1.0)}
    rows = []
    for name, r in results.items():
        rows.append({"method": name, "lr": r["lr"], "iters": r["iters"], "converged": r["converged"],
                     "final_f": r["f"], "a": r["x"][0], "b": r["x"][1],
                     "dist_to_scipy": np.linalg.norm(r["x"] - ref), "time_ms": 1000 * r["time_s"]})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "a1_gauss_newton")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fstar = prob.f(ref)
    for name, r in results.items():
        ax1.semilogy(np.maximum(r["hist"] - fstar, 1e-16), label=name)
    ax1.set(xscale="symlog", xlabel="iteration", ylabel="f - f*", title="Nonlinear least squares: y = a·exp(b·t)")
    ax1.legend()
    ax2.plot(prob.t, prob.y, "k.", label="data")
    ax2.plot(prob.t, results["Gauss-Newton"]["x"][0] * np.exp(results["Gauss-Newton"]["x"][1] * prob.t),
             label="Gauss-Newton fit")
    ax2.legend()
    save_plot(fig, TRACK, "a1_gauss_newton")
    plt.close(fig)
    return df


def main():
    set_seed(0)
    pd.set_option("display.width", 200)
    print("== Landscapes ==\n", landscapes().to_string(index=False))
    print("\n== Stochastic ==\n", stochastic().to_string(index=False))
    print("\n== Gauss-Newton ==\n", gauss_newton().to_string(index=False))


if __name__ == "__main__":
    main()
