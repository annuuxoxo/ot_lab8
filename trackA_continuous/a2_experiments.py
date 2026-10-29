"""Track A, part 2:
1. Hand-written backprop checked against PyTorch autograd; SGD vs Momentum vs Adam; effect of L2 regularization.
2. Curvature on logistic regression: full Hessian, diagonal approximation, empirical Fisher, Hessian-vector products;
   preconditioned GD vs plain GD.
3. Convergence rates: convex vs strongly convex (L2) objective.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification, make_moons

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import save_plot, save_results, set_seed  # noqa: E402
import mlp  # noqa: E402

TRACK = "trackA_continuous"
torch.set_default_dtype(torch.float64)


# ---------------- 1. Backprop + training ----------------

def backprop_check(X, y, lam=0.01):
    """Compare hand-written gradients with PyTorch autograd on the same parameters."""
    params = mlp.init_params(2, 16, seed=0)
    mine = mlp.backward(params, X, y, lam)
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
    p = torch.sigmoid(torch.tanh(Xt @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"]).ravel()
    L = torch.nn.functional.binary_cross_entropy(p, yt) + 0.5 * lam * (tp["W1"].pow(2).sum() + tp["W2"].pow(2).sum())
    L.backward()
    return max(np.abs(mine[k] - tp[k].grad.numpy()).max() for k in params)


def training_experiment():
    X, y = make_moons(400, noise=0.25, random_state=0)
    Xte, yte = make_moons(400, noise=0.25, random_state=1)
    diff = backprop_check(X, y)
    print(f"max |hand-written grad - autograd grad| = {diff:.2e}")

    init = mlp.init_params(2, 16, seed=0)
    rows, curves, models = [], {}, {}
    configs = [("SGD", "sgd", 0.5, 0.0), ("Momentum", "momentum", 0.05, 0.0), ("Adam", "adam", 0.01, 0.0),
               ("Adam + L2 (lam=0.001)", "adam", 0.01, 0.001), ("Adam + L2 (lam=0.01)", "adam", 0.01, 0.01),
               ("Adam + L2 (lam=0.1)", "adam", 0.01, 0.1)]
    for name, method, lr, lam in configs:
        params, hist = mlp.train(init, X, y, method, lr, epochs=200, batch=32, lam=lam, seed=0)
        curves[name] = hist
        models[name] = params
        rows.append({"config": name, "lr": lr, "lambda": lam, "train_loss": hist[-1],
                     "train_acc": mlp.accuracy(params, X, y), "test_acc": mlp.accuracy(params, Xte, yte),
                     "weight_norm": np.sqrt(np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "a2_mlp_training")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for name, h in curves.items():
        axes[0].plot(h, label=name)
    axes[0].set(yscale="log", xlabel="epoch", ylabel="training loss (incl. L2 term)", title="MLP on make_moons")
    axes[0].legend(fontsize=7)
    xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    for ax, name in zip(axes[1:], ["Adam", "Adam + L2 (lam=0.001)", "Adam + L2 (lam=0.1)"]):
        p, _ = mlp.forward(models[name], grid)
        ax.contourf(xx, yy, p.reshape(xx.shape), levels=20, cmap="RdBu", alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=10)
        ax.set_title(f"{name}: decision boundary")
    save_plot(fig, TRACK, "a2_mlp")
    plt.close(fig)
    return df, diff


# ---------------- 2. Curvature ----------------

def logreg_problem(lam):
    X, y = make_classification(500, 10, n_informative=6, random_state=0)
    X = X * np.linspace(0.2, 5, 10)  # badly scaled features -> ill-conditioned Hessian
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64)

    def f(w):
        z = Xt @ w
        return torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + 0.5 * lam * w @ w

    return f, Xt, yt


def hvp(f, w, v):
    """Hessian-vector product by double backward (Pearlmutter trick): H v = d/dw (grad f . v).
    Costs about two gradient evaluations and never forms H."""
    w = w.detach().requires_grad_(True)
    g = torch.autograd.grad(f(w), w, create_graph=True)[0]
    return torch.autograd.grad(g @ v, w)[0]


def curvature_experiment(lam=1e-3, iters=5000):
    f, Xt, yt = logreg_problem(lam)
    d = Xt.shape[1]
    w0 = torch.zeros(d)

    # Check HVP against the full Hessian
    H = torch.autograd.functional.hessian(f, w0)
    v = torch.randn(d, generator=torch.Generator().manual_seed(0))
    hvp_err = (hvp(f, w0, v) - H @ v).abs().max().item()
    print(f"max |HVP - H v| = {hvp_err:.2e}")

    def grad(w):
        w = w.detach().requires_grad_(True)
        return torch.autograd.grad(f(w), w)[0]

    def per_sample_grads(w):
        """Gradient of each sample's loss (for the empirical Fisher = mean of g g^T)."""
        p = torch.sigmoid(Xt @ w)
        return (p - yt)[:, None] * Xt + lam * w

    def diag_hessian(w):
        """Closed form for logistic regression: H_jj = mean_i p_i (1 - p_i) x_ij^2 + lam."""
        p = torch.sigmoid(Xt @ w)
        return (p * (1 - p)) @ Xt ** 2 / len(yt) + lam

    # check the closed form against d Hessian-vector products with unit vectors e_j (diag_j = (H e_j)_j)
    diag_by_hvp = torch.stack([hvp(f, w0, torch.eye(d)[i])[i] for i in range(d)])
    assert torch.allclose(diag_by_hvp, diag_hessian(w0)), "diagonal Hessian formula disagrees with HVPs"

    def precond(name, w):
        if name == "GD":
            return torch.ones(d)
        if name == "Diagonal Hessian":
            return diag_hessian(w)
        if name == "Diagonal empirical Fisher":
            G = per_sample_grads(w)
            return (G ** 2).mean(0)
        raise ValueError(name)

    f_star_w = w0.clone()
    for _ in range(50):  # Newton to get a precise optimum for f - f*
        g = grad(f_star_w)
        f_star_w = f_star_w - torch.linalg.solve(torch.autograd.functional.hessian(f, f_star_w), g)
    f_star = f(f_star_w).item()

    rows, curves = [], {}
    for name in ["GD", "Diagonal Hessian", "Diagonal empirical Fisher"]:
        best = None
        for lr in np.logspace(-2, 0, 13):  # same fine grid for every method (a coarse grid favoured GD by luck)
            w, hist = w0.clone(), [f(w0).item() - f_star]
            for _ in range(iters):
                w = w - lr * grad(w) / (precond(name, w) + 1e-8)
                hist.append(f(w).item() - f_star)
            if np.isfinite(hist[-1]) and (best is None or hist[-1] < best[1][-1]):
                best = (lr, hist)
        curves[name] = best[1]
        rows.append({"method": name, "best_lr": best[0], "final_gap": best[1][-1],
                     "iters_to_1e-6": next((i for i, g in enumerate(best[1]) if g < 1e-6), None)})

    # Compare curvature matrices at the OPTIMUM. (At w = 0 every p_i = 0.5, so (p_i - y_i)^2 = p_i(1 - p_i) = 0.25
    # and the empirical Fisher equals the Hessian exactly -- a misleading place to compare them.)
    Hs = torch.autograd.functional.hessian(f, f_star_w)
    G = per_sample_grads(f_star_w)
    F_emp = G.T @ G / len(yt)
    D = torch.diag(Hs)

    def cond(M):
        e = torch.linalg.eigvalsh(M)
        return (e.max() / e.min()).item()

    info = {"Hessian condition number at w*": cond(Hs),
            "condition number after diagonal-Hessian preconditioning": cond(Hs / torch.sqrt(D[:, None] * D[None, :])),
            "||empirical Fisher - H|| / ||H|| at w*": (torch.linalg.norm(F_emp - Hs) / torch.linalg.norm(Hs)).item()}
    for k, val in info.items():
        rows.append({"method": "info: " + k, "best_lr": None, "final_gap": val, "iters_to_1e-6": None})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "a2_curvature")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in curves.items():
        ax.semilogy(np.maximum(h, 1e-16), label=name)
    ax.set(xlabel="iteration", ylabel="f - f*", title="Logistic regression: plain vs diagonally preconditioned GD")
    ax.legend()
    save_plot(fig, TRACK, "a2_curvature")
    plt.close(fig)
    return df, hvp_err


# ---------------- 3. Convergence: convex vs strongly convex ----------------

def convergence_experiment(iters=3000):
    """GD on logistic regression on SEPARABLE data with no L2 (convex but not strongly convex: the minimum is at
    infinity, f -> 0 sublinearly) vs with L2 (strongly convex: linear convergence)."""
    X, y = make_classification(300, 2, n_redundant=0, n_informative=2, class_sep=3.0, random_state=3)
    keep = (X[:, 0] > 0.5) == (y == 1)  # make it exactly linearly separable
    X, y = X[keep], y[keep]
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
    rows, curves = [], {}
    for name, lam in [("convex (no L2, separable data)", 0.0), ("strongly convex (L2 = 0.01)", 0.01)]:
        def f(w, lam=lam):
            return torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w, yt) + 0.5 * lam * w @ w

        w = torch.zeros(2)
        hist = []
        for _ in range(iters):
            w = w.detach().requires_grad_(True)
            fw = f(w)
            hist.append(fw.item())
            g = torch.autograd.grad(fw, w)[0]
            w = w - 1.0 * g
        f_star = 0.0 if lam == 0 else min(hist) - 1e-13
        gaps = np.array(hist) - f_star
        curves[name] = gaps
        k = np.arange(1, len(gaps) + 1)
        # slope of log(gap) vs log(k) (sublinear ~ -1) and vs k (linear rate)
        half = len(gaps) // 2
        above = gaps > 1e-10  # ignore the machine-precision floor when fitting rates
        kk, gg = k[above], gaps[above]
        h2 = len(kk) // 2
        rows.append({"case": name, "final_gap": gaps[-1],
                     # sublinear O(1/k^a): slope of log(gap) vs log(k), expect about -1
                     "loglog_slope": np.polyfit(np.log(kk[h2:]), np.log(gg[h2:]), 1)[0],
                     # linear: gap ~ rho^k, so rho = exp(slope of log(gap) vs k); rho < 1 clearly = linear rate
                     "linear_rate_rho": float(np.exp(np.polyfit(kk[h2:], np.log(gg[h2:]), 1)[0])),
                     "final_||w||": torch.linalg.norm(w).item()})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "a2_convergence")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in curves.items():
        a1.semilogy(np.maximum(g, 1e-16), label=name)
        a2.loglog(np.arange(1, len(g) + 1), np.maximum(g, 1e-16), label=name)
    k = np.arange(1, iters + 1)
    a2.loglog(k, curves["convex (no L2, separable data)"][0] / k, "k--", label="O(1/k) reference")
    a1.set(xlabel="iteration", ylabel="f - f*", title="semilog: straight line = linear convergence")
    a2.set(xlabel="iteration", ylabel="f - f*", title="log-log: slope ≈ -1 = sublinear O(1/k)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    save_plot(fig, TRACK, "a2_convergence")
    plt.close(fig)
    return df


def main():
    set_seed(0)
    pd.set_option("display.width", 200)
    df, diff = training_experiment()
    print("== MLP ==\n", df.to_string(index=False))
    df, err = curvature_experiment()
    print("\n== Curvature ==\n", df.to_string(index=False))
    print("\n== Convergence ==\n", convergence_experiment().to_string(index=False))


if __name__ == "__main__":
    main()
