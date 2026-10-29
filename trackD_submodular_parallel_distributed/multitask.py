"""Track D part 5: multi-task optimization - equal weighting vs MGDA vs GradNorm on a two-task model.

Shared trunk (Linear 10->32, ReLU) + two heads:
  task 1 = regression with a LARGE target scale (its loss is ~100x bigger),
  task 2 = binary classification.
With equal weights the large-scale loss dominates the shared gradient, starving task 2.
MGDA (2 tasks, closed form): find alpha in [0,1] minimizing ||alpha g1 + (1-alpha) g2|| over the shared gradients;
  that min-norm direction decreases both losses (a Pareto-stationary point when it is zero).
GradNorm: learnable task weights w_i, updated so that each task's weighted gradient norm on the last shared layer
  matches the average norm times (relative training rate)^alpha; weights renormalized to sum to 2.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import save_results  # noqa: E402

TRACK = "trackD_submodular_parallel_distributed"
torch.set_default_dtype(torch.float32)


def make_data(n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 10, generator=g)
    w1, w2 = torch.randn(10, generator=g), torch.randn(10, generator=g)
    y1 = 30 * torch.sin(X @ w1 / 2) + torch.randn(n, generator=g)  # big-scale regression target
    y2 = ((X @ w2 + 0.5 * (X[:, 0] ** 2 - 1)) > 0).float()  # classification
    return X[:1500], y1[:1500], y2[:1500], X[1500:], y1[1500:], y2[1500:]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
        self.h1, self.h2 = nn.Linear(32, 1), nn.Linear(32, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.h1(z).squeeze(-1), self.h2(z).squeeze(-1)


def losses(net, X, y1, y2):
    o1, o2 = net(X)
    return nn.functional.mse_loss(o1, y1), nn.functional.binary_cross_entropy_with_logits(o2, y2)


def train(method, seed, epochs=200, lr=1e-2, alpha=1.5):
    torch.manual_seed(seed)
    Xtr, y1, y2, Xte, t1, t2 = make_data(seed=seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    shared = list(net.trunk.parameters())
    w = torch.ones(2, requires_grad=True)  # GradNorm task weights
    w_opt = torch.optim.Adam([w], lr=0.025)
    L0 = None
    alphas = []
    for _ in range(epochs):
        L1, L2 = losses(net, Xtr, y1, y2)
        opt.zero_grad()
        if method == "equal weights":
            (L1 + L2).backward()
        elif method.startswith("MGDA"):
            g1 = torch.autograd.grad(L1, shared, retain_graph=True)
            g2 = torch.autograd.grad(L2, shared, retain_graph=True)
            g1f, g2f = torch.cat([g.flatten() for g in g1]), torch.cat([g.flatten() for g in g2])
            if "normalized" in method:  # MGDA-UB style: compare directions, not raw magnitudes
                n1, n2 = g1f.norm() + 1e-12, g2f.norm() + 1e-12
                g1, g2 = [g / n1 for g in g1], [g / n2 for g in g2]
                g1f, g2f = g1f / n1, g2f / n2
            # min_a ||a g1 + (1-a) g2||^2  ->  a = ((g2 - g1) . g2) / ||g1 - g2||^2, clipped to [0, 1]
            a = torch.clamp(((g2f - g1f) @ g2f) / ((g1f - g2f) @ (g1f - g2f) + 1e-12), 0, 1).item()
            alphas.append(a)
            # Task heads: each gets the gradient of its OWN loss (L1 + L2 gives exactly that for the heads).
            (L1 + L2).backward()
            # Shared trunk: replace the gradient with the min-norm combination of the task gradients.
            for prm, a1, a2 in zip(shared, g1, g2):
                prm.grad = a * a1 + (1 - a) * a2
        elif method == "GradNorm":
            if L0 is None:
                L0 = torch.stack([L1, L2]).detach()
            last = net.trunk[0].weight
            G = torch.stack([torch.autograd.grad(w[i] * L, last, retain_graph=True, create_graph=True)[0].norm()
                             for i, L in enumerate((L1, L2))])
            ratio = torch.stack([L1, L2]).detach() / L0
            r = ratio / ratio.mean()  # relative inverse training rate
            target = (G.mean() * r ** alpha).detach()
            gn_loss = (G - target).abs().sum()
            w_opt.zero_grad()
            w.grad = torch.autograd.grad(gn_loss, w, retain_graph=True)[0]
            (w[0].detach() * L1 + w[1].detach() * L2).backward()
            w_opt.step()
            with torch.no_grad():  # renormalize so the weights sum to the number of tasks
                w.clamp_(min=1e-3)
                w.mul_(2 / w.sum())
        opt.step()
    with torch.no_grad():
        o1, o2 = net(Xte)
        mse = nn.functional.mse_loss(o1, t1).item()
        acc = ((o2 > 0).float() == t2).float().mean().item()
    extra = {"MGDA": f"mean alpha (weight on task 1) = {np.mean(alphas):.3f}",
             "MGDA (normalized grads)": f"mean alpha (weight on task 1) = {np.mean(alphas):.3f}",
             "GradNorm": f"final weights = {w.detach().numpy().round(3).tolist()}"}.get(method, "")
    return mse, acc, extra


def main(seeds=3):
    rows = []
    for method in ["equal weights", "MGDA", "MGDA (normalized grads)", "GradNorm"]:
        res = [train(method, s) for s in range(seeds)]
        rows.append({"method": method, "task1_test_mse_mean": np.mean([r[0] for r in res]),
                     "task1_test_mse_std": np.std([r[0] for r in res]),
                     "task2_test_acc_mean": np.mean([r[1] for r in res]), "task2_test_acc_std": np.std([r[1] for r in res]),
                     "note_seed0": res[0][2]})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "d2_multitask")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(main().to_string(index=False))
