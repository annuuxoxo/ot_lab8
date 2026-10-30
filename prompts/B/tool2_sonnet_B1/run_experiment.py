import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from knapsack import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def timed(f, *a):
    t = time.perf_counter()
    r = f(*a)
    return r, time.perf_counter() - t


rows = []
for n in (10, 50, 100, 200):
    for s in range(3):
        v, w, cap = make_instance(n, seed=1000 * n + s)
        opt, t_dp = timed(dp, v, w, cap)
        g, t_g = timed(greedy, v, w, cap)
        (b, nodes), t_b = timed(branch_and_bound, v, w, cap)
        i, t_i = timed(ilp, v, w, cap)
        for m, obj, t, nd in (("greedy", g, t_g, None), ("dp", opt, t_dp, None),
                              ("bnb", b, t_b, nodes), ("ilp", i, t_i, None)):
            rows.append(dict(n=n, seed=s, method=m, objective=obj,
                             gap_pct=100 * (opt - obj) / opt, runtime_s=t, bnb_nodes=nd))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "knapsack_results.csv"), index=False)
summ = df.groupby(["n", "method"]).agg(objective=("objective", "mean"), gap_pct=("gap_pct", "mean"),
        runtime_s=("runtime_s", "mean"), bnb_nodes=("bnb_nodes", "mean")).reset_index()
summ.to_csv(os.path.join(RES, "knapsack_summary.csv"), index=False)
print(df.to_string(index=False))
print(summ.to_string(index=False))

plt.figure(figsize=(6, 4))
for m, g in summ.groupby("method"):
    plt.plot(g.n, g.runtime_s, marker="o", label=m)
plt.yscale("log"); plt.xlabel("n"); plt.ylabel("mean runtime (s)")
plt.title("Knapsack runtime vs n"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(RES, "runtime_vs_n.png"), dpi=150)
