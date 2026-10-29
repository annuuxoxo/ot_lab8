"""Track F experiments: O0 vs optimized code on 8 example programs, per-pass ablation, register allocation."""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common.utils import save_results  # noqa: E402
import frontend  # noqa: E402
import ir  # noqa: E402
import passes  # noqa: E402
import regalloc  # noqa: E402

TRACK = "trackF_compiler"
PROGRAMS = sorted(glob.glob(os.path.join(HERE, "examples", "*.toy")))


def median_time(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def compare():
    rows = []
    for path in PROGRAMS:
        name = os.path.basename(path)[:-4]
        src = open(path).read()
        code0, t_front = median_time(lambda: ir.generate(frontend.parse(src)))
        code1, t_opt = median_time(lambda: passes.optimize(code0))
        (out0, ex0, c0), t_run0 = median_time(lambda: ir.run(code0))
        (out1, ex1, c1), t_run1 = median_time(lambda: ir.run(code1))
        assert out0 == out1, f"{name}: optimized program prints different output!"
        rows.append({"program": name, "same_output": out0 == out1,
                     "static_O0": ir.static_count(code0), "static_opt": ir.static_count(code1),
                     "executed_O0": ex0, "executed_opt": ex1, "executed_reduction_pct": 100 * (1 - ex1 / ex0),
                     "weighted_cost_O0": c0, "weighted_cost_opt": c1, "cost_reduction_pct": 100 * (1 - c1 / c0),
                     "run_ms_O0": 1000 * t_run0, "run_ms_opt": 1000 * t_run1,
                     "compile_ms_frontend": 1000 * t_front, "compile_ms_optimizer": 1000 * t_opt})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "f1_O0_vs_opt")
    return df


def ablation():
    """Executed instructions with each pass alone vs all passes (shows what each pass contributes)."""
    rows = []
    for path in PROGRAMS:
        name = os.path.basename(path)[:-4]
        code0 = ir.generate(frontend.parse(open(path).read()))
        out0, ex0, _ = ir.run(code0)
        row = {"program": name, "O0": ex0}
        for p in passes.ALL:
            out, ex, _ = ir.run(passes.optimize(code0, (p,)))
            assert out == out0
            row[p] = ex
        row["all"] = ir.run(passes.optimize(code0))[1]
        rows.append(row)
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "f1_pass_ablation")
    return df


def pass_ordering():
    """Phase ordering as an optimization problem.
    Decision: a permutation of the 6 passes (applied repeatedly in that order until nothing changes).
    Objective: total executed instructions over the 8 example programs.  Search space: 6! = 720 orderings.
    (a) exhaustive search = exact optimum;  (b) greedy: build the order one position at a time, each time appending
    the unused pass that gives the lowest objective for the prefix so far."""
    import itertools
    codes = [ir.generate(frontend.parse(open(p).read())) for p in PROGRAMS]
    cache = {}

    def objective(order):
        order = tuple(order)
        if order not in cache:
            cache[order] = sum(ir.run(passes.optimize(c, order))[1] for c in codes)
        return cache[order]

    t0 = time.perf_counter()
    scores = {perm: objective(perm) for perm in itertools.permutations(passes.ALL)}
    t_ex = time.perf_counter() - t0
    best = min(scores, key=scores.get)
    worst = max(scores, key=scores.get)
    t0 = time.perf_counter()
    n_before = len(cache)
    order = []
    while len(order) < len(passes.ALL):
        order.append(min((p for p in passes.ALL if p not in order), key=lambda p: objective(order + [p])))
    t_gr = time.perf_counter() - t0
    vals = np.array(list(scores.values()))
    rows = [
        {"method": "exhaustive: best order", "order": " > ".join(best), "total_executed": scores[best],
         "orders_evaluated": len(scores), "time_s": t_ex},
        {"method": "exhaustive: worst order", "order": " > ".join(worst), "total_executed": scores[worst],
         "orders_evaluated": len(scores), "time_s": t_ex},
        {"method": "greedy construction", "order": " > ".join(order), "total_executed": objective(order),
         "orders_evaluated": len(cache) - n_before + 1, "time_s": t_gr},
        {"method": "original default order", "order": " > ".join(ORIGINAL_ORDER), "total_executed": objective(ORIGINAL_ORDER),
         "orders_evaluated": 1, "time_s": None},
        {"method": "info: distinct objective values among 720 orders", "order": f"{len(set(vals))} distinct; "
         f"{(vals == vals.min()).sum()} orders reach the best", "total_executed": None, "orders_evaluated": None, "time_s": None},
    ]
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "f2_pass_ordering")
    return df, best


ORIGINAL_ORDER = ("const_fold", "cse", "strength", "dce", "licm", "coalesce")


def register_allocation(Ks=(2, 3, 4, 6)):
    rows = []
    for path in PROGRAMS:
        name = os.path.basename(path)[:-4]
        code = passes.optimize(ir.generate(frontend.parse(open(path).read())))
        for K in Ks:
            r = regalloc.allocate(code, K)
            r["program"] = name
            rows.append(r)
    df = pd.DataFrame(rows)[["program", "vars", "interference_edges", "K", "greedy_spilled", "greedy_spill_cost",
                             "ilp_spilled", "ilp_spill_cost", "greedy_ms", "ilp_ms", "ilp_status"]]
    save_results(df, TRACK, "f2_regalloc")
    return df


def main():
    pd.set_option("display.width", 250)
    print("== O0 vs optimized ==\n", compare().to_string(index=False))
    print("\n== Pass ablation (executed instructions) ==\n", ablation().to_string(index=False))
    df, best = pass_ordering()
    print("\n== Pass ordering ==\n", df.to_string(index=False))
    print("\n== Register allocation ==\n", register_allocation().to_string(index=False))


if __name__ == "__main__":
    main()
