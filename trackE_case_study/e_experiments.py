"""Track E experiments: first-fit baseline vs DSATUR (+ back-to-back aware, + local search) vs ILP."""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import save_plot, save_results  # noqa: E402
import timetabling as tt  # noqa: E402

TRACK = "trackE_case_study"
SIZES = (20, 40)
SEEDS = (0, 1, 2)
ILP_LIMIT = {20: 60, 40: 120}


def heuristics(inst):
    """Name -> (timetable, seconds) for the four heuristic methods."""
    out = {}
    for name, f in [("first-fit (baseline)", tt.first_fit),
                    ("DSATUR", tt.dsatur),
                    ("DSATUR b2b-aware", lambda i: tt.dsatur(i, b2b_aware=True))]:
        t0 = time.perf_counter()
        out[name] = (f(inst), time.perf_counter() - t0)
    t0 = time.perf_counter()
    ls = tt.local_search(inst, tt.dsatur(inst, b2b_aware=True))
    out["DSATUR b2b + local search"] = (ls, time.perf_counter() - t0)
    return out


def main():
    rows, inst_rows = [], []
    for n in SIZES:
        for seed in SEEDS:
            inst = tt.make_instance(n, seed)
            conf = tt.conflicts(inst)
            lb, clique, room_lb = tt.clique_lower_bound(inst)
            inst_rows.append({"courses": n, "seed": seed, "students": inst["n_students"],
                              "enrolments": sum(len(e) for e in inst["enrolment"]),
                              "conflict_density": round(len(conf) / (n * (n - 1) / 2), 3),
                              "max_clique": clique, "room_bound": room_lb, "period_lower_bound": lb,
                              "rooms": "/".join(map(str, inst["rooms"]))})
            res = heuristics(inst)
            ls_tt = res["DSATUR b2b + local search"][0]
            ff_period = tt.check(inst, res["first-fit (baseline)"][0])["period_slots"]
            n_slots = min(ff_period, tt.check(inst, ls_tt)["period_slots"] + 3)
            t0 = time.perf_counter()
            ilp_tt, info = tt.ilp(inst, n_slots, time_limit=ILP_LIMIT[n], warm=ls_tt)
            res["ILP (CBC, warm start)"] = (ilp_tt, time.perf_counter() - t0)
            for name, (sol, sec) in res.items():
                c = tt.check(inst, sol)
                row = {"courses": n, "seed": seed, "method": name, **c, "seconds": sec,
                       "period_lower_bound": lb}
                if name.startswith("ILP"):
                    row.update(ilp_status=info["status"], ilp_vars=info["n_vars"], ilp_cons=info["n_cons"],
                               ilp_slots_allowed=n_slots, ilp_time_limit=ILP_LIMIT[n])
                rows.append(row)
                print(n, seed, f"{name:28s}", c["violations"], c["period_slots"], c["back_to_back"],
                      c["objective"], f"{sec:.3f}s", row.get("ilp_status", ""), flush=True)
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "e_results")
    save_results(pd.DataFrame(inst_rows), TRACK, "e_instances")
    summary = (df.groupby(["courses", "method"], sort=False)
                 .agg(period_mean=("period_slots", "mean"), period_std=("period_slots", "std"),
                      b2b_mean=("back_to_back", "mean"), b2b_std=("back_to_back", "std"),
                      objective_mean=("objective", "mean"), objective_std=("objective", "std"),
                      violations=("violations", "sum"), seconds_mean=("seconds", "mean"))
                 .round(3).reset_index())
    save_results(summary, TRACK, "e_summary")
    print(summary.to_string())

    # scaling of the heuristics (ILP omitted: it already hits its time limit at 40 courses)
    scale = []
    for n in (20, 40, 80, 160):
        inst = tt.make_instance(n, 0)
        for name, (sol, sec) in heuristics(inst).items():
            c = tt.check(inst, sol)
            scale.append({"courses": n, "method": name, "seconds": sec, "objective": c["objective"],
                          "period_slots": c["period_slots"], "back_to_back": c["back_to_back"],
                          "violations": c["violations"], "period_lower_bound": tt.clique_lower_bound(inst)[0]})
    sdf = pd.DataFrame(scale)
    save_results(sdf, TRACK, "e_scaling")
    print(sdf.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    methods = list(dict.fromkeys(df["method"]))
    for ax, n in zip(axes, SIZES):
        s = summary[summary.courses == n].set_index("method").loc[methods]
        ax.bar(range(len(methods)), s["objective_mean"], yerr=s["objective_std"], capsize=4,
               color=["#888", "#4c72b0", "#4c72b0", "#2a9d8f", "#e76f51"])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(" ", "\n", 1) for m in methods], fontsize=7)
        ax.set_title(f"{n} courses (mean ± std, 3 seeds)")
        ax.set_ylabel(f"objective = {tt.SLOT_WEIGHT}·slots + back-to-backs")
    fig.tight_layout()
    save_plot(fig, TRACK, "e_objective")


if __name__ == "__main__":
    main()
