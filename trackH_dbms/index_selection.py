"""Track H: index selection under a storage budget, formulated as a 0/1 knapsack and solved with greedy, DP and ILP,
compared with SQLite's default optimizer (no secondary indexes; SQLite may still build temporary automatic indexes).

  variables   x_i in {0,1}             build candidate index i or not
  maximize    sum_i benefit_i x_i      benefit_i = workload time saved by index i on its own (measured)
  subject to  sum_i size_i x_i <= B    size_i = pages the index occupies (measured), B = storage budget

Benefits are measured one index at a time, so the knapsack assumes they add up; they do not exactly (two indexes can
serve the same query), so every selected set is then BUILT and its real workload time measured.
Correctness: every query's sorted result must be identical (same hash) in every configuration.

Each query is capped at CAP seconds: some single indexes make SQLite's planner pick a catastrophic plan (Q10 goes from
0.09 s to > 15 s with an index on orders(o_orderdate) alone), and a timed-out query is charged CAP seconds.
Because benefits interact, a fourth method is added: MEASURED greedy (forward selection that builds each candidate on
top of the current set and keeps the one that reduces the real workload time most, within the budget).
"""
import hashlib
import os
import shutil
import sqlite3
import statistics
import sys
import time

import numpy as np
import pandas as pd
import pulp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common.utils import RESULTS, save_results  # noqa: E402
from data import DB  # noqa: E402

TRACK = "trackH_dbms"
CAP = 5.0  # seconds per query
WORK = os.path.join(HERE, "data", "work.db")

# 8 TPC-H-style queries (SQLite syntax, fixed parameters)
QUERIES = {
    "Q3_shipping_priority": """
        SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey
          AND o_orderdate < '1995-03-15' AND l_shipdate > '1995-03-15'
        GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10""",
    "Q4_order_priority": """
        SELECT o_orderpriority, COUNT(*) FROM orders
        WHERE o_orderdate >= '1993-07-01' AND o_orderdate < '1993-10-01'
          AND EXISTS (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
        GROUP BY o_orderpriority ORDER BY o_orderpriority""",
    "Q6_forecast_revenue": """
        SELECT SUM(l_extendedprice * l_discount) FROM lineitem
        WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1994-01-15'
          AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24""",
    "Q10_returned_items": """
        SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, n_name
        FROM customer, orders, lineitem, nation
        WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate >= '1993-10-01'
          AND o_orderdate < '1993-11-01' AND l_returnflag = 'R' AND c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, n_name ORDER BY revenue DESC LIMIT 20""",
    "Q12_shipping_modes": """
        SELECT l_shipmode, COUNT(*) FROM orders, lineitem
        WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate
          AND l_shipdate < l_commitdate AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1994-02-01'
        GROUP BY l_shipmode ORDER BY l_shipmode""",
    "Q14_promotion_effect": """
        SELECT SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
               / SUM(l_extendedprice * (1 - l_discount))
        FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= '1995-09-01' AND l_shipdate < '1995-09-08'""",
    "P1_customer_orders": """
        SELECT o_orderkey, o_totalprice FROM orders WHERE o_custkey = 7381 ORDER BY o_orderkey""",
    "P2_part_suppliers": """
        SELECT ps_suppkey, ps_supplycost, s_name FROM partsupp, supplier
        WHERE ps_partkey = 15119 AND ps_suppkey = s_suppkey ORDER BY ps_suppkey""",
}

# candidate indexes: columns used in WHERE / JOIN clauses of the workload
CANDIDATES = {
    "lineitem(l_orderkey)": "lineitem(l_orderkey)",
    "lineitem(l_shipdate)": "lineitem(l_shipdate)",
    "lineitem(l_receiptdate)": "lineitem(l_receiptdate)",
    "lineitem(l_partkey)": "lineitem(l_partkey)",
    "lineitem(l_returnflag)": "lineitem(l_returnflag)",
    "orders(o_orderkey)": "orders(o_orderkey)",
    "orders(o_orderdate)": "orders(o_orderdate)",
    "orders(o_custkey)": "orders(o_custkey)",
    "customer(c_custkey)": "customer(c_custkey)",
    "customer(c_mktsegment)": "customer(c_mktsegment)",
    "part(p_partkey)": "part(p_partkey)",
    "partsupp(ps_partkey)": "partsupp(ps_partkey)",
    "supplier(s_suppkey)": "supplier(s_suppkey)",
}


def fresh_copy():
    shutil.copyfile(DB, WORK)
    return sqlite3.connect(WORK)


def pages(con):
    return con.execute("PRAGMA page_count").fetchone()[0]


def build(con, names):
    for name in names:
        idx = "idx_" + name.replace("(", "_").replace(")", "")  # e.g. idx_lineitem_l_shipdate
        con.execute(f"CREATE INDEX {idx} ON {CANDIDATES[name]}")
    con.execute("ANALYZE")  # give the planner statistics in every configuration (incl. the baseline)
    con.commit()


def result_hash(rows):
    norm = sorted(tuple(round(x, 6) if isinstance(x, float) else x for x in r) for r in rows)
    return hashlib.sha256(repr(norm).encode()).hexdigest()[:16]


def timed(con, sql):
    """Run a query with a CAP-second limit. Returns (rows or None if it timed out, seconds)."""
    t0 = time.perf_counter()
    deadline = t0 + CAP
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
    try:
        rows = con.execute(sql).fetchall()
        return rows, time.perf_counter() - t0
    except sqlite3.OperationalError:  # interrupted by the progress handler
        return None, CAP
    finally:
        con.set_progress_handler(None, 0)


def run_workload(con, reps=3):
    """Median latency per query over `reps` runs (after one warm-up run), plus result hashes ('TIMEOUT' if capped)."""
    lat, hashes = {}, {}
    for q, sql in QUERIES.items():
        rows, t = timed(con, sql)  # warm-up (page cache)
        if rows is None:
            lat[q], hashes[q] = CAP, "TIMEOUT"
            continue
        hashes[q] = result_hash(rows)
        lat[q] = statistics.median([timed(con, sql)[1] for _ in range(reps)])
    return lat, hashes


def same_results(hashes, base_hash):
    """Every query that finished returned exactly the baseline result; also report timeouts."""
    ok = all(h == base_hash[q] for q, h in hashes.items() if h != "TIMEOUT")
    return ok, [q for q, h in hashes.items() if h == "TIMEOUT"]


def plans(con):
    return {q: [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()] for q, sql in QUERIES.items()}


# ---------------- knapsack solvers ----------------

def greedy(benefit, size, B):
    chosen, used = [], 0
    for i in sorted(benefit, key=lambda k: benefit[k] / size[k], reverse=True):
        if benefit[i] > 0 and used + size[i] <= B:
            chosen.append(i)
            used += size[i]
    return chosen


def dp(benefit, size, B):
    """Exact 0/1 knapsack over integer page counts. best[c] = max benefit with capacity c (1-D table, items one by one)."""
    items = [i for i in benefit if benefit[i] > 0]
    best = np.zeros(B + 1)
    keep = np.zeros((len(items), B + 1), dtype=bool)
    for k, i in enumerate(items):
        w, v = size[i], benefit[i]
        if w > B:
            continue
        cand = best[:B + 1 - w] + v
        better = cand > best[w:]
        keep[k, w:] = better
        best[w:] = np.where(better, cand, best[w:])
    chosen, c = [], B
    for k in range(len(items) - 1, -1, -1):
        if keep[k, c]:
            chosen.append(items[k])
            c -= size[items[k]]
    return chosen[::-1]


def ilp(benefit, size, B):
    m = pulp.LpProblem("index_selection", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x{k}", cat="Binary") for k, i in enumerate(benefit)}
    m += pulp.lpSum(benefit[i] * x[i] for i in benefit)
    m += pulp.lpSum(size[i] * x[i] for i in benefit) <= B
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return [i for i in benefit if x[i].value() > 0.5]


def measured_greedy(base_pages, size, B, base_total):
    """Forward selection on REAL measurements: at each step build every remaining candidate on top of the current set,
    run the workload once, and keep the index that lowers the total time most (if it fits and helps)."""
    con = fresh_copy()
    chosen, used, current, evals = [], 0, base_total, 0
    while True:
        best, best_t = None, current
        for name in CANDIDATES:
            if name in chosen or used + size[name] > B:
                continue
            idx = "idx_" + name.replace("(", "_").replace(")", "")
            con.execute(f"CREATE INDEX {idx} ON {CANDIDATES[name]}")
            con.execute("ANALYZE")
            t = sum(run_workload(con, reps=1)[0].values())
            evals += 1
            con.execute(f"DROP INDEX {idx}")
            if t < best_t * 0.98:  # require a >2% improvement (timing noise)
                best, best_t = name, t
        if best is None:
            break
        chosen.append(best)
        used += size[best]
        current = best_t
        con.execute(f"CREATE INDEX idx_{best.replace('(', '_').replace(')', '')} ON {CANDIDATES[best]}")
    con.close()
    return chosen, evals


def main(budget_fracs=(0.10, 0.25, 0.50)):
    pd.set_option("display.width", 250)
    # 1. baseline = default optimizer, no secondary indexes
    con = fresh_copy()
    build(con, [])
    base_pages = pages(con)
    page_size = con.execute("PRAGMA page_size").fetchone()[0]
    base_lat, base_hash = run_workload(con)
    base_plans = plans(con)
    con.close()
    base_total = sum(base_lat.values())
    print(f"baseline workload: {base_total * 1000:.1f} ms", flush=True)

    # 2. measure size and standalone benefit of every candidate
    cand_rows, benefit, size = [], {}, {}
    for name in CANDIDATES:
        con = fresh_copy()
        build(con, [name])
        size[name] = pages(con) - base_pages
        lat, hashes = run_workload(con)
        con.close()
        ok, timeouts = same_results(hashes, base_hash)
        assert ok, f"results changed with {name}!"
        benefit[name] = (base_total - sum(lat.values())) * 1000  # ms saved per workload run (negative = harmful)
        cand_rows.append({"index": name, "size_pages": size[name], "size_MB": size[name] * page_size / 1e6,
                          "benefit_ms": benefit[name], "benefit_per_MB": benefit[name] / (size[name] * page_size / 1e6),
                          "timeouts": ", ".join(timeouts)})
        print(f"  {name}: {size[name]} pages, benefit {benefit[name]:.1f} ms", flush=True)
    cands = pd.DataFrame(cand_rows).sort_values("benefit_ms", ascending=False)
    save_results(cands, TRACK, "h_candidates")
    total_pages = sum(size.values())

    # 3. select under several budgets, then BUILD each selection and measure the real workload
    rows, plan_dump = [], {"baseline (default optimizer, no indexes)": base_plans}
    rows.append({"budget_frac": 0, "budget_MB": 0, "method": "default optimizer (no indexes)", "indexes": "",
                 "n_indexes": 0, "storage_MB": 0, "predicted_saving_ms": 0, "workload_ms": base_total * 1000,
                 "throughput_qps": len(QUERIES) / base_total, "speedup": 1.0, "results_identical": True, "solve_ms": 0})
    configs = []
    for frac in budget_fracs:
        B = int(frac * total_pages)
        for method, fn in [("greedy (benefit/size)", greedy), ("dynamic programming", dp), ("ILP (CBC)", ilp)]:
            t0 = time.perf_counter()
            chosen = fn(benefit, size, B)
            configs.append((frac, B, method, chosen, (time.perf_counter() - t0) * 1000))
        t0 = time.perf_counter()
        chosen, evals = measured_greedy(base_pages, size, B, base_total)
        configs.append((frac, B, f"measured greedy ({evals} builds)", chosen, (time.perf_counter() - t0) * 1000))
        print(f"  budget {frac:.0%}: measured greedy -> {chosen}", flush=True)
    configs.append((1.0, total_pages, "all candidates", list(CANDIDATES), 0))
    measured = {}
    for frac, B, method, chosen, solve_ms in configs:
        key = tuple(sorted(chosen))
        if key not in measured:  # DP and ILP usually choose the same set; build and time each distinct set once
            con = fresh_copy()
            build(con, chosen)
            lat, hashes = run_workload(con)
            ok, timeouts = same_results(hashes, base_hash)
            measured[key] = (sum(lat.values()), ok and not timeouts, pages(con) - base_pages, lat, timeouts)
            plan_dump[f"{method}, budget {frac:.0%}"] = plans(con)
            con.close()
        tot, same, used, lat, timeouts = measured[key]
        rows.append({"budget_frac": frac, "budget_MB": B * page_size / 1e6, "method": method,
                     "indexes": "; ".join(sorted(chosen)), "n_indexes": len(chosen), "storage_MB": used * page_size / 1e6,
                     "predicted_saving_ms": sum(benefit[i] for i in chosen), "workload_ms": tot * 1000,
                     "throughput_qps": len(QUERIES) / tot, "speedup": base_total / tot, "results_identical": same,
                     "timeouts": ", ".join(timeouts), "solve_ms": solve_ms})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "h_selection")

    # per-query latency: baseline vs the ILP choice at the middle budget
    mid = [c for c in configs if c[2] == "ILP (CBC)" and c[0] == budget_fracs[1]][0]
    q_rows = [{"query": q, "baseline_ms": base_lat[q] * 1000,
               "ilp_25pct_ms": measured[tuple(sorted(mid[3]))][3][q] * 1000} for q in QUERIES]
    q_df = pd.DataFrame(q_rows)
    q_df["speedup"] = q_df.baseline_ms / q_df.ilp_25pct_ms
    save_results(q_df, TRACK, "h_per_query")
    with open(os.path.join(RESULTS, TRACK, "h_query_plans.txt"), "w") as f:
        for cfg, pl in plan_dump.items():
            f.write(f"===== {cfg} =====\n")
            for q, steps in pl.items():
                f.write(f"{q}:\n" + "".join(f"    {s}\n" for s in steps))
    os.remove(WORK)
    print("\n== Candidates ==\n", cands.to_string(index=False))
    print("\n== Selection ==\n", df.drop(columns=["indexes"]).to_string(index=False))
    print("\n== Per query ==\n", q_df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
