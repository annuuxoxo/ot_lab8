# Track H — DBMS optimization: index selection on TPC-H

Run: `python trackH_dbms/run.py` (≈ 5 min; generates the data first) · Tests: `pytest tests/test_trackH.py`
All numbers from `results/trackH_dbms/*.csv`; query plans in `results/trackH_dbms/h_query_plans.txt`. Measured on the M4.

## Setup
- **DBMS:** SQLite 3 (Python `sqlite3`).
- **Data:** TPC-H at **scale factor 0.1**, generated with DuckDB's `tpch` extension and exported to SQLite (`data.py`):
  lineitem 600,572 rows · orders 150,000 · partsupp 80,000 · part 20,000 · customer 15,000 · supplier 1,000 · nation 25 · region 5 (119 MB).
  Starting point: no indexes at all.
- **Workload:** 8 queries. Six are TPC-H style (Q3, Q4, Q6, Q10, Q12, Q14 with fixed parameters); two are point lookups (orders of one customer, suppliers of one part).
- **Baseline = SQLite's default optimizer**, with `ANALYZE` statistics but no secondary indexes. It is already smart: it
  builds temporary *automatic indexes* and Bloom filters for joins. Baseline workload: **1,006 ms**.
- **Fairness:** every configuration starts from a fresh copy of the same database and gets `ANALYZE`. Each query has
  a warm-up run, then the median of 3 timed runs, with a 5 s cap per query.

## Formulation
Choose which of 13 candidate indexes (columns in the workload's WHERE / JOIN clauses) to build.

  x_i ∈ {0,1} build index i · **maximize** Σ benefit_i·x_i · **subject to** Σ size_i·x_i ≤ B

- benefit_i = workload time saved with index i alone (measured)
- size_i = pages it occupies (measured)
- B = storage budget: 10% / 25% / 50% of all candidates' size (5 / 13 / 25 MB)

That's a 0/1 knapsack, solved with greedy (benefit/size), dynamic programming over pages, and ILP (CBC). Every chosen
set is then actually built and timed, because the model assumes benefits simply add up.

## Candidate measurements (`h_candidates.csv`)
| index | size | benefit alone |
|---|---|---|
| lineitem(l_orderkey) | 7.2 MB | **+458 ms** |
| lineitem(l_shipdate) | 11.4 MB | +149 ms |
| customer(c_custkey) | 0.16 MB | +70 ms |
| lineitem(l_returnflag), l_receiptdate, part(p_partkey), partsupp, supplier, c_mktsegment | | +1 to +76 ms |
| orders(o_custkey) | 1.6 MB | −186 ms |
| lineitem(l_partkey) | 6.6 MB | −1,089 ms |
| **orders(o_orderkey)** | 1.8 MB | **−4,591 ms (Q10 hits the 5 s cap)** |
| **orders(o_orderdate)** | 2.8 MB | **−4,827 ms (Q10 hits the 5 s cap)** |

**An index can make a query much slower.** With `orders(o_orderdate)` alone, SQLite reorders Q10's join: it starts
from nation and customer, then reaches lineitem through an automatic index on `l_returnflag`, which matches a third of
the table, repeated for every customer. Q10 goes from 0.17 s to more than 5 s (over 15 s without the cap). Add
`lineitem(l_orderkey)` as well and the plan becomes orders-by-date → lineitem-by-key, which is fast (plans in `h_query_plans.txt`).

## Results (`h_selection.csv`)
| budget | method | indexes chosen | storage | workload | speedup | throughput |
|---|---|---|---|---|---|---|
| — | default optimizer, no indexes | — | 0 | 1,006 ms | 1.00× | 8.0 q/s |
| 10% (5 MB) | greedy = DP = ILP | c_custkey, c_mktsegment, p_partkey, ps_partkey, s_suppkey | 1.5 MB | 929 ms | 1.08× | 8.6 q/s |
| | **measured greedy** | c_custkey, **o_orderkey**, s_suppkey | 2.0 MB | **739 ms** | **1.36×** | 10.8 q/s |
| 25% (13 MB) | greedy = DP = ILP | above + l_orderkey | 8.7 MB | **457 ms** | **2.20×** | 17.5 q/s |
| | measured greedy | c_custkey, l_orderkey | 7.3 MB | 462 ms | 2.18× | 17.3 q/s |
| 50% (25 MB) | greedy | + l_shipdate | 20.1 MB | **323 ms** | **3.11×** | 24.7 q/s |
| | DP = ILP (model optimum) | c_custkey, l_orderkey, l_returnflag, l_shipdate, p_partkey, s_suppkey | 25.0 MB | 331 ms | 3.04× | 24.2 q/s |
| | measured greedy | c_custkey, l_orderkey, l_shipdate | 18.8 MB | 339 ms | 2.97× | 23.6 q/s |
| 100% | all 13 candidates | | 50.5 MB | **204 ms** | **4.93×** | 39.2 q/s |

- **DP and ILP always pick the same set** (both exact for the model), and greedy matches them at 10% and 25%. Solve
  times: greedy 0.01 ms, DP 0.1–0.4 ms, ILP 75–1,100 ms (CBC start-up). At this size, the choice of solver hardly matters.
- **The model's error matters more than the solver.** At 50%, the model's optimum (DP/ILP) was slightly slower in reality
  than greedy's set (331 vs 323 ms). With all 13 indexes the model predicts a loss of 9.8 s, but the real workload
  is the fastest of all (4.93×): the "harmful" indexes become excellent once their partners exist.
- **Measured greedy,** which builds every candidate on top of the current set, captures these interactions. It wins
  clearly when the budget is small (1.36× vs 1.08×, using an index the model rejects as harmful), but needs 26–42 real
  index builds (60–85 s) vs microseconds for the model-based solvers, and it is myopic too (stops early at 50%).
- **Per query** (`h_per_query.csv`, ILP at 25%): Q4 7.8×, Q10 5.8×, Q3 4.1×, the point lookup P2 423× (0.01 ms); but
  Q12 got **slower** (131 → 169 ms) and Q6/Q14 barely changed (they scan date ranges that need `l_shipdate`, not chosen at 25%).

## Correctness
Every query's sorted result (floats rounded to 6 decimals) has the **same hash in every configuration** (asserted for
each candidate, and `results_identical = True` for every selected set, with no timeouts in any selected set). Tests:
DP and ILP equal brute force on 10 random knapsacks (including negative benefits); greedy is feasible and never beats
the optimum; the result hash ignores row order; the timeout cap interrupts a runaway query; TPC-H row counts.

## Limitations
SQLite only (no PostgreSQL or real buffer-pool tuning); 8 queries; latencies are warm-cache medians on one laptop;
the knapsack treats benefits as additive, which this experiment shows is the main weakness of the formulation.
Join-order selection is not covered (a next step).
