# H1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 82c2b52

## Original prompt
```
In trackH_dbms/ with SQLite: load TPC-H at scale factor 0.1 (e.g. generate with DuckDB's tpch extension, export to
SQLite); document the row counts. Use 6 TPC-H-style queries as the workload.
Pick ~10 candidate indexes from the WHERE/JOIN columns; measure each one's size and its standalone benefit
(median of 5 runs). Index selection under a storage budget = 0/1 knapsack: solve with greedy (benefit/size) and
ILP (PuLP). Build each chosen set and measure the real workload latency vs the no-index default, for 2 budgets.
Save EXPLAIN QUERY PLAN before/after. Test: query result hashes are identical in every configuration.
```

## AI response (excerpt)
Generated trackH_dbms/data.py (TPC-H SF 0.1 via DuckDB dbgen -> SQLite, no indexes) and index_selection.py (8-query workload, 13 candidates, measured size and standalone benefit, knapsack by greedy / DP / ILP, build-and-measure of each selection, EXPLAIN QUERY PLAN dump, result hashing).

## Revised prompt / corrections
- Used SQLite instead of PostgreSQL as planned for the 5-day scope.
- First run never finished: an index on orders(o_orderdate) or orders(o_orderkey) alone makes SQLite choose a catastrophic plan for Q10 (>15 s vs 0.09 s); there was no per-query timeout. Added a 5 s cap with a penalty and timeout-aware result checks (failure log #17).
- Because benefits clearly interact, added a 4th method: measured greedy (forward selection on real builds).

## Verification note
Every completed query returned the identical result hash in every configuration; no selected set had a timeout. Tests: DP and ILP equal brute force on 10 random knapsacks, greedy feasible, hash order-invariant, timeout works.
