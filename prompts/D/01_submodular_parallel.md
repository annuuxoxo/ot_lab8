# D1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 5ef2fd7

## Original prompt
```
In trackD_submodular_parallel_distributed/:
1. Max-coverage objective: test diminishing returns on random samples, compare greedy with brute force for small k,
   and check the (1-1/e) bound. Show one non-submodular objective where greedy fails.
2. CPU-bound tasks of uneven size on multiprocessing with p = 1, 2, 4, max cores: static split vs LPT greedy vs
   dynamic queue (work stealing). Report makespan, speedup and efficiency (3 repeats), fit Amdahl's serial fraction,
   show one Gustafson scaled-size run, and one tiny-task case where parallel is slower (overhead).
   Add a lock-contention vs local-counter micro-benchmark (synchronization/locality).
3. A short simulated demo of data / model / pipeline parallelism on a tiny NumPy MLP (label it as a simulation).
```

## AI response (excerpt)
Generated `submodular.py` (max coverage, empirical monotonicity/diminishing-returns test, greedy vs brute force, non-submodular pair-bonus and adversarial objectives), `parallel.py` (threads; static round-robin, LPT, central queue, real work stealing with per-worker deques; Amdahl fit, Gustafson scaled run, tiny-task overhead, lock vs private counter, row vs column traversal) and `parallelism_demo.py` (data/model/pipeline parallelism on a NumPy MLP).

## Revised prompt / corrections
- Plan said multiprocessing; the AI used threads on GIL-releasing work so that true work stealing (shared deques) could be implemented.
- First workload was NumPy matmul: speedup only 1.37× on 10 cores. Investigation showed NumPy uses Apple Accelerate (shared AMX coprocessor) which threadpoolctl cannot limit, and the tasks were tiny (16 ms total). Switched to SHA-256 hashing (per-core, releases the GIL) -> 6.8× (failure log #11).

## Verification note
Tests: every scheduler executes each task exactly once (monkeypatched task recorder); coverage passes the submodularity test and greedy meets the bound; data/model parallel outputs equal the single-worker result.
