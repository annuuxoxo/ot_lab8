# Track D — Submodular, parallel and distributed optimization

Run: `python trackD_submodular_parallel_distributed/run.py` (≈ 30 s) · Tests: `pytest tests/test_trackD.py`
All numbers from `results/trackD_submodular_parallel_distributed/*.csv`. Parallel timings are **measured** on an Apple M4
(10 cores = 4 performance + 6 efficiency), medians of 3 runs. FedAvg / multi-task / data-model-pipeline parallelism are
simulated on one machine.

## 1. Submodular maximization (`submodular.py`, `d1_submodular.csv`)
**Problem:** pick k of 20 sets to cover as many of 200 elements as possible; f(S) = |∪S|.
**Assumptions tested, not assumed:** on 2,000 random (A ⊆ B, e) samples, f is monotone (0 violations) and has
diminishing returns, f(A+e) − f(A) ≥ f(B+e) − f(B) (0 violations). So the (1 − 1/e) ≈ 0.632 greedy guarantee applies.

| objective | DR violations | greedy / optimum, k = 2…6 | guarantee holds? |
|---|---|---|---|
| coverage | 0% | 1.00 at every k | yes (assumptions met) |
| coverage + bonus for 6 complementary pairs | 14.3% | 0.76, **0.61**, 0.81, 0.93, 0.81 | no: falls below 0.632 at k = 3 |
| adversarial pair (items worth 0 alone, 100 together) | — | **0.01** | no |

Greedy costs O(km) evaluations (0.14 ms at k = 6) vs C(m, k) for brute force (59 ms). On non-submodular objectives
the bound simply does not apply, and greedy can be arbitrarily bad.

## 2. Parallel scheduling (`parallel.py`, `d1_scheduling.csv`, `d1_scheduling.png`)
**Workload:** 120 SHA-256 tasks of very uneven size (4 to 200 × 256 KB, lognormal). `hashlib` releases Python's GIL,
so threads run on separate cores. Serial time 0.387 s. Load imbalance = slowest worker's busy time / mean busy time.

| strategy | speedup at p = 4 | p = 10 | efficiency at p = 10 | load imbalance at p = 10 | Amdahl serial fraction (fit) |
|---|---|---|---|---|---|
| static round-robin | 3.09 | 5.51 | 0.55 | 1.38 | 0.098 |
| LPT greedy (makespan heuristic) | 3.56 | 6.80 | 0.68 | 1.06 | 0.050 |
| central queue (dynamic) | 3.50 | 6.81 | 0.68 | 1.07 | 0.053 |
| work stealing (own deque + steal) | 3.50 | 6.71 | 0.67 | 1.09 | 0.056 |

- **Load balancing is the main lever.** A static split leaves workers idle (38% imbalance). LPT, which uses known task costs, and dynamic schemes, which need no cost estimate, all cut makespan by about 19%.
- **Work stealing** ties the central queue here (19 steals at p = 10). Its advantage is less contention on the shared queue, which only shows up with many more, smaller tasks or many more cores.
- **Amdahl:** the fitted serial fraction is about 5%, which caps speedup near 20×. The curve bends after 4 threads: the extra 6 cores are slower efficiency cores, so the "serial fraction" partly reflects mixed hardware, not truly serial code.
- **Gustafson** (`d1_gustafson.csv`): when the work grows with p (24 tasks per thread), scaled speedup is 1.97 / 3.30 / 4.67 / 5.93 / **6.96** for p = 2…10. That's closer to linear than the fixed-size case, as Gustafson's law predicts.
- **Overhead** (`d1_overhead.csv`): 20,000 tiny tasks (4 KB hashes). With 2 threads there's a small gain (1.25–1.67×), but with 4 and 10 threads it is **2–3× slower than serial** (speedup 0.37–0.50). Thread switching and GIL hand-offs cost more than the work.
- **Synchronization** (`d1_sync_locality.csv`): 4 threads × 200k increments: shared counter + lock 69.6 ms vs private counters summed at the end 13.7 ms (5× faster).
- **Locality:** summing a 4000 × 4000 array row by row (contiguous memory) 12.5 ms vs column by column (32 KB stride) 32.4 ms. Same work, 2.6× slower because of cache misses.

## 3. Data / model / pipeline parallelism (`parallelism_demo.py`, simulated)
| scheme | same result as one worker? | communicated per step | note |
|---|---|---|---|
| data parallel, 2 / 4 / 8 workers | yes (diff 3e-17) | 4,224 / 8,448 / 16,896 floats (gradients) | full model copy per worker; communication grows with workers |
| model parallel, 2 workers | yes (exact) | 32,768 floats (activations + their gradients) | each worker holds one layer |
| pipeline, 2 stages, 1 / 4 / 8 / 32 micro-batches | yes | same as model parallel | idle "bubble" 50% / 20% / 11% / 3% → speedup 1.00 / 1.60 / 1.78 / 1.94× (max 2×) |

## 4. Distributed: FedAvg vs centralized (`fed.py`, `d2_fedavg.csv`, `d2_fedavg.png`)
Logistic regression, 4,200 training points split over 10 clients, 40 rounds, 3 seeds. Centralized = 1 epoch over all data per round.

| setting | acc. after 5 rounds | final acc. | rounds to reach 98% of centralized |
|---|---|---|---|
| centralized | 0.866 | 0.878 | 4 |
| FedAvg IID, E = 1 / 5 / 20 local epochs | 0.832 / 0.854 / 0.870 | 0.866 / 0.874 / 0.889 | 34 / 7 / **2** |
| FedAvg non-IID (95% one class per client), E = 1 / 5 / 20 | 0.829 / 0.851 / 0.863 | 0.864 / 0.870 / 0.881 | 35 / 8 / 4 |

More local computation per round cuts the communication rounds needed by up to 17× (34 → 2). Non-IID data slows
convergence (E = 20: 4 rounds instead of 2) because clients drift toward their own label. The effect is mild here because
logistic regression is convex, so averaging still points the right way. FedAvg with E = 20 does 20× more computation
per round than the centralized reference; its higher final accuracy reflects that extra compute, not a better method.

## 5. Multi-task: equal weights vs MGDA vs GradNorm (`multitask.py`, `d2_multitask.csv`, 3 seeds)
Shared trunk + 2 heads. Task 1 is a regression whose loss is about 100× larger than task 2's (a classification).

| method | task 1 test MSE | task 2 test accuracy | what happened |
|---|---|---|---|
| equal weights | 72.5 ± 46.7 | 0.880 ± 0.020 | task 1's large loss dominates the shared gradient, so task 2 suffers |
| MGDA (raw gradients) | 198.0 ± 30.5 | 0.979 ± 0.001 | α = 0: the min-norm combination picks the *smaller* gradient (task 2), so task 1 is neglected |
| **MGDA (normalized gradients)** | **69.7 ± 44.7** | **0.979 ± 0.003** | best on both; with unit gradients the min-norm α is always 0.5 |
| GradNorm (α = 1.5) | 132.5 ± 30.5 | 0.970 ± 0.004 | learned weights [0.013, 1.987]: equalizes gradient norms by almost switching task 1 off |

MGDA's result depends on gradient scale. Normalizing first (as MGDA-UB implementations do) fixes it. GradNorm balances
gradient magnitudes but isn't tuned here (only α = 1.5 was tried).

## Strategy selection (Chapter 6 methods used in Tracks A–D)
| method | variables | constraints | problem size it suits | exact / heuristic | hardware | convergence / guarantee | communication |
|---|---|---|---|---|---|---|---|
| GD / momentum / NAG | continuous | none (or simple) | any (needs full gradient) | local optimum | CPU/GPU, data parallel | O(1/k) convex, linear if strongly convex | gradient all-reduce |
| SGD / mini-batch / Adam | continuous | none | large datasets | local, noisy | GPU, data parallel | noise floor at constant lr | per step |
| Newton / Gauss–Newton | continuous, small d | none | d up to ~10³ (d×d solve) | local, fast | CPU | quadratic near optimum; attracted to saddles (Newton) | high (Hessian) |
| Greedy | discrete | cardinality / knapsack | large | heuristic; (1−1/e) only if monotone submodular | CPU | one pass | low |
| DP | discrete, integer weights | capacity | n·C fits in memory | exact | CPU | — | — |
| Branch-and-Bound / ILP | discrete / mixed | linear | small–medium | exact (exponential worst case) | CPU | proves optimality | — |
| Simulated annealing | binary (QUBO) | as penalties | medium | heuristic | CPU, parallel restarts | probabilistic | low |
| QAOA / VQE | binary / Hamiltonian | as penalties | ≤ ~20 qubits simulated | heuristic (variational) | quantum simulator / noisy hardware | barren plateaus, noise, shots | many circuit runs |
| LPT / work stealing | task assignment | p workers | many tasks | heuristic (LPT ≤ 4/3 OPT) | multicore | — | scheduling overhead |
| FedAvg | continuous, distributed data | data stays on clients | many clients | local | edge devices | slows down with non-IID data | 1 model per client per round |
| MGDA / GradNorm | continuous, multi-objective | Pareto | few tasks | Pareto-stationary (MGDA) | GPU | — | per-task gradients |

## Correctness checks (`tests/test_trackD.py`, all passing)
Coverage is monotone and submodular and greedy meets the bound; the pair bonus violates diminishing returns; every
scheduler runs each task exactly once; data- and model-parallel outputs match the single worker; FedAvg client splits
are disjoint; the MGDA closed form matches a grid search.

## Limitations
Python threads only parallelize work that releases the GIL; timings from one machine with mixed core types;
FedAvg/multi-task are simulations without real network costs; GradNorm and FedAvg learning rates were not tuned.
