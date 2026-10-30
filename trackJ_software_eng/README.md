# Track J — Software-engineering optimization: test-suite minimization

Run: `python trackJ_software_eng/run.py` (≈ 8 min; clones the target project first) · Tests: `pytest tests/test_trackJ.py`
All numbers from `results/trackJ_software_eng/*.csv` and `j_meta.json`. Measured on the M4, Python 3.12.

## Target codebase
[python-tabulate](https://github.com/astanin/python-tabulate) at commit `268615a` (pinned in `j_experiments.py`,
cloned into the gitignored `target/` folder):
- library: `tabulate/__init__.py` 2,900 lines + `cli.py` 227 lines
- tests: 6,078 lines, 366 tests (322 pass, 44 skip because optional packages such as `wcwidth` are not installed)

The suite is only run, never changed. The mutation step edits `tabulate/__init__.py` temporarily and restores it with `git checkout` (verified clean afterwards).

## 1. Baseline profile: where the time goes
Full suite, 322 passing tests:
- time inside tests: **0.807 s**
- end-to-end wall time: **1.060 s** (median of 5; range 1.056–1.141)
- pytest start-up alone, with 0 tests selected: 0.139 s

| bottleneck | cost | evidence |
|---|---|---|
| 10 CLI tests that start a new Python process (`python tabulate/cli.py ...`) | 0.606 s, **75% of test time** | `j_tests.csv` |
| one-time imports: pandas 107 ms and numpy 18 ms (marginal, after pytest and tabulate) | charged to whichever test imports them first | `j_import_costs.csv` |
| the other ~300 tests | median 0.13 ms each | `j_tests.csv` |

The 10 slowest tests take 86% of the time, so a few tests decide the cost.

## 2. Formulation
- **Decision variables:**
  - x_t ∈ {0,1} for each of the 322 passing tests: keep test t
  - z_m ∈ {0,1} for each heavy module m (numpy, pandas): the reduced run imports m
- **Objective:** minimize Σ_t base_t·x_t + Σ_m import_m·z_m, the predicted run time of the reduced suite.
  - base_t = the test's time minus any import it happened to pay for
  - each import is paid once per run
- **Constraints:**
  - coverage: for each of the 938 lines that tests cover, Σ_{t covers line} x_t ≥ 1
  - imports: z_m ≥ x_t whenever test t needs module m

  This is a weighted set cover with a fixed charge for shared setup. Lines with identical covering sets are merged
  into one constraint (289 constraints).
- **Measurement:**
  - per-test coverage: each test run alone under coverage.py with **subprocess measurement on**, minus the 550 lines that are executed by importing alone
  - module needs: modules first imported while the test runs alone
  - times: median of 3 full runs
- **Methods:**
  - greedy: most uncovered lines per second of marginal cost, then remove redundant tests; ln(n)+1 approximation for weighted set cover
  - ILP: PuLP/CBC, exact
  - naive baselines: the full suite, and random subsets of the same size as the ILP's (10 seeds)
  - the first model version: ILP on raw per-test times with no import costs (see §4)

## 3. Results (`j_selection.csv`, `j_mutation.csv`, plot `j_minimization.png`)

| suite | tests | predicted time | measured time inside tests | wall time (median of 5) | line coverage (measured by re-running) | mutants killed (of 40) |
|---|---|---|---|---|---|---|
| full suite (baseline) | 322 | 0.806 s | 0.807 s | 1.060 s | 938 / 938 (100%) | **33** |
| greedy | 70 | 0.2235 s | 0.247 s | 0.427 s | 938 / 938 (100%) | 31 |
| **ILP (optimal, proven)** | **68** | **0.2232 s** | **0.245 s (−70%)** | **0.425 s (−60%, 2.5× faster)** | **938 / 938 (100%)** | 30 |
| random, same size (seed 0) | 68 | 0.123 s | 0.122 s | 0.278 s | 720 / 938 (76.8%) | 27 |
| random, 10 seeds (model) | 68 | 0.243 s (mean) | | | 76.8–82.7% | |

- **Gain:** 79% fewer tests, 70% less test time, and 2.5× faster end to end, with every covered line still covered.
  Measured by actually running the reduced suite under coverage, not just predicted.
- **The ILP is cheap to solve** (0.14 s, proven optimal) and greedy is within 0.1% of it (0.2235 vs 0.2232 s). At this size, the solver choice hardly matters. What matters is the cost model (next section).
- **What's left in the reduced suite:**
  - pytest start-up (0.14 s)
  - the pandas + numpy imports (0.126 s), which the 3 pandas tests need: they are the only tests covering tabulate's DataFrame code
  - 2 of the 10 subprocess CLI tests, kept to cover `cli.py`'s script path

  The random suite is faster only because it skips those, and it loses 23% of the coverage.
- **Cost in correctness (fault detection):** 40 seeded operator mutants of `tabulate/__init__.py` on executed lines.
  - The full suite kills 33. The other 7 survive every suite: they are equivalent mutants or unchecked behaviour.
  - The ILP suite kills 30, greedy 31 and random 27. **The ILP suite misses 3 of the 33 faults that the full suite catches (9%)**, even though it covers every line.
  - Example: mutant 24 (`- → +` on line 2773) runs under the reduced suite but no remaining test checks its output.
  - Line coverage is a weaker criterion than "checks the behaviour".

## 4. Getting the model right (failure log #19, #20)
The first version profiled coverage within one run and used raw per-test times. Two flaws only showed up when the
"optimal" suite was actually run:
1. **Coverage blind spot:** the CLI tests run tabulate in a child process, so in-process coverage recorded 0 lines for
   them, and the optimizer dropped them all (the script path of `cli.py` went untested). Fix: per-test isolated runs
   with `patch = subprocess`. The CLI tests now cover 348–401 lines each, and the lines to cover went from 930 to 938.
2. **Order-dependent costs:** in the full run, pandas' 0.11 s import is charged to `test_pandas` (0.125 s), so the
   other pandas tests looked almost free (0.1 ms). The old ILP suite had a modelled cost of 0.012 s, but inside the
   run `test_pandas_firstrow` alone took 0.231 s, and a random suite was faster end to end than the "optimal" one
   (0.27 vs 0.33 s).
   - Fix: model imports as shared setup costs.
   - Now the model predicts 0.223 s against 0.245 s measured (within 10%). Raw times would have predicted 0.097 s for the same suite (2.5× too low).
   - With corrected coverage, the naive ILP happens to choose the same 68 tests (pandas tests are needed for coverage anyway), so the fix improved the *prediction* more than the *choice* here.

## 5. Maintainability and risk
- A minimized suite is a CI fast path, not a replacement. It lost 9% of the mutant detection, and any new code or test
  invalidates the coverage data, so the selection must be recomputed (about 3 minutes of isolated profiling).
- Selection by line coverage keeps one test per behaviour "path" and drops tests that check edge-case outputs on
  already-covered lines. Run the full suite before release or nightly.
- No test code was modified, so readability and maintainability of the tests themselves are unchanged. The cost is a selection file to maintain plus the profiling tool.

## Correctness checks
- `tests/test_trackJ.py` (all passing):
  - ILP equals brute force on 10 random instances, and greedy is never better than it
  - both keep full coverage; greedy removes redundant tests; a case where greedy is suboptimal
  - shared import costs are charged once and change the optimal choice when they should
- Real-run validation: every reduced suite passes on unmodified tabulate, and its measured coverage is 938/938.
- The target checkout is restored after the mutation runs (`git status` clean).

## Limitations
One small project; line coverage (not branch coverage); 40 mutants from simple operator swaps; timings of a
1-second suite on one laptop (5 repeats, range reported); skipped tests (44, optional packages) are not part of the optimization.
