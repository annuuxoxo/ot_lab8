# J1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-30
- **Commit:** 6486a3d

## Original prompt
```
In trackJ_software_eng/: clone a small public Python project with >1,000 lines and a fast pytest suite (suggest 2,
e.g. tabulate or python-slugify). Profile per-test runtime and per-test line coverage (coverage.py --cov-context=test).
Formulate: x_t ∈ {0,1}, minimize total runtime s.t. every line covered by the full suite stays covered (set cover).
Solve with greedy and ILP; compare with the full suite. Report runtime saved and coverage retained, and briefly note
the maintainability risk.
```

## AI response (excerpt)
Chose python-tabulate (2,900-line module; python-slugify is under 1,000 lines). Generated tj_plugin.py (exact
per-test durations, per-test coverage contexts, selection by node id), minimize.py (greedy, ILP, random baseline),
j_experiments.py (profile, solve, run each reduced suite for real, 40-mutant fault-detection check) and tests.

## Revised prompt / corrections
- Running the cloned project's tests and installing pytest-cov were blocked by the auto-mode safety check; after asking the user, running tabulate's own tests in the venv was approved. pytest-cov was replaced by the small plugin (coverage.py was already installed).
- My own unit test asserted the wrong greedy choice (the "big" test actually had the better lines-per-second ratio); fixed the test case, not the code.
- First full run: the CLI tests showed 0 covered lines (they run in a subprocess), so the optimizer dropped them all (failure log #19). Switched to isolated per-test runs under coverage.py with `patch = subprocess`.
- Same run: a random subset ran faster than the "optimal" ILP suite. Cause: lazy pandas/numpy imports charged to whichever test runs first (failure log #20). Added module tracking and a fixed-charge set cover (import costs paid once).
- Added the mutation check, because the prompt asked about risk and line coverage alone cannot show it.

## Verification note
Each reduced suite was re-run on unmodified tabulate: all pass and measured coverage is 938/938 lines. The model now predicts 0.223 s vs 0.245 s measured. ILP = brute force on 10 random instances in the unit tests. Target checkout restored (git status clean) after mutation runs.
