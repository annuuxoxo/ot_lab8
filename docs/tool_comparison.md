# AI Tool Comparison

**Task:** prompt B1 (0/1 knapsack: greedy, DP, own Branch-and-Bound, PuLP ILP; seeded instances n = 10/50/100/200;
tests against brute force), given to two set-ups with the same wording (`prompts/B/01_knapsack.md`).

| | Tool 1 | Tool 2 |
|---|---|---|
| Set-up | Claude Code (CLI), **Claude Opus 5.5**, the main session that built this repo | **Claude Sonnet 5.5**, run as a separate Claude Code agent with a fresh context, on 2026-10-30 |
| Isolation | — | Worked in an empty scratch folder; told not to read this repository; got only the B1 prompt plus the Python path |
| Output kept in | `trackB_discrete/` (commit `131d1cf`) | `prompts/B/tool2_sonnet_B1/` (code, summary CSV, plot, as produced) |

**Caveat:** both tools are Claude models from the same vendor, used through the same CLI, so this compares two
models more than two products. A comparison with a non-Claude tool (ChatGPT, Copilot or Gemini) on the same prompt
would be stronger. Add a third column here if one is run.

## Results
| criterion | Opus 5.5 (tool 1) | Sonnet 5.5 (tool 2) |
|---|---|---|
| Correct on first try? | Algorithms yes, but **8 tests failed** at first: the ILP used the PuLP 2.x API while pip had installed PuLP 4.2, which changed the API and dropped the bundled solver (failure log #4) | **Yes:** 20/20 tests passed on the first run. I re-ran its tests, and its DP, B&B and ILP equal this repo's DP on all 12 benchmark instances |
| Hallucinations | No invented functions; the API mismatch was out-of-date library knowledge | None found (correct PuLP 2.x API, correct NumPy DP slicing) |
| Time to working code | Longer: one environment fix (pin `pulp==2.9.0`), plus a readability rewrite of the feature-selection counting | 47 s wall time, 3 tool calls, 153 lines |
| Prompt iterations / revisions | 2 revisions (PuLP pin; refactor) | 0 revisions |
| Scope and rigour | Did more than asked: returns the chosen items and checks feasibility with an independent `check()`; "weakly correlated" (harder) instances; feature-selection experiment; 8 brute-force test instances | Did exactly what was asked: returns only objective values, so the solution itself (which items) is never checked for feasibility; brute-force tests on 20 instances (n = 5–15), more than tool 1 |
| Measurement quality | Single timing per run (no repeats) | Single timing per run (no repeats); the ILP time includes CBC start-up, which it does not mention |
| Readability | Docstrings, separated experiment and test files, README with interpretation | Compact and readable, few comments; no README or interpretation of the results |

**Fairness note:** tool 2 ran in this repo's existing venv, where `pulp==2.9.0` was already pinned. So it could not
hit the PuLP 4.2 problem that tool 1 hit during setup. Its "first try" advantage is partly the environment, not the model.

## Verdict
For a well-specified textbook task, both produced correct code. The smaller model was faster and needed no revisions
in a prepared environment. The larger model added the verification that matters for grading: feasibility checks of
the actual solutions, harder instances, and an interpretation of the results. Neither repeated its timings. Both
outputs needed a human to check them against an independent solver before they could be trusted, which is the lesson
of the failure log as a whole.
