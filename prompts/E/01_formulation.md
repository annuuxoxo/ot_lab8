# E1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-30
- **Commit:** e85049f

## Original prompt
```
Formulate exam timetabling for ~40 courses as an optimization problem: courses sharing students can't share a slot,
rooms have capacity, and we minimize slots used plus back-to-back exams. Give decision variables, objective, hard/soft
constraints, assumptions and metrics. Ask me for missing information instead of inventing it.
Propose a seeded synthetic enrolment dataset.
```

## AI response (excerpt)
Binary x[c,t,r] (course, slot, room), y[t] (slot used, prefix-ordered), continuous b[c,d,t] for back-to-backs of
conflicting pairs; objective 20·Σy + Σ shared(c,d)·b; hard constraints: once per course, no clash, room capacity,
one exam per room per slot. Missing information was not invented: it was listed as explicit, parameterized assumptions
(slot weight 20, 3 slots/day, 4 rooms, one room per exam) in trackE_case_study/README.md. Dataset: students in
departments of 5 courses, 3–5 courses each, Zipf-like popularity, rooms scaled to the largest course.

## Revised prompt / corrections
- The first generator (80% own-department courses, 4 departments) gave a conflict density of 0.9 (17 of 20 courses all clashed), which is unrealistic. Changed to departments of 5 courses and 97% own-department, giving density 0.49–0.77 (failure log #18).
- The first lower bound used only the largest clique (11) while every method needed ~20 slots. Added a room-capacity bound (courses that only fit the k largest rooms); at 40 courses it is tight (19–20) and the ILP meets it on every seed.

## Verification note
Hand-checked the back-to-back definition with a 3-course example (in tests). The ILP's pair-based back-to-back count equals the checker's per-student count on brute-forced tiny instances.
