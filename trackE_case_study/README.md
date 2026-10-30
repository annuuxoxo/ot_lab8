# Track E — Main case study: university exam timetabling

Run: `python trackE_case_study/run.py` (≈ 10 min, mostly the ILP time limits) · Tests: `pytest tests/test_trackE.py`
All numbers from `results/trackE_case_study/*.csv`. Measured on the M4 (single thread; CBC uses one thread).

## 1. The process and why it needs optimization
Every term, a registry office has to place each course's exam in a time slot and a room. When it is done by hand,
courses are typically taken from a list and put into the first slot that still works. That is our **baseline:
first-fit in course-list order**. It produces a valid timetable, but (results below) it needs up to 3 more slots than
necessary (each slot is an extra half-day of rooms and invigilators) and leaves hundreds of students with two exams
back to back on the same day. The domain (education/transport-style timetabling) is different from Tracks F–J.

## 2. Formulation
**Data** (`timetabling.make_instance`, seeded, synthetic; no real enrolment data was available):
- N courses (20 or 40), 15·N students. Each student belongs to a department of 5 courses and takes 3–5 courses, 97% of them from their own department.
- Course popularity is skewed (Zipf-like), so a few courses are large.
- 4 rooms with capacities 100% / 60% / 40% / 25% of the largest course.
- 3 slots per day.
- Resulting conflict density (share of course pairs that share a student): see `e_instances.csv`.

**Decision variables** (ILP form, `timetabling.ilp`):
- x[c,t,r] ∈ {0,1}: course c's exam is in slot t, room r (only rooms large enough for c are created)
- y[t] ∈ {0,1}: slot t is part of the exam period (y[t] ≥ y[t+1], so the period is a prefix of slots)
- b[c,d,t] ≥ 0: conflicting courses c and d are in consecutive same-day slots t and t+1

**Objective:** minimize 20 · (number of slots in the period) + (number of student back-to-backs)

**Hard constraints**
1. every course is scheduled exactly once
2. no student has two exams in the same slot (conflicting courses get different slots)
3. the room fits the course's enrolment
4. a room holds one exam per slot
5. exams only in slots inside the period

**Soft constraint** (in the objective): back-to-back exams, meaning a student has exams in slots t and t+1 on the same day.

**Assumptions** (stated rather than invented as facts):
- One slot costs as much as 20 back-to-backs. This is a policy weight, so it is a parameter (`SLOT_WEIGHT`).
- An exam is not split across rooms.
- Every slot has the same 4 rooms.
- Overnight gaps don't count as back-to-back.

**Success metrics:** period length (slots), back-to-backs, the combined objective, hard-constraint violations (must be 0, from an independent checker), and runtime.

## 3. Methods
| method | idea | file |
|---|---|---|
| first-fit (baseline) | courses in list order, earliest slot with no clash and a free room that fits | `first_fit` |
| DSATUR | graph colouring: next course = the one whose conflicting courses already use the most distinct slots; lowest feasible slot | `dsatur` |
| DSATUR b2b-aware | same order, but choose the open slot that adds the fewest back-to-backs | `dsatur(b2b_aware=True)` |
| DSATUR b2b + local search | then hill-climb: swap two whole slots, or move one course, whenever back-to-backs drop | `local_search` |
| ILP (CBC) | the model above, warm-started from local search, time limit 60 s (20 courses) / 120 s (40 courses) | `ilp` |

**Lower bound** on the period: max(largest clique of conflicting courses, ⌈N / rooms⌉), computed with networkx.

## 4. Results
Instances (`e_instances.csv`): 20 courses / 300 students / ~1,200 enrolments (conflict density 0.72–0.77), and
40 courses / 600 students / ~2,400 enrolments (density 0.49–0.57). **Every timetable below has 0 hard-constraint
violations** according to the independent checker.

**Main table** (`e_summary.csv`, mean ± std over 3 seeded instances; objective = 20·slots + back-to-backs; lower is better)

| courses | method | slots | back-to-backs | objective | vs baseline | runtime |
|---|---|---|---|---|---|---|
| 20 | first-fit (baseline) | 12.3 ± 0.6 | 175 ± 29 | 421.7 ± 38.6 | — | 0.3 ms |
| | DSATUR | 11.7 ± 1.2 | 248 ± 15 | 481.7 ± 23.2 | **+14% (worse)** | 0.3 ms |
| | DSATUR b2b-aware | 11.3 ± 0.6 | 163 ± 43 | 390.0 ± 53.0 | −8% | 0.3 ms |
| | DSATUR b2b + local search | 11.3 ± 0.6 | 51 ± 20 | 277.7 ± 18.9 | −34% | 40 ms |
| | ILP (CBC, 60 s limit) | 12.0 ± 1.0 | **22 ± 9** | **262.0 ± 15.6** | **−38%** | 59 s |
| 40 | first-fit (baseline) | 20.7 ± 2.1 | 183 ± 65 | 596.7 ± 52.2 | — | 1 ms |
| | DSATUR | 19.7 ± 0.6 | 324 ± 161 | 717.0 ± 152.2 | **+20% (worse)** | 1 ms |
| | DSATUR b2b-aware | 20.3 ± 0.6 | 208 ± 145 | 615.0 ± 136.2 | +3% | 2 ms |
| | DSATUR b2b + local search | 20.3 ± 0.6 | **14 ± 6** | 420.7 ± 12.0 | −29% | 0.44 s |
| | ILP (CBC, 120 s limit) | **19.3 ± 0.6 (= lower bound on every seed)** | 16 ± 7 | **403.0 ± 4.4** | **−32%** | 115 s |

Plot: `e_objective.png`. Per-instance rows: `e_results.csv`.

- **Evidence that the process needed optimization:** the list-order baseline leaves **175–183 student back-to-backs**
  and uses up to 3 slots more than the lower bound (`e_results.csv`, 40 courses seed 2: 23 slots vs bound 20).
  The optimized timetables cut back-to-backs by **87% (ILP, 20 courses) to 92% (local search, 40 courses)** and the combined objective by 32–38%.
- **Plain DSATUR is worse than the baseline here.** It is the textbook colouring heuristic, but it minimizes slots by
  packing courses tightly, which creates *more* back-to-backs (248 vs 175), and it barely shortens the period. The
  lower bound explains why: at 40 courses the **binding limit is rooms, not clashes**. The room bound (courses that
  only fit the 1, 2, … largest rooms) is 19–20 slots, while the largest clique is only 9–11. DSATUR orders courses by
  clashes and ignores room sizes. (First-fit happened to reach the bound, 19, on seed 0.)
- **Local search is the most cost-effective step:** it takes 40–440 ms and gets within 4–6% of the ILP's objective.
- **ILP:** 1,500–7,200 binary variables and 4,000–21,000 constraints. CBC never *proved* optimality within the time
  limit (status "Solution Found" on all 6). Still, it beat local search on every instance by trading slots against
  back-to-backs: at 20 courses, seed 0, it uses 13 slots with 20 back-to-backs (280) instead of 12 slots with 42
  (282). At 40 courses it hit the lower bound on the period on all 3 seeds, so its slot count is provably optimal.
- **Scaling** (`e_scaling.csv`, seed 0, heuristics only):

  | courses | first-fit | DSATUR b2b + local search | back-to-backs (FF → LS) | slots (FF / LS / bound) |
  |---|---|---|---|---|
  | 80 | 3 ms | 1.4 s | 94 → 20 | 26 / 26 / 20 |
  | 160 | 12 ms | 11.7 s | 152 → 0 | 47 / 48 / 40 |

  The ILP already needs its full 2-minute limit at 40 courses, so at 80+ courses only the heuristics are practical.
  Local search grows roughly as N² per round (every course × every slot, each re-checked).

## 5. Strategy selection
| method | variable type | problem size it suits | exact / heuristic | hardware | parallelism |
|---|---|---|---|---|---|
| first-fit | discrete assignment | any (O(N·T)) | heuristic, order-dependent | any CPU, < 1 ms | none needed |
| DSATUR (+ b2b) | discrete (graph colouring) | thousands of courses (O(N²)) | heuristic; exact on some graph classes only | CPU, ms | none needed |
| local search | discrete, moves on a feasible timetable | large; cost per round O(N·T) checks | heuristic (local optimum) | CPU | restarts run in parallel trivially |
| ILP (CBC) | binary x, y; continuous b | small: about 20–40 courses within minutes | exact, with a proof of optimality only if it finishes | CPU; one thread here; commercial solvers parallelize branch-and-bound | parallel tree search |
| (not used) QUBO / annealing | binary | tens of courses (Track C: embedding blows up) | heuristic | annealer / simulator | — |
| (not used) continuous methods (Track A) | — | — | not applicable: variables are discrete and constraints combinatorial | — | — |

**Choice for this problem:** variables are binary and the constraints are combinatorial, so continuous methods
(Track A) don't apply. At registry scale (hundreds of courses), DSATUR or first-fit followed by local search is the
practical method: it runs in seconds and gets within ~5% of the ILP. Use the ILP for small instances, or to check
heuristics and give a bound. A QUBO/annealer version would need penalty terms for every hard constraint, and Tracks B
and C showed that large penalties hurt annealing and that dense QUBOs over ~35 variables don't embed on Chimera.

## 6. Correctness
`tests/test_trackE.py` (all passing):
- **Independent checker:** `check()` recomputes every hard constraint from the raw enrolment lists, without the conflict graph or any solver code. A planted student clash, room overflow, double-booked room and missing course are each detected.
- Every heuristic is feasible on 4 instances (20–80 courses), and the period is never below the lower bound.
- Local search never makes the period or the back-to-back count worse.
- On two tiny instances (5 courses, 2 rooms), the ILP's optimum **equals brute force over all 5⁵ slot assignments**. The ILP's own objective equals the checker's, which confirms the pair-based back-to-back count in the model.
- Every reported timetable in `e_results.csv` has 0 violations, according to the checker.

## Limitations
Synthetic data: the structure (departments, Zipf popularity) is plausible, but it is not a real registry. One room
per exam; no invigilator or student-preference constraints; slot weight 20 is a policy choice; the ILP is run with a
time limit, so it is not always proven optimal; CBC is single-threaded here.
