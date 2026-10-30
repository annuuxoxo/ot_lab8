"""Exam timetabling: data generator, independent checker, and three solvers.

Instance
- courses 0..N-1, each with an enrolment list (student ids)
- rooms with capacities; a room holds at most one exam per slot
- slots 0,1,2,... grouped into days of SLOTS_PER_DAY; slots t and t+1 on the same day are "back-to-back"

A timetable is a dict course -> (slot, room).
Objective (lower is better): SLOT_WEIGHT * period_length + back_to_backs
- period_length = last used slot + 1 (how many slots the exam period needs)
- back_to_backs = number of (student, t) where the student sits exams in slots t and t+1 of the same day
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
import pulp

SLOTS_PER_DAY = 3
SLOT_WEIGHT = 20   # one extra slot costs as much as 20 student back-to-backs (assumption, see README)


# ---------------------------------------------------------------- data
def make_instance(n_courses: int, seed: int, students_per_course: int = 15, courses_per_dept: int = 5):
    """Seeded synthetic enrolment.

    Students belong to a department (5 courses each) and take 3-5 courses, 97% from their own department.
    Course popularity is skewed (Zipf-like), so a few courses are large. Room capacities
    are set relative to the largest course so every exam fits the big hall.
    """
    rng = np.random.default_rng(seed)
    n_depts = max(1, n_courses // courses_per_dept)
    n_students = students_per_course * n_courses
    dept_of_course = np.arange(n_courses) % n_depts
    popularity = 1.0 / np.arange(1, n_courses + 1) ** 0.8
    popularity = popularity[rng.permutation(n_courses)]
    enrol = defaultdict(set)
    for s in range(n_students):
        d = rng.integers(n_depts)
        k = rng.integers(3, 6)
        own = np.flatnonzero(dept_of_course == d)
        other = np.flatnonzero(dept_of_course != d)
        chosen = set()
        while len(chosen) < k:
            pool = own if (rng.random() < 0.97 or len(other) == 0) else other
            p = popularity[pool] / popularity[pool].sum()
            chosen.add(int(rng.choice(pool, p=p)))
        for c in chosen:
            enrol[c].add(s)
    enrolment = [sorted(enrol[c]) for c in range(n_courses)]
    biggest = max(len(e) for e in enrolment)
    rooms = [biggest, int(0.6 * biggest), int(0.4 * biggest), int(0.25 * biggest)]
    return {"n_courses": n_courses, "n_students": n_students, "enrolment": enrolment,
            "rooms": rooms, "seed": seed}


def conflicts(inst):
    """Shared-student counts for every pair of courses that conflict: {(c, d): n_shared}, c < d."""
    sets = [set(e) for e in inst["enrolment"]]
    out = {}
    for c, d in combinations(range(inst["n_courses"]), 2):
        n = len(sets[c] & sets[d])
        if n:
            out[(c, d)] = n
    return out


def same_day_next(t: int) -> bool:
    """True if slot t+1 is on the same day as slot t."""
    return t % SLOTS_PER_DAY != SLOTS_PER_DAY - 1


# ---------------------------------------------------------------- checker (independent of the solvers)
def check(inst, timetable):
    """Validate the hard constraints from the raw enrolment lists and recompute the metrics.

    Does not use conflicts() or any solver code, so a bug there cannot hide here.
    Returns a dict with the violation counts and the metrics.
    """
    n, rooms, enrolment = inst["n_courses"], inst["rooms"], inst["enrolment"]
    v = {"unscheduled": 0, "student_clash": 0, "room_capacity": 0, "room_double_booked": 0}
    v["unscheduled"] = sum(1 for c in range(n) if c not in timetable)
    used_room = defaultdict(int)
    for c, (t, r) in timetable.items():
        if not (0 <= r < len(rooms)) or len(enrolment[c]) > rooms[r]:
            v["room_capacity"] += 1
        used_room[(t, r)] += 1
    v["room_double_booked"] = sum(k - 1 for k in used_room.values() if k > 1)
    slots_of_student = defaultdict(list)
    for c, (t, _) in timetable.items():
        for s in enrolment[c]:
            slots_of_student[s].append(t)
    b2b = 0
    for s, ts in slots_of_student.items():
        v["student_clash"] += len(ts) - len(set(ts))
        tset = set(ts)
        b2b += sum(1 for t in tset if same_day_next(t) and t + 1 in tset)
    period = (max(t for t, _ in timetable.values()) + 1) if timetable else 0
    return {**v, "violations": sum(v.values()), "period_slots": period, "back_to_back": b2b,
            "objective": SLOT_WEIGHT * period + b2b}


# ---------------------------------------------------------------- heuristics
def _room_for(inst, c, taken):
    """Smallest free room that fits course c in a slot whose rooms `taken` are busy, or None."""
    size = len(inst["enrolment"][c])
    fits = [r for r, cap in enumerate(inst["rooms"]) if cap >= size and r not in taken]
    return min(fits, key=lambda r: inst["rooms"][r]) if fits else None


def _b2b_added(slot, c, slot_courses, conf):
    """Back-to-backs created by putting course c into `slot` (approx: shared-student counts with neighbours)."""
    added = 0
    for nb in (slot - 1, slot + 1):
        if nb < 0:
            continue
        lo = min(slot, nb)
        if not same_day_next(lo):
            continue
        for d in slot_courses.get(nb, ()):
            added += conf.get((min(c, d), max(c, d)), 0)
    return added


def first_fit(inst, order=None):
    """Baseline: take courses in the given order (default: course id, like a registry list) and put
    each into the earliest slot with no student clash and a free room that fits."""
    conf = conflicts(inst)
    nbrs = defaultdict(set)
    for c, d in conf:
        nbrs[c].add(d)
        nbrs[d].add(c)
    order = list(range(inst["n_courses"])) if order is None else order
    tt, slot_courses, slot_rooms = {}, defaultdict(set), defaultdict(set)
    for c in order:
        t = 0
        while True:
            if not (nbrs[c] & slot_courses[t]):
                r = _room_for(inst, c, slot_rooms[t])
                if r is not None:
                    break
            t += 1
        tt[c] = (t, r)
        slot_courses[t].add(c)
        slot_rooms[t].add(r)
    return tt


def dsatur(inst, b2b_aware: bool = False):
    """DSATUR graph colouring with rooms.

    Repeatedly pick the unscheduled course whose conflicting courses already occupy the most distinct
    slots (saturation); break ties by number of conflicts, then by enrolment. Place it in the lowest
    feasible slot (plain), or, if b2b_aware, in the feasible already-open slot that adds the fewest
    back-to-backs (a new slot is opened only when no open slot is feasible).
    """
    conf = conflicts(inst)
    nbrs = defaultdict(set)
    for c, d in conf:
        nbrs[c].add(d)
        nbrs[d].add(c)
    size = [len(e) for e in inst["enrolment"]]
    tt, slot_courses, slot_rooms = {}, defaultdict(set), defaultdict(set)
    todo = set(range(inst["n_courses"]))
    while todo:
        def key(c):
            sat = len({tt[d][0] for d in nbrs[c] if d in tt})
            return (sat, len(nbrs[c]), size[c], -c)
        c = max(todo, key=key)
        n_open = (max(t for t, _ in tt.values()) + 1) if tt else 0
        feasible = []
        for t in range(n_open + 1):
            if nbrs[c] & slot_courses[t]:
                continue
            r = _room_for(inst, c, slot_rooms[t])
            if r is not None:
                feasible.append((t, r))
        if not feasible:  # every open slot is blocked: open further slots until one fits
            t = n_open + 1
            while _room_for(inst, c, slot_rooms[t]) is None:
                t += 1
            feasible = [(t, _room_for(inst, c, slot_rooms[t]))]
        if b2b_aware:
            in_open = [f for f in feasible if f[0] < n_open] or feasible
            t, r = min(in_open, key=lambda f: (_b2b_added(f[0], c, slot_courses, conf), f[0]))
        else:
            t, r = feasible[0]
        tt[c] = (t, r)
        slot_courses[t].add(c)
        slot_rooms[t].add(r)
        todo.remove(c)
    return tt


def local_search(inst, timetable, max_rounds: int = 50):
    """Reduce back-to-backs without lengthening the period (first-improvement hill climbing).

    Moves: (1) swap the contents of two whole slots (always keeps clashes and rooms valid);
    (2) move one course to another slot inside the period where it has no clash and a free room.
    Every accepted move is re-scored with the independent checker.
    """
    tt = dict(timetable)
    period = check(inst, tt)["period_slots"]
    best = check(inst, tt)["back_to_back"]
    conf = conflicts(inst)
    nbrs = defaultdict(set)
    for c, d in conf:
        nbrs[c].add(d)
        nbrs[d].add(c)
    for _ in range(max_rounds):
        improved = False
        for a, b in combinations(range(period), 2):          # slot swaps
            cand = {c: ((b if t == a else a if t == b else t), r) for c, (t, r) in tt.items()}
            score = check(inst, cand)["back_to_back"]
            if score < best:
                tt, best, improved = cand, score, True
        for c in range(inst["n_courses"]):                    # single-course moves
            t0 = tt[c][0]
            for t in range(period):
                if t == t0 or any(tt[d][0] == t for d in nbrs[c]):
                    continue
                taken = {r for d, (td, r) in tt.items() if td == t}
                r = _room_for(inst, c, taken)
                if r is None:
                    continue
                cand = dict(tt)
                cand[c] = (t, r)
                res = check(inst, cand)
                if res["violations"] == 0 and res["period_slots"] <= period and res["back_to_back"] < best:
                    tt, best, improved = cand, res["back_to_back"], True
                    t0 = t
        if not improved:
            break
    return tt


# ---------------------------------------------------------------- exact
def ilp(inst, n_slots: int, time_limit: float = 60.0, warm=None):
    """ILP (PuLP/CBC).

    x[c,t,r] = 1 if course c sits in slot t, room r (only rooms that fit c)
    y[t]     = 1 if slot t is inside the exam period (y[t] >= y[t+1], so period = sum_t y[t])
    b[c,d,t] = 1 if conflicting courses c, d sit in consecutive same-day slots t, t+1 (either order)
    min  SLOT_WEIGHT * sum_t y[t] + sum_{c<d,t} shared(c,d) * b[c,d,t]
    s.t. each course exactly once; each room at most one exam per slot; conflicting courses not in
         the same slot; x <= y; b[c,d,t] >= z[c,t] + z[d,t+1] - 1 and >= z[d,t] + z[c,t+1] - 1
    where z[c,t] = sum_r x[c,t,r].
    The back-to-back term sums shared students over pairs. A student sits at most one exam per slot,
    so each (student, t) back-to-back belongs to exactly one pair: the two counts are equal (tested).
    """
    conf = conflicts(inst)
    N, R, T = inst["n_courses"], inst["rooms"], range(n_slots)
    size = [len(e) for e in inst["enrolment"]]
    fits = {c: [r for r in range(len(R)) if R[r] >= size[c]] for c in range(N)}
    m = pulp.LpProblem("exam_timetabling", pulp.LpMinimize)
    x = {(c, t, r): pulp.LpVariable(f"x_{c}_{t}_{r}", cat="Binary")
         for c in range(N) for t in T for r in fits[c]}
    y = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in T}
    z = {(c, t): pulp.lpSum(x[c, t, r] for r in fits[c]) for c in range(N) for t in T}
    pairs_t = [(c, d, t) for (c, d) in conf for t in T if t + 1 < n_slots and same_day_next(t)]
    b = {k: pulp.LpVariable("b_%d_%d_%d" % k, lowBound=0) for k in pairs_t}
    m += SLOT_WEIGHT * pulp.lpSum(y.values()) + pulp.lpSum(conf[c, d] * b[c, d, t] for c, d, t in pairs_t)
    for c in range(N):
        m += pulp.lpSum(z[c, t] for t in T) == 1
        for t in T:
            m += z[c, t] <= y[t]
    for t in T:
        if t + 1 < n_slots:
            m += y[t] >= y[t + 1]
        for r in range(len(R)):
            m += pulp.lpSum(x[c, t, r] for c in range(N) if (c, t, r) in x) <= 1
        for c, d in conf:
            m += z[c, t] + z[d, t] <= 1
    for c, d, t in pairs_t:
        m += b[c, d, t] >= z[c, t] + z[d, t + 1] - 1
        m += b[c, d, t] >= z[d, t] + z[c, t + 1] - 1
    if warm is not None:
        for k, var in x.items():
            var.setInitialValue(1 if warm.get(k[0]) == (k[1], k[2]) else 0)
        used = {t for t, _ in warm.values()}
        for t in T:
            y[t].setInitialValue(1 if t <= max(used) else 0)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, warmStart=warm is not None)
    m.solve(solver)
    tt = {}
    for (c, t, r), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            tt[c] = (t, r)
    # CBC returns status "Optimal" even when it stops at the time limit with a feasible solution;
    # sol_status tells them apart (1 = proven optimal, 2 = feasible, not proven).
    proven = m.sol_status == pulp.LpSolutionOptimal
    return tt, {"status": "optimal" if proven else pulp.LpSolution[m.sol_status],
                "n_vars": len(m.variables()), "n_cons": len(m.constraints),
                "model_objective": pulp.value(m.objective)}


def clique_lower_bound(inst):
    """Lower bound on the period: the max of
    - the largest clique of conflicting courses (they all need different slots),
    - a room bound: courses that only fit the k largest rooms can use at most k of them per slot,
      so they need at least ceil(count_k / k) slots (k = 1 .. rooms; k = rooms gives ceil(N / rooms)).
    Returns (bound, clique size, room bound)."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(inst["n_courses"]))
    g.add_edges_from(conflicts(inst))
    clique = max(len(q) for q in nx.find_cliques(g))
    caps = sorted(inst["rooms"], reverse=True)
    sizes = [len(e) for e in inst["enrolment"]]
    room_lb = 0
    for k in range(1, len(caps) + 1):
        limit = caps[k] if k < len(caps) else 0       # courses bigger than the (k+1)-th room
        count = sum(1 for s in sizes if s > limit)
        room_lb = max(room_lb, -(-count // k))
    return max(clique, room_lb), clique, room_lb
