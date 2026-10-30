"""Track E: the checker catches violations, every method is feasible, and the ILP is exact on a tiny case."""
import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trackE_case_study"))
import timetabling as tt  # noqa: E402


@pytest.fixture(scope="module")
def inst():
    return tt.make_instance(20, 0)


def test_instance_is_seeded():
    assert tt.make_instance(20, 3)["enrolment"] == tt.make_instance(20, 3)["enrolment"]
    assert tt.make_instance(20, 3)["enrolment"] != tt.make_instance(20, 4)["enrolment"]


def test_checker_detects_each_violation(inst):
    good = tt.first_fit(inst)
    assert tt.check(inst, good)["violations"] == 0
    c, d = next(iter(tt.conflicts(inst)))            # two courses sharing a student
    bad = dict(good)
    bad[d] = (good[c][0], (good[c][1] + 1) % len(inst["rooms"]))
    assert tt.check(inst, bad)["student_clash"] > 0
    bad = dict(good)
    big = max(range(inst["n_courses"]), key=lambda k: len(inst["enrolment"][k]))
    bad[big] = (good[big][0], len(inst["rooms"]) - 1)  # smallest room
    assert tt.check(inst, bad)["room_capacity"] == 1
    bad = dict(good)
    del bad[0]
    assert tt.check(inst, bad)["unscheduled"] == 1
    bad = {k: (0, 0) for k in range(inst["n_courses"])}
    assert tt.check(inst, bad)["room_double_booked"] == inst["n_courses"] - 1


def test_back_to_back_definition():
    inst = {"n_courses": 3, "enrolment": [[0], [0], [0]], "rooms": [5]}
    # slots 0,1 same day -> 1 back-to-back; slot 2 -> 3 is a new day (SLOTS_PER_DAY = 3)
    assert tt.check(inst, {0: (0, 0), 1: (1, 0), 2: (3, 0)})["back_to_back"] == 1
    assert tt.check(inst, {0: (2, 0), 1: (3, 0), 2: (5, 0)})["back_to_back"] == 0


@pytest.mark.parametrize("n,seed", [(20, 0), (20, 1), (40, 2), (80, 0)])
def test_heuristics_feasible(n, seed):
    inst = tt.make_instance(n, seed)
    lb = tt.clique_lower_bound(inst)[0]
    for sol in (tt.first_fit(inst), tt.dsatur(inst), tt.dsatur(inst, b2b_aware=True)):
        res = tt.check(inst, sol)
        assert res["violations"] == 0
        assert res["period_slots"] >= lb


def test_local_search_never_worse(inst):
    start = tt.dsatur(inst, b2b_aware=True)
    before, after = tt.check(inst, start), tt.check(inst, tt.local_search(inst, start))
    assert after["violations"] == 0
    assert after["period_slots"] <= before["period_slots"]
    assert after["back_to_back"] <= before["back_to_back"]


def _brute_force(inst, n_slots):
    """Best objective over all slot assignments; rooms assigned by sorted matching (they don't affect cost)."""
    best = None
    rooms = sorted(range(len(inst["rooms"])), key=lambda r: -inst["rooms"][r])
    for slots in itertools.product(range(n_slots), repeat=inst["n_courses"]):
        sol, ok = {}, True
        for t in set(slots):
            cs = sorted([c for c in range(inst["n_courses"]) if slots[c] == t],
                        key=lambda c: -len(inst["enrolment"][c]))
            if len(cs) > len(rooms):
                ok = False
                break
            for c, r in zip(cs, rooms):
                sol[c] = (t, r)
        if not ok:
            continue
        res = tt.check(inst, sol)
        if res["violations"] == 0 and (best is None or res["objective"] < best):
            best = res["objective"]
    return best


def test_ilp_matches_brute_force():
    for seed in (0, 1):
        inst = tt.make_instance(5, seed, students_per_course=6)
        inst["rooms"] = inst["rooms"][:2]      # 2 rooms so rooms matter
        sol, info = tt.ilp(inst, n_slots=5, time_limit=30)
        res = tt.check(inst, sol)
        assert info["status"] == "optimal"
        assert res["violations"] == 0
        assert res["objective"] == _brute_force(inst, 5)
        assert abs(info["model_objective"] - res["objective"]) < 1e-6   # model's b2b count = checker's
