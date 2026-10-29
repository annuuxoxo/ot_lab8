import glob
import os
import sys

import pytest

HERE = os.path.join(os.path.dirname(__file__), "..", "trackF_compiler")
sys.path.insert(0, HERE)
import frontend  # noqa: E402
import fuzz  # noqa: E402
import ir  # noqa: E402
import passes  # noqa: E402
import regalloc  # noqa: E402


def compile_src(src):
    return ir.generate(frontend.parse(src))


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(HERE, "examples", "*.toy"))), ids=os.path.basename)
def test_optimized_example_prints_same_output(path):
    code = compile_src(open(path).read())
    assert ir.run(code)[0] == ir.run(passes.optimize(code))[0]


def test_known_outputs():
    assert ir.run(compile_src("print(7 / 2); print(-7 / 2); print(-7 % 3); print(2 * (3 + 4));"))[0] == [3, -3, -1, 14]
    fact = ir.run(compile_src(open(os.path.join(HERE, "examples", "factorial.toy")).read()))[0]
    assert fact[-1] == 479001600  # 12!


def test_differential_fuzzing_500_random_programs():
    for s in range(500):
        code = compile_src(fuzz.program(s))
        expected = ir.run(code)[0]
        assert ir.run(passes.optimize(code))[0] == expected, s
        for p in passes.ALL:
            assert ir.run(passes.optimize(code, (p,)))[0] == expected, (s, p)


def test_individual_passes_do_their_job():
    code = compile_src("x = 2 * 3 + 4; print(x);")
    assert [str(i).strip() for i in passes.optimize(code)] == ["print 10"]  # folded, propagated, dead x removed
    code = compile_src("a = 5; b = 3; y = a * b; z = a * b; print(y + z);")
    assert sum(i.op == "mul" for i in passes.cse(code)) == 1
    assert passes.strength(compile_src("y = x * 8;"))[0].op == "shl"
    loop = compile_src("n = 9; i = 0; while (i < 3) { k = n * n; i = i + 1; } print(i);")
    opt = passes.licm(loop)
    head = next(j for j, ins in enumerate(opt) if ins.op == "label")
    assert any(ins.op == "mul" for ins in opt[:head])  # n*n moved before the loop


def test_register_allocation_is_valid_and_ilp_never_worse():
    code = passes.optimize(compile_src(open(os.path.join(HERE, "examples", "nested_invariant.toy")).read()))
    for K in (2, 3, 4):
        r = regalloc.allocate(code, K)  # allocate() asserts both colourings are valid
        assert r["ilp_spill_cost"] <= r["greedy_spill_cost"] + 1e-9
