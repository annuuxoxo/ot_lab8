"""Optimization passes on the IR. Each pass takes a list of Instr and returns a new list (inputs are not modified).

  const_fold  constant folding + constant/copy propagation + algebraic identities (local, per basic block)
  cse         common sub-expression elimination (local, per basic block)
  strength    strength reduction: x * 2^k -> x shl k
  dce         dead-code elimination: unreachable code, jumps to the next line, and assignments whose result is never
              read (global liveness over the CFG)
  licm        loop-invariant code motion: move computations whose operands do not change inside a loop to just
              before the loop
  coalesce    copy coalescing: '%t = op a b ; x = mov %t' -> 'x = op a b' when %t is not used again
"""
from dataclasses import replace

from ir import BINOPS, PURE, Instr, evaluate, liveness

COMMUTATIVE = {"add", "mul", "eq", "ne"}


def _block_boundary(ins):
    return ins.op in ("label", "jmp", "jz")


def const_fold(code):
    out = []
    consts, copies = {}, {}  # var -> int, var -> var  (valid until the end of the basic block)

    def sub(x):
        if isinstance(x, str):
            x = copies.get(x, x)
            return consts.get(x, x)
        return x

    def kill(v):
        consts.pop(v, None)
        copies.pop(v, None)
        for k in [k for k, src in copies.items() if src == v]:  # copies OF v are stale once v changes
            del copies[k]

    for ins in code:
        if ins.op == "label":
            consts.clear()
            copies.clear()
            out.append(ins)
            continue
        ins = replace(ins, a=sub(ins.a), b=sub(ins.b))
        if ins.op == "jz" and isinstance(ins.a, int):
            if ins.a == 0:
                out.append(Instr("jmp", label=ins.label))  # always taken
            # else: never taken -> drop it
            consts.clear()
            copies.clear()
            continue
        if ins.op in BINOPS:
            a, b = ins.a, ins.b
            if isinstance(a, int) and isinstance(b, int) and not (ins.op in ("div", "mod") and b == 0):
                ins = Instr("mov", ins.dst, evaluate(ins.op, a, b))
            # algebraic identities (never remove a possible division by zero)
            elif ins.op in ("add", "sub") and b == 0 or ins.op in ("mul", "div") and b == 1 or ins.op == "shl" and b == 0:
                ins = Instr("mov", ins.dst, a)
            elif ins.op in ("add", "mul") and a == (0 if ins.op == "add" else 1):
                ins = Instr("mov", ins.dst, b)
            elif ins.op == "mul" and (a == 0 or b == 0):
                ins = Instr("mov", ins.dst, 0)
        elif ins.op == "neg" and isinstance(ins.a, int):
            ins = Instr("mov", ins.dst, -ins.a)
        if ins.dst is not None:
            kill(ins.dst)
            if ins.op == "mov":
                if isinstance(ins.a, int):
                    consts[ins.dst] = ins.a
                elif ins.a != ins.dst:
                    copies[ins.dst] = ins.a
        out.append(ins)
        if _block_boundary(ins):
            consts.clear()
            copies.clear()
    return out


def cse(code):
    out = []
    avail = {}  # (op, a, b) -> variable that currently holds that value

    for ins in code:
        if _block_boundary(ins):
            avail.clear()
            out.append(ins)
            continue
        avail_key = None
        if ins.op in BINOPS or ins.op == "neg":
            a, b = ins.a, ins.b
            if ins.op in COMMUTATIVE and str(a) > str(b):
                a, b = b, a
            key = (ins.op, a, b)
            if key in avail:
                ins = Instr("mov", ins.dst, avail[key])  # reuse the earlier result
            else:
                avail_key = key
        if ins.dst is not None:
            # dst changes: forget expressions that read it or were stored in it
            for k in [k for k, v in avail.items() if v == ins.dst or ins.dst in (k[1], k[2])]:
                del avail[k]
            if ins.op != "mov" and avail_key is not None and ins.dst not in (avail_key[1], avail_key[2]):
                avail[avail_key] = ins.dst
        out.append(ins)
    return out


def strength(code):
    out = []
    for ins in code:
        if ins.op == "mul":
            for x, y in ((ins.a, ins.b), (ins.b, ins.a)):
                if isinstance(y, int) and y > 1 and y & (y - 1) == 0:
                    ins = Instr("shl", ins.dst, x, y.bit_length() - 1)
                    break
        out.append(ins)
    return out


def dce(code):
    # 1. unreachable code: anything after an unconditional jump up to the next label
    out, dead = [], False
    for ins in code:
        if ins.op == "label":
            dead = False
        if not dead:
            out.append(ins)
        if ins.op == "jmp":
            dead = True
    # 2. a jump to the label that immediately follows it does nothing
    code = [ins for i, ins in enumerate(out)
            if not (ins.op == "jmp" and i + 1 < len(out) and out[i + 1].op == "label" and out[i + 1].label == ins.label)]
    # 3. remove labels nobody jumps to (they are free, but this makes later blocks larger for local passes)
    targets = {ins.label for ins in code if ins.op in ("jmp", "jz")}
    code = [ins for ins in code if ins.op != "label" or ins.label in targets]
    # 4. dead assignments: pure instruction whose result is not live afterwards (repeat until stable)
    while True:
        live_out = liveness(code)
        keep = [ins for i, ins in enumerate(code)
                if not (ins.op in PURE and ins.dst not in live_out[i]
                        and not (ins.op in ("div", "mod") and ins.b == 0))]
        if len(keep) == len(code):
            return keep
        code = keep


def _loops(code):
    """(head index, end index) for every while loop, found by the paired label names from codegen."""
    pos = {ins.label: i for i, ins in enumerate(code) if ins.op == "label"}
    return [(pos[l], pos["Lendwhile_" + l.split("_")[1]]) for l in pos if l.startswith("Lwhile_")
            and "Lendwhile_" + l.split("_")[1] in pos]


def licm(code):
    """Hoist 'dst = op a b' out of a loop when: op is pure and cannot fault, dst is a compiler temporary assigned
    exactly once in the loop, and no operand is assigned anywhere in the loop. Repeat until nothing moves
    (hoisting from an inner loop can make the instruction invariant in the outer loop too)."""
    code = list(code)
    moved = True
    while moved:
        moved = False
        for head, end in sorted(_loops(code), key=lambda he: he[1] - he[0]):  # innermost first
            body = code[head + 1:end]
            defs = [ins.dst for ins in body if ins.dst is not None]
            for k, ins in enumerate(body):
                safe_op = ins.op in BINOPS | {"neg"} and not (ins.op in ("div", "mod") and not (isinstance(ins.b, int) and ins.b != 0))
                if (safe_op and isinstance(ins.dst, str) and ins.dst.startswith("%") and defs.count(ins.dst) == 1
                        and all(u not in defs for u in ins.uses())):
                    del code[head + 1 + k]
                    code.insert(head, ins)  # just before the loop header label
                    moved = True
                    break
            if moved:
                break
    return code


def coalesce(code):
    live_out = liveness(code)
    out, skip = [], False
    for i, ins in enumerate(code):
        if skip:
            skip = False
            continue
        nxt = code[i + 1] if i + 1 < len(code) else None
        if (ins.op in PURE and isinstance(ins.dst, str) and ins.dst.startswith("%") and nxt is not None
                and nxt.op == "mov" and nxt.a == ins.dst and ins.dst not in live_out[i + 1]):
            out.append(replace(ins, dst=nxt.dst))  # compute straight into the final variable
            skip = True
        else:
            out.append(ins)
    return out


PASSES = {"const_fold": const_fold, "cse": cse, "strength": strength, "dce": dce, "licm": licm, "coalesce": coalesce}
# Default order = best of all 720 orderings found by exhaustive search (f_experiments.pass_ordering)
ALL = ("cse", "strength", "dce", "licm", "coalesce", "const_fold")


def optimize(code, passes=ALL, max_rounds=20):
    """Apply the selected passes in order, repeatedly, until the code stops changing."""
    for _ in range(max_rounds):
        before = [str(i) for i in code]
        for p in passes:
            code = PASSES[p](code)
        if [str(i) for i in code] == before:
            break
    return code
