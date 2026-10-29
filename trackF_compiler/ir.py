"""Three-address intermediate representation (IR): code generation from the AST, basic blocks / CFG,
liveness analysis, and an interpreter that counts executed instructions.

Instruction forms (operands are ints or variable names; compiler temporaries start with '%'):
  mov d a            d = a
  add/sub/mul/div/mod/shl/lt/le/gt/ge/eq/ne d a b      d = a op b
  neg d a            d = -a
  print a
  label L
  jmp L
  jz a L             if a == 0: goto L
"""
from dataclasses import dataclass, field

BINOPS = {"add", "sub", "mul", "div", "mod", "shl", "lt", "le", "gt", "ge", "eq", "ne"}
PURE = BINOPS | {"mov", "neg"}  # instructions whose only effect is writing dst
# Rough cycle costs used for a 'weighted cost' metric (multiply and divide are slower than add/shift)
COST = {"mul": 3, "div": 10, "mod": 10}


@dataclass
class Instr:
    op: str
    dst: object = None
    a: object = None
    b: object = None
    label: str = None

    def uses(self):
        return [x for x in (self.a, self.b) if isinstance(x, str)]

    def __str__(self):
        if self.op == "label":
            return f"{self.label}:"
        if self.op == "jmp":
            return f"    jmp {self.label}"
        if self.op == "jz":
            return f"    jz {self.a}, {self.label}"
        if self.op == "print":
            return f"    print {self.a}"
        args = ", ".join(str(x) for x in (self.a, self.b) if x is not None)
        return f"    {self.dst} = {self.op} {args}"


def trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def evaluate(op, a, b=None):
    """Semantics of every arithmetic op (C-like truncating division). Used by the interpreter AND by constant
    folding, so the optimizer and the 'hardware' always agree."""
    if op == "mov":
        return a
    if op == "neg":
        return -a
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return trunc_div(a, b)
    if op == "mod":
        return a - b * trunc_div(a, b)
    if op == "shl":
        return a << b
    return int({"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b, "ne": a != b}[op])


# ---------------- code generation ----------------

@dataclass
class CodeGen:
    code: list = field(default_factory=list)
    ntemp: int = 0
    nlabel: int = 0

    def temp(self):
        self.ntemp += 1
        return f"%t{self.ntemp}"

    def new_label(self, kind):
        self.nlabel += 1
        return f"L{kind}_{self.nlabel}"

    def expr(self, e):
        """Returns the operand (int or name) holding the value of e."""
        kind = e[0]
        if kind == "num":
            return e[1]
        if kind == "var":
            return e[1]
        if kind == "neg":
            a = self.expr(e[1])
            t = self.temp()
            self.code.append(Instr("neg", t, a))
            return t
        _, op, l, r = e
        a, b = self.expr(l), self.expr(r)
        t = self.temp()
        self.code.append(Instr(op, t, a, b))
        return t

    def stmt(self, s):
        kind = s[0]
        if kind == "assign":
            self.code.append(Instr("mov", s[1], self.expr(s[2])))
        elif kind == "print":
            self.code.append(Instr("print", a=self.expr(s[1])))
        elif kind == "if":
            l_else, l_end = self.new_label("else"), self.new_label("endif")
            self.code.append(Instr("jz", a=self.expr(s[1]), label=l_else))
            for x in s[2]:
                self.stmt(x)
            self.code.append(Instr("jmp", label=l_end))
            self.code.append(Instr("label", label=l_else))
            for x in s[3]:
                self.stmt(x)
            self.code.append(Instr("label", label=l_end))
        elif kind == "while":
            n = self.nlabel = self.nlabel + 1
            head, end = f"Lwhile_{n}", f"Lendwhile_{n}"  # paired names let LICM find loops
            self.code.append(Instr("label", label=head))
            self.code.append(Instr("jz", a=self.expr(s[1]), label=end))
            for x in s[2]:
                self.stmt(x)
            self.code.append(Instr("jmp", label=head))
            self.code.append(Instr("label", label=end))


def generate(ast):
    g = CodeGen()
    for s in ast:
        g.stmt(s)
    return g.code


# ---------------- CFG and liveness ----------------

def basic_blocks(code):
    """Split into maximal straight-line blocks. A block starts at index 0, at a label, or after a jump.
    Returns list of (start, end) index ranges and successor lists."""
    leaders = {0}
    for i, ins in enumerate(code):
        if ins.op == "label":
            leaders.add(i)
        if ins.op in ("jmp", "jz") and i + 1 < len(code):
            leaders.add(i + 1)
    starts = sorted(x for x in leaders if x < len(code))
    blocks = [(s, (starts[k + 1] if k + 1 < len(starts) else len(code))) for k, s in enumerate(starts)]
    label_block = {code[s].label: b for b, (s, e) in enumerate(blocks) if code[s].op == "label"}
    succ = []
    for b, (s, e) in enumerate(blocks):
        last = code[e - 1]
        nxt = [b + 1] if b + 1 < len(blocks) else []
        if last.op == "jmp":
            succ.append([label_block[last.label]])
        elif last.op == "jz":
            succ.append(sorted(set(nxt + [label_block[last.label]])))
        else:
            succ.append(nxt)
    return blocks, succ


def liveness(code):
    """Backward dataflow: live_out[i] = variables whose current value may be read after instruction i."""
    blocks, succ = basic_blocks(code)
    live_in_b = [set() for _ in blocks]
    changed = True
    while changed:
        changed = False
        for b in reversed(range(len(blocks))):
            live = set().union(*[live_in_b[s] for s in succ[b]]) if succ[b] else set()
            s, e = blocks[b]
            for i in range(e - 1, s - 1, -1):
                if code[i].dst is not None:
                    live.discard(code[i].dst)
                live |= set(code[i].uses())
            if live != live_in_b[b]:
                live_in_b[b], changed = live, True
    live_out = [set() for _ in code]
    for b, (s, e) in enumerate(blocks):
        live = set().union(*[live_in_b[x] for x in succ[b]]) if succ[b] else set()
        for i in range(e - 1, s - 1, -1):
            live_out[i] = set(live)
            if code[i].dst is not None:
                live.discard(code[i].dst)
            live |= set(code[i].uses())
    return live_out


# ---------------- interpreter ----------------

def run(code, max_steps=10_000_000):
    """Execute the IR. Returns (printed values, executed instruction count, weighted cost).
    Labels are not counted (they generate no machine code)."""
    labels = {ins.label: i for i, ins in enumerate(code) if ins.op == "label"}
    env, out = {}, []
    pc = steps = cost = 0

    def val(x):
        return x if isinstance(x, int) else env.get(x, 0)  # variables start at 0

    while pc < len(code):
        ins = code[pc]
        pc += 1
        if ins.op == "label":
            continue
        steps += 1
        cost += COST.get(ins.op, 1)
        if steps > max_steps:
            raise RuntimeError("step limit exceeded")
        if ins.op == "jmp":
            pc = labels[ins.label]
        elif ins.op == "jz":
            if val(ins.a) == 0:
                pc = labels[ins.label]
        elif ins.op == "print":
            out.append(val(ins.a))
        else:
            env[ins.dst] = evaluate(ins.op, val(ins.a), None if ins.b is None else val(ins.b))
    return out, steps, cost


def static_count(code):
    return sum(1 for ins in code if ins.op != "label")
