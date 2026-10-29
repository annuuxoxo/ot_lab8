"""Random program generator for differential testing: optimized and unoptimized code must print the same values.
Loops always terminate (a private counter that the body never assigns); divisors are always >= 1."""
import random

VARS = ["a", "b", "c", "d"]


def expr(r, depth):
    if depth == 0 or r.random() < 0.3:
        return str(r.randint(0, 9)) if r.random() < 0.4 else r.choice(VARS)
    op = r.choice(["+", "-", "*", "/", "%", "<", "==", "!=", "-u"])
    if op == "-u":
        return f"(-{expr(r, depth - 1)})"
    if op in "/%":
        d = expr(r, depth - 1)
        return f"({expr(r, depth - 1)} {op} ({d} * {d} + 1))"  # divisor >= 1, never zero
    return f"({expr(r, depth - 1)} {op} {expr(r, depth - 1)})"


def block(r, depth, counter):
    lines = []
    for _ in range(r.randint(1, 4)):
        k = r.random()
        if k < 0.5 or depth == 0:
            lines.append(f"{r.choice(VARS)} = {expr(r, 2)};")
        elif k < 0.65:
            lines.append(f"print({expr(r, 2)});")
        elif k < 0.85:
            lines.append(f"if ({expr(r, 2)}) {{ {block(r, depth - 1, counter)} }} else {{ {block(r, depth - 1, counter)} }}")
        else:
            c = f"loop{counter[0]}"
            counter[0] += 1
            lines.append(f"{c} = 0; while ({c} < {r.randint(0, 4)}) {{ {block(r, depth - 1, counter)} {c} = {c} + 1; }}")
    return " ".join(lines)


def program(seed):
    r = random.Random(seed)
    init = " ".join(f"{v} = {r.randint(-5, 9)};" for v in VARS)
    body = block(r, 2, [0])
    return f"{init} {body} " + " ".join(f"print({v});" for v in VARS)
