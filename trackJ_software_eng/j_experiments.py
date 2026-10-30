"""Track J: test-suite minimization on python-tabulate (greedy vs ILP vs random vs full suite).

Steps
1. clone tabulate at a pinned commit into trackJ_software_eng/target/ (gitignored)
2. profile:
   - per-test time: median of 3 full-suite runs
   - per-test line coverage: every test run ALONE under coverage.py, with subprocess measurement on
     (the CLI tests run tabulate in a child process, which in-process coverage cannot see)
   - one-time import costs: which heavy modules (numpy, pandas, ...) each test needs, and what
     importing them costs; in a single run that cost lands on whichever test imports them first
3. solve the set cover with greedy and ILP (with shared import costs), plus the naive ILP that
   uses raw per-test times, and random subsets of the same size
4. run each reduced suite for real: wall time (5 interleaved repeats) and line coverage retained
5. mutation check: 40 seeded operator mutants of tabulate/__init__.py, killed by which suites?
"""
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common.utils import RESULTS, save_plot, save_results  # noqa: E402
import minimize  # noqa: E402

TRACK = "trackJ_software_eng"
REPO = "https://github.com/astanin/python-tabulate.git"
COMMIT = "268615a5c27dc40e5c22454c07b44d5c50410da0"
TARGET = os.path.join(HERE, "target")
WORK = os.path.join(RESULTS, TRACK, "work")
MODULE = os.path.join("tabulate", "__init__.py")
N_MUTANTS = 40
TIMING_REPEATS = 3
WALL_REPEATS = 5
HEAVY_MS = 2.0          # modules whose import costs more than this are modelled as shared setup


def ensure_target():
    if not os.path.isdir(TARGET):
        subprocess.run(["git", "clone", "-q", REPO, TARGET], check=True)
    subprocess.run(["git", "-C", TARGET, "checkout", "-q", COMMIT], check=True)
    subprocess.run(["git", "-C", TARGET, "checkout", "-q", "--", "."], check=True)  # undo any leftover mutant


def _env(select=None, times=None):
    env = dict(os.environ, PYTHONPATH=HERE, PYTHONDONTWRITEBYTECODE="1")
    for key in ("TJ_SELECT", "TJ_TIMES", "TJ_COVER"):
        env.pop(key, None)
    if select:
        env["TJ_SELECT"] = select
    if times:
        env["TJ_TIMES"] = times
    return env


PYTEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "tj_plugin", "test"]


def run_pytest(select=None, times=None, stop_first=False, timeout=300):
    """Run tabulate's suite in a subprocess; return (exit code, wall seconds)."""
    cmd = [sys.executable] + PYTEST + (["-x"] if stop_first else [])
    t0 = time.perf_counter()
    try:
        code = subprocess.run(cmd, cwd=TARGET, env=_env(select, times), capture_output=True,
                              timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = -1   # a hanging mutant counts as killed
    return code, time.perf_counter() - t0


def run_coverage(select, tag, times=None):
    """Run the selected tests under coverage.py (child processes measured too); return covered lines."""
    import coverage
    data = os.path.join(WORK, "cov", tag)
    os.makedirs(os.path.dirname(data), exist_ok=True)
    rc = os.path.join(os.path.dirname(data), f"rc_{tag}.ini")
    with open(rc, "w") as f:
        f.write(f"[run]\nsource = {os.path.join(TARGET, 'tabulate')}\npatch = subprocess\n"
                f"parallel = true\ndata_file = {data}\n")
    for old in os.listdir(os.path.dirname(data)):
        if old == tag or old.startswith(tag + "."):
            os.remove(os.path.join(os.path.dirname(data), old))
    cmd = [sys.executable, "-m", "coverage", "run", f"--rcfile={rc}"] + PYTEST
    subprocess.run(cmd, cwd=TARGET, env=_env(select, times), capture_output=True, timeout=300)
    cov = coverage.Coverage(data_file=data, config_file=rc)
    cov.combine(keep=False)
    cov.save()
    lines = set()
    d = cov.get_data()
    for f in d.measured_files():
        assert os.path.realpath(f).startswith(os.path.realpath(TARGET)), f"measured a tabulate outside target/: {f}"
        rel = os.path.relpath(os.path.realpath(f), os.path.realpath(TARGET))
        lines |= {(rel, n) for n in d.lines(f)}
    return lines


def write_selection(name, tests):
    path = os.path.join(WORK, f"select_{name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(tests))
    return path


def import_costs(modules, repeats=5):
    """Marginal import time (s) of each module, after pytest and tabulate are loaded, importing the
    cheaper modules first (so e.g. pandas' cost excludes numpy, which pandas imports itself)."""
    def timed(mods):
        code = ("import time, pytest, tabulate\nimport importlib\nout = []\n"
                f"for m in {mods!r}:\n    t = time.perf_counter()\n    try:\n        importlib.import_module(m)\n"
                "    except Exception:\n        pass   # internal names such as cython_runtime\n"
                "    out.append(time.perf_counter() - t)\nprint(out)")
        runs = []
        for _ in range(repeats):
            p = subprocess.run([sys.executable, "-c", code], cwd=TARGET, env=_env(), capture_output=True,
                               text=True, check=True)
            runs.append(eval(p.stdout.strip()))
        return [statistics.median(col) for col in zip(*runs)]
    alone = {m: timed([m])[0] for m in sorted(modules)}
    order = sorted(modules, key=lambda m: alone[m])
    return dict(zip(order, timed(order))), alone


def profile():
    runs = []
    for i in range(TIMING_REPEATS):
        path = os.path.join(WORK, f"times_{i}.json")
        code, wall = run_pytest(times=path)
        runs.append(json.load(open(path)))
        print(f"timing run {i}: exit {code}, {wall:.2f} s", flush=True)
    passed = [t for t in runs[0] if all(r.get(t, {}).get("outcome") == "passed" for r in runs)]
    raw = {t: statistics.median(r[t]["seconds"] for r in runs) for t in passed}
    outcomes = pd.Series({t: runs[0][t]["outcome"] for t in runs[0]}).value_counts().to_dict()

    empty = write_selection("none", [])
    import_only = run_coverage(empty, "import_only")
    cover, needs_all = {}, {}
    for i, t in enumerate(passed):
        sel = write_selection("one", [t])
        tj = os.path.join(WORK, "one_times.json")
        cover[t] = run_coverage(sel, "one", times=tj) - import_only
        needs_all[t] = set(json.load(open(tj))[t]["new_modules"])
        if i % 50 == 0:
            print(f"isolated coverage {i}/{len(passed)}", flush=True)

    imp, alone = import_costs(set().union(*needs_all.values()))
    setup = {m: c for m, c in imp.items() if c * 1000 > HEAVY_MS}
    needs = {t: needs_all[t] & set(setup) for t in passed}
    # base cost = raw time minus the imports this test happened to pay for in the full run
    charged = {t: set(runs[0][t]["new_modules"]) & set(setup) for t in passed}
    base = {t: max(0.0, raw[t] - sum(setup[m] for m in charged[t])) for t in passed}
    modules = pd.DataFrame({"module": list(imp), "marginal_import_ms": [imp[m] * 1000 for m in imp],
                            "standalone_import_ms": [alone[m] * 1000 for m in imp],
                            "shared_setup": [m in setup for m in imp],
                            "tests_needing": [sum(m in needs_all[t] for t in passed) for m in imp]})
    save_results(modules.sort_values("marginal_import_ms", ascending=False), TRACK, "j_import_costs")
    return raw, base, cover, import_only, needs, setup, charged, outcomes


def mutants(cover_lines, n, seed=0):
    """Seeded single-operator mutants on lines of tabulate/__init__.py that the suite executes."""
    ops = [(" == ", " != "), (" != ", " == "), (" < ", " <= "), (" > ", " >= "), (" <= ", " < "),
           (" >= ", " > "), (" + ", " - "), (" - ", " + "), (" and ", " or "), (" or ", " and "),
           ("True", "False"), ("False", "True"), (" not ", " ")]
    src = open(os.path.join(TARGET, MODULE)).read().split("\n")
    cands = []
    for (f, line) in sorted(cover_lines):
        if f != MODULE:
            continue
        text = src[line - 1]
        if text.strip().startswith(("#", '"', "'")):
            continue
        for a, b in ops:
            for m in re.finditer(re.escape(a), text):
                cands.append((line, m.start(), a, b))
    rng = random.Random(seed)
    rng.shuffle(cands)
    out = []
    for line, pos, a, b in cands:
        mutated = list(src)
        mutated[line - 1] = src[line - 1][:pos] + b + src[line - 1][pos + len(a):]
        code = "\n".join(mutated)
        try:
            compile(code, MODULE, "exec")
        except SyntaxError:
            continue
        out.append({"line": line, "from": a.strip(), "to": b.strip() or "(removed)", "code": code})
        if len(out) == n:
            break
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    ensure_target()
    loc = {p: sum(1 for _ in open(os.path.join(TARGET, p))) for p in (MODULE, "tabulate/cli.py")}

    raw, base, cover, import_only, needs, setup, charged, outcomes = profile()
    full = sorted(raw)
    req = minimize.requirements(cover)
    print(f"{len(full)} passing tests, {len(req)} lines covered by tests, {len(import_only)} import-only lines, "
          f"shared setup: { {m: round(c * 1000, 1) for m, c in setup.items()} } ms", flush=True)

    t0 = time.perf_counter()
    greedy = sorted(minimize.greedy(cover, base, needs, setup))
    greedy_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ilp, info = minimize.ilp(cover, base, needs=needs, setup=setup)
    ilp = sorted(ilp)
    ilp_s = time.perf_counter() - t0
    naive, naive_info = minimize.ilp(cover, raw)          # the first model: raw per-test times, no imports
    naive = sorted(naive)
    k = len(ilp)
    randoms = [minimize.random_subset(cover, k, s) for s in range(10)]

    selections = {"full suite": full, "greedy": greedy, "ILP": ilp, "ILP, naive costs": naive,
                  "random (seed 0)": randoms[0]}
    sel_files = {name: (None if name == "full suite" else write_selection(re.sub(r"\W+", "_", name), tests))
                 for name, tests in selections.items()}

    walls = {name: [] for name in selections}
    in_suite = {}
    for rep in range(WALL_REPEATS):
        for name in selections:
            tj = os.path.join(WORK, "suite_times.json")
            code, wall = run_pytest(select=sel_files[name], times=tj)
            assert code == 0, f"{name} suite failed on unmodified tabulate"
            walls[name].append(wall)
            in_suite.setdefault(name, []).append(sum(v["seconds"] for v in json.load(open(tj)).values()
                                                     if v["outcome"] == "passed"))
    empty_walls = [run_pytest(select=os.path.join(WORK, "select_none.txt"))[1] for _ in range(WALL_REPEATS)]
    rows = []
    for name, tests in selections.items():
        got = run_coverage(sel_files[name] or write_selection("all", full), "suite") - import_only
        rows.append({"suite": name, "tests": len(tests),
                     "model_cost_s": minimize.suite_cost(tests, base, needs, setup),
                     "naive_model_cost_s": sum(raw[t] for t in tests),
                     "measured_test_time_s": statistics.median(in_suite[name]),
                     "wall_seconds_median": statistics.median(walls[name]),
                     "wall_seconds_min": min(walls[name]), "wall_seconds_max": max(walls[name]),
                     "lines_required": len(req), "lines_covered_measured": len(got & req),
                     "coverage_retained": len(got & req) / len(req),
                     "solve_seconds": {"greedy": greedy_s, "ILP": ilp_s}.get(name, 0.0)})
    rows.append({"suite": "pytest start-up only (0 tests)", "tests": 0,
                 "wall_seconds_median": statistics.median(empty_walls),
                 "wall_seconds_min": min(empty_walls), "wall_seconds_max": max(empty_walls)})
    rand_cov = [len(minimize.covered_by(r, cover)) / len(req) for r in randoms]
    rows.append({"suite": "random x10 (model only)", "tests": k,
                 "model_cost_s": statistics.mean(minimize.suite_cost(r, base, needs, setup) for r in randoms),
                 "coverage_retained": statistics.mean(rand_cov),
                 "coverage_retained_min": min(rand_cov), "coverage_retained_max": max(rand_cov),
                 "lines_required": len(req)})
    sel = pd.DataFrame(rows)
    save_results(sel, TRACK, "j_selection")
    print(sel.round(4).to_string(), flush=True)

    # mutation check
    suites = ["full suite", "greedy", "ILP", "random (seed 0)"]
    original = open(os.path.join(TARGET, MODULE)).read()
    mrows = []
    try:
        for i, mut in enumerate(mutants(req, N_MUTANTS)):
            open(os.path.join(TARGET, MODULE), "w").write(mut["code"])
            row = {"mutant": i, "line": mut["line"], "op": f'{mut["from"]} -> {mut["to"]}'}
            for name in suites:
                code, _ = run_pytest(select=sel_files[name], stop_first=True, timeout=120)
                row[name] = code != 0
            mrows.append(row)
            print("mutant", i, row, flush=True)
    finally:
        open(os.path.join(TARGET, MODULE), "w").write(original)
        subprocess.run(["git", "-C", TARGET, "checkout", "-q", "--", "."], check=True)
    mdf = pd.DataFrame(mrows)
    save_results(mdf, TRACK, "j_mutation")
    killed = {n: int(mdf[n].sum()) for n in suites}
    missed = {n: int((mdf["full suite"] & ~mdf[n]).sum()) for n in suites[1:]}
    print("killed", killed, "killed by full but missed", missed)

    tests_df = pd.DataFrame({"nodeid": full, "raw_seconds": [raw[t] for t in full],
                             "base_seconds": [base[t] for t in full],
                             "imports_charged_in_full_run": [",".join(sorted(charged[t])) for t in full],
                             "needs": [",".join(sorted(needs[t])) for t in full],
                             "lines": [len(cover[t]) for t in full],
                             "in_greedy": [t in set(greedy) for t in full], "in_ilp": [t in set(ilp) for t in full],
                             "in_naive_ilp": [t in set(naive) for t in full]})
    save_results(tests_df.sort_values("raw_seconds", ascending=False), TRACK, "j_tests")
    meta = {"repo": REPO, "commit": COMMIT, "loc": loc, "outcomes_full_run": outcomes,
            "passing_tests": len(full), "lines_required": len(req), "import_only_lines": len(import_only),
            "shared_setup_ms": {m: c * 1000 for m, c in setup.items()},
            "ilp": info, "naive_ilp": naive_info, "greedy_solve_s": greedy_s, "ilp_solve_s": ilp_s,
            "mutants": len(mdf), "killed": killed, "killed_by_full_missed_by": missed,
            "python": sys.version.split()[0]}
    json.dump(meta, open(os.path.join(RESULTS, TRACK, "j_meta.json"), "w"), indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    names = ["full suite", "ILP, naive costs", "greedy", "ILP", "random (seed 0)"]
    colors = ["#888", "#e9c46a", "#4c72b0", "#e76f51", "#bbb"]
    s = sel.set_index("suite")
    axes[0].bar(names, [s.loc[n, "measured_test_time_s"] for n in names], color=colors)
    axes[0].set_ylabel("measured time inside tests (s)")
    axes[1].bar(names, [s.loc[n, "wall_seconds_median"] for n in names], color=colors)
    axes[1].axhline(s.loc["pytest start-up only (0 tests)", "wall_seconds_median"], ls="--", c="k", lw=1)
    axes[1].set_ylabel("end-to-end wall time, median of 5 (s)")
    axes[2].bar(suites, [killed[n] for n in suites], color=["#888", "#4c72b0", "#e76f51", "#bbb"])
    axes[2].set_ylabel(f"mutants killed (of {len(mdf)})")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    fig.tight_layout()
    save_plot(fig, TRACK, "j_minimization")


if __name__ == "__main__":
    main()
