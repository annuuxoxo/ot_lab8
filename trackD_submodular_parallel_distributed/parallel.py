"""Track D part 2: parallel program optimization, MEASURED on the M4's cores.

Tasks are SHA-256 hashes of buffers of uneven size. hashlib releases Python's GIL while hashing large buffers, so plain
threads run truly in parallel, one task per core, with no hidden library threading.
(A first version used NumPy matrix multiplications, but on Apple Silicon NumPy calls Accelerate, which runs on a shared
matrix coprocessor and cannot be limited by threadpoolctl, so it did not measure per-core parallelism: failure log #11.)

Scheduling strategies (all assign the same task list to p worker threads):
  static      : round-robin by index, fixed in advance (no load information)
  LPT greedy  : Longest Processing Time first - sort by estimated cost, give each task to the least-loaded worker
  central queue: workers repeatedly take the next task from one shared queue (dynamic self-scheduling)
  work stealing: each worker has its own deque (filled round-robin); it pops its own tasks from one end and, when empty,
                 steals from the other end of a random victim's deque
Metrics: makespan (wall time), speedup S(p) = T(1)/T(p), efficiency E(p) = S(p)/p; Amdahl and Gustafson fits.
"""
import os
import random
import sys
import threading
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import save_plot, save_results  # noqa: E402

TRACK = "trackD_submodular_parallel_distributed"
CORES = os.cpu_count()

BLOCK = 256 * 1024  # one cost unit = hashing 256 KB
_BUF = np.random.default_rng(0).integers(0, 256, 200 * BLOCK, dtype=np.uint8).tobytes()
_VIEW = memoryview(_BUF)


def task(m):
    """Hash m blocks (m * 256 KB). Cost is proportional to m."""
    return hashlib.sha256(_VIEW[: m * BLOCK]).digest()


def mats(m):
    """Kept for the tiny-task experiment (no setup needed for hashing)."""
    return None


def make_tasks(n=120, seed=0, lo=4, hi=200):
    """Uneven task sizes: a few big buffers among many small ones (lognormal, mean ~ 40 blocks = 10 MB)."""
    rng = np.random.default_rng(seed)
    return [int(s) for s in np.clip(rng.lognormal(np.log(30), 0.8, n), lo, hi).astype(int)]


# ---------------- schedulers ----------------

def _run_lists(lists):
    """Each thread runs its own fixed list. Returns makespan and per-worker busy time."""
    busy = [0.0] * len(lists)

    def worker(i):
        t0 = time.perf_counter()
        for m in lists[i]:
            task(m)
        busy[i] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(lists))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, busy


def static_rr(tasks, p):
    return _run_lists([tasks[i::p] for i in range(p)])


def lpt(tasks, p):
    loads, lists = [0.0] * p, [[] for _ in range(p)]
    for m in sorted(tasks, reverse=True):  # estimated cost = m^3
        i = int(np.argmin(loads))
        lists[i].append(m)
        loads[i] += m  # estimated cost = number of blocks
    return _run_lists(lists)


def central_queue(tasks, p):
    q = deque(tasks)
    lock = threading.Lock()
    busy = [0.0] * p

    def worker(i):
        t0 = time.perf_counter()
        while True:
            with lock:  # every task grab synchronizes on the one shared queue
                if not q:
                    break
                m = q.popleft()
            task(m)
        busy[i] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(p)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, busy


def work_stealing(tasks, p, seed=0):
    deques = [deque(tasks[i::p]) for i in range(p)]
    locks = [threading.Lock() for _ in range(p)]
    busy, steals = [0.0] * p, [0] * p

    def worker(i):
        rnd = random.Random(seed + i)
        t0 = time.perf_counter()
        while True:
            m = None
            with locks[i]:
                if deques[i]:
                    m = deques[i].pop()  # own work: take from the back
            if m is None:
                victims = [v for v in range(p) if v != i]
                rnd.shuffle(victims)
                for v in victims:
                    with locks[v]:
                        if deques[v]:
                            m = deques[v].popleft()  # steal from the front (the victim works at the back)
                            steals[i] += 1
                            break
                if m is None:
                    break  # nothing left anywhere
            task(m)
        busy[i] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(p)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, busy, sum(steals)


STRATEGIES = {"static round-robin": static_rr, "LPT greedy": lpt, "central queue": central_queue,
              "work stealing": lambda t, p: work_stealing(t, p)[:2]}


def amdahl(p, s):
    return 1 / (s + (1 - s) / p)


def scheduling_experiment(repeats=3):
    tasks = make_tasks()
    ps = sorted(set([1, 2, 4, 6, 8, CORES]))
    t_serial = np.median([_run_lists([tasks])[0] for _ in range(repeats)])
    rows = []
    for name, fn in STRATEGIES.items():
        for p in ps:
            times, imb = [], []
            for _ in range(repeats):
                mk, busy = fn(tasks, p)
                times.append(mk)
                imb.append(max(busy) / (np.mean(busy) + 1e-12))  # load imbalance: 1.0 = perfectly balanced
            mk = float(np.median(times))
            rows.append({"strategy": name, "p": p, "makespan_s": mk, "makespan_std": float(np.std(times)),
                         "speedup": t_serial / mk, "efficiency": t_serial / mk / p, "load_imbalance": float(np.median(imb))})
    df = pd.DataFrame(rows)
    _, _, steals = work_stealing(tasks, CORES)
    fits = {}
    for name, g in df.groupby("strategy", sort=False):
        (s,), _ = curve_fit(amdahl, g["p"], g["speedup"], p0=[0.05], bounds=(0, 1))
        fits[name] = s
    df["amdahl_serial_fraction_fit"] = df["strategy"].map(fits)
    save_results(df, TRACK, "d1_scheduling")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    pp = np.linspace(1, CORES, 50)
    for name, g in df.groupby("strategy", sort=False):
        line, = a1.plot(g["p"], g["speedup"], "o-", label=f"{name} (Amdahl s={fits[name]:.3f})")
        a1.plot(pp, amdahl(pp, fits[name]), ":", color=line.get_color(), alpha=0.6)
        a2.plot(g["p"], g["efficiency"], "o-", label=name)
    a1.plot(pp, pp, "k--", alpha=0.4, label="ideal")
    a1.axvline(4, color="grey", lw=0.8)
    a1.text(4.1, 1, "4 performance\ncores", fontsize=7)
    a1.set(xlabel="threads p", ylabel="speedup T(1)/T(p)", title=f"{len(tasks)} uneven SHA-256 tasks, M4 ({CORES} cores)")
    a2.set(xlabel="threads p", ylabel="efficiency S(p)/p", title="Parallel efficiency")
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    save_plot(fig, TRACK, "d1_scheduling")
    plt.close(fig)
    return df, steals, t_serial


def gustafson_experiment(base=24, repeats=3):
    """Scaled problem: work grows with p (base tasks per thread). Gustafson's scaled speedup = p*W / T(p) vs W / T(1)."""
    rows = []
    t1 = None
    for p in sorted(set([1, 2, 4, 6, 8, CORES])):
        tasks = make_tasks(base * p, seed=p)
        tp = np.median([work_stealing(tasks, p)[0] for _ in range(repeats)])
        work = sum(tasks)
        rate = work / tp
        if p == 1:
            t1 = rate
        rows.append({"p": p, "tasks": len(tasks), "time_s": tp, "scaled_speedup": rate / t1})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "d1_gustafson")
    return df


def overhead_experiment(n=20000):
    """Tiny tasks (hash 4 KB): the per-task synchronization (lock + thread switch) costs more than the task itself."""
    tiny = memoryview(_BUF)[:4096]
    global task
    real_task = task
    task = lambda m: hashlib.sha256(tiny).digest()  # noqa: E731
    tasks = [1] * n
    t1 = _run_lists([tasks])[0]
    rows = [{"setting": "serial", "p": 1, "time_s": t1, "speedup": 1.0}]
    for p in (2, 4, CORES):
        t = central_queue(tasks, p)[0]
        rows.append({"setting": "central queue (1 lock per task)", "p": p, "time_s": t, "speedup": t1 / t})
        t = static_rr(tasks, p)[0]
        rows.append({"setting": "static chunks (no locking)", "p": p, "time_s": t, "speedup": t1 / t})
    task = real_task
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "d1_overhead")
    return df


def sync_locality_experiment():
    rows = []
    # Synchronization: 4 threads each add 1 to a counter 200k times - shared counter behind one lock vs private counters.
    n, p = 200_000, 4
    counter = [0]
    lock = threading.Lock()

    def shared():
        for _ in range(n):
            with lock:
                counter[0] += 1

    local = [0] * p

    def private(i):
        c = 0
        for _ in range(n):
            c += 1
        local[i] = c

    for name, target, args in [("shared counter + lock", shared, lambda i: ()), ("private counters, summed at end", private, lambda i: (i,))]:
        ts = [threading.Thread(target=target, args=args(i)) for i in range(p)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        rows.append({"experiment": "synchronization", "variant": name, "time_s": time.perf_counter() - t0})
    assert counter[0] == n * p and sum(local) == n * p

    # Locality: sum a 4000x4000 C-ordered (row-major) array row by row vs column by column.
    A = np.random.default_rng(0).standard_normal((4000, 4000))
    for name, fn in [("row-wise (contiguous memory)", lambda: sum(A[i].sum() for i in range(A.shape[0]))),
                     ("column-wise (stride 32 KB)", lambda: sum(A[:, j].sum() for j in range(A.shape[1])))]:
        t0 = time.perf_counter()
        fn()
        rows.append({"experiment": "locality", "variant": name, "time_s": time.perf_counter() - t0})
    df = pd.DataFrame(rows)
    save_results(df, TRACK, "d1_sync_locality")
    return df


def main():
    pd.set_option("display.width", 220)
    df, steals, ts = scheduling_experiment()
    print(f"serial time {ts:.3f}s; steals at p={CORES}: {steals}")
    print("== Scheduling ==\n", df.to_string(index=False))
    print("\n== Gustafson ==\n", gustafson_experiment().to_string(index=False))
    print("\n== Overhead ==\n", overhead_experiment().to_string(index=False))
    print("\n== Sync / locality ==\n", sync_locality_experiment().to_string(index=False))


if __name__ == "__main__":
    main()
