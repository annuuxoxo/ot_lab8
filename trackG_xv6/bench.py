"""Track G driver: build xv6 with a given scheduler, boot it in QEMU, run commands, collect output.

  python trackG_xv6/bench.py calib            calibration run (RR)
  python trackG_xv6/bench.py bench [repeats]  schedbench 1/2/3 under RR and MLFQ (1 CPU), results -> results/trackG_xv6
  python trackG_xv6/bench.py usertests        usertests -q under RR and MLFQ (3 CPUs)
"""
import os
import re
import subprocess
import sys
import time

import pandas as pd
import pexpect

HERE = os.path.dirname(os.path.abspath(__file__))
XV6 = os.path.join(HERE, "xv6")
RESULTS = os.path.join(os.path.dirname(HERE), "results", "trackG_xv6")
os.makedirs(RESULTS, exist_ok=True)


# scheduler variants: name -> (SCHED, extra compiler flags)
VARIANTS = {"RR": ("RR", ""), "MLFQ": ("MLFQ", ""),
            "MLFQ-2/4/8": ("MLFQ", "-DMLFQ_Q0=2 -DMLFQ_Q1=4 -DMLFQ_Q2=8")}


def build(variant):
    sched, flags = VARIANTS[variant]
    subprocess.run(["make", "clean"], cwd=XV6, capture_output=True, check=True)
    subprocess.run(["make", f"SCHED={sched}", f"MLFQFLAGS={flags}", "kernel/kernel", "fs.img"], cwd=XV6,
                   capture_output=True, check=True)


def boot(variant, cpus):
    sched, flags = VARIANTS[variant]
    child = pexpect.spawn("make", [f"SCHED={sched}", f"MLFQFLAGS={flags}", f"CPUS={cpus}", "qemu"], cwd=XV6, encoding="utf-8",
                          timeout=900, codec_errors="replace")
    child.expect(r"\$ ")
    return child


def command(child, cmd, until, timeout=900):
    child.sendline(cmd)
    child.expect(until, timeout=timeout)
    out = child.before + child.after
    child.expect(r"\$ ", timeout=60)
    return out


def shutdown(child):
    child.sendcontrol("a")
    child.send("x")
    child.expect(pexpect.EOF, timeout=30)


def calib():
    build("RR")
    c = boot("RR", 1)
    print(command(c, "schedbench 0", r"CALIB.*ticks"))
    shutdown(c)


STAT = re.compile(r"STAT (\d+) (\w+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)")


CALIB = re.compile(r"CALIB (\d+) units took (\d+) ticks")


def bench(repeats=3, variants=("RR", "MLFQ"), out_name="g_schedbench_raw"):
    """Alternate RR and MLFQ on every repeat (RR,MLFQ then MLFQ,RR, ...) and calibrate at the start of every boot.
    A first version ran all RR boots and then all MLFQ boots; QEMU's speed on the host drifted by ~1.75x between the
    two batches and was mistaken for a scheduler effect (failure log #15). Times are reported in ticks and also
    normalized by that boot's calibration (ticks for one long job running alone)."""
    rows, calib_rows = [], []
    for rep in range(repeats):
        order = tuple(variants) if rep % 2 == 0 else tuple(reversed(variants))
        for sched in order:
            build(sched)
            c = boot(sched, 1)
            cal = int(CALIB.search(command(c, "schedbench 0", r"CALIB.*ticks")).group(2))
            calib_rows.append({"scheduler": sched, "repeat": rep, "calib_ticks_long_job_alone": cal})
            for w in (1, 2, 3):
                t0 = time.time()
                out = command(c, f"schedbench {w}", rf"DONE {w}")
                for m in STAT.finditer(out):
                    wl, kind, pid, ct, st, et, rt, lvl = m.groups()
                    ct, st, et, rt = int(ct), int(st), int(et), int(rt)
                    rows.append({"scheduler": sched, "repeat": rep, "workload": int(wl), "kind": kind, "pid": int(pid),
                                 "turnaround": et - ct, "response": st - ct, "run": rt, "waiting": et - ct - rt,
                                 "final_level": int(lvl), "calib": cal, "turnaround_norm": (et - ct) / cal})
                print(sched, rep, w, f"calib={cal}", f"{time.time() - t0:.1f}s", flush=True)
            shutdown(c)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, f"{out_name}.csv"), index=False)
    pd.DataFrame(calib_rows).to_csv(os.path.join(RESULTS, f"{out_name}_calibration.csv"), index=False)
    return df


def usertests():
    rows = []
    for sched in ("RR", "MLFQ"):
        build(sched)
        c = boot(sched, 3)
        t0 = time.time()
        out = command(c, "usertests -q", r"ALL TESTS PASSED|SOME TESTS FAILED|FAILED|panic", timeout=1800)
        passed = "ALL TESTS PASSED" in out
        rows.append({"scheduler": sched, "cpus": 3, "result": "ALL TESTS PASSED" if passed else "FAILED",
                     "minutes": (time.time() - t0) / 60})
        print(sched, rows[-1], flush=True)
        with open(os.path.join(RESULTS, f"usertests_{sched}.log"), "w") as f:
            f.write(out)
        shutdown(c)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "g_usertests.csv"), index=False)
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if what == "calib":
        calib()
    elif what == "bench":
        print(bench(int(sys.argv[2]) if len(sys.argv) > 2 else 3).groupby(["workload", "kind", "scheduler"])
              [["turnaround", "turnaround_norm", "response", "run", "waiting"]].mean())
    elif what == "quanta":  # MLFQ time-slice sensitivity: default slices vs longer slices, interleaved
        print(bench(int(sys.argv[2]) if len(sys.argv) > 2 else 4, ("MLFQ", "MLFQ-2/4/8"), "g_quanta_raw")
              .groupby(["workload", "kind", "scheduler"])[["turnaround", "response", "run", "waiting"]].mean())
    elif what == "usertests":
        usertests()
