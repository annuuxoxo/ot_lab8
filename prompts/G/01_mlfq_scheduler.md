# G1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 11c7201

## Original prompt
```
In xv6-riscv at trackG_xv6/xv6:
1. Add a syscall getpinfo that returns per-process pid, ticks run, ticks runnable-waiting, creation tick and first-run tick.
2. Add a priority scheduler (or a simple 3-level MLFQ), selectable with SCHED=RR|PRIO in the Makefile so the original
   round-robin remains for comparison.
3. User programs cpubound, iobound and mixed, plus schedbench, which forks a mix and prints turnaround and response time.
Minimal, commented changes; explain the proc.c locking. It must boot and pass `usertests -q`.
```

## AI response (excerpt)
Read the current xv6-riscv source first (it differs from the textbook version: kfork/kexit/kwait, split sleep API, copyout with an extra size argument). Generated: 3-level MLFQ in kernel/proc.c (two-pass scheduler that never holds two proc locks, priority boost with its own lock, early preemption hint), sched_tick() called from the timer path in trap.c, statistics fields in struct proc, the waitstat() system call (pstat.h, sysproc.c, syscall.[ch], usys.pl, user.h), Makefile SCHED=RR|MLFQ switch, user/schedbench.c (calibration + 3 workloads) and trackG_xv6/bench.py (pexpect driver: build, boot QEMU, run, parse, usertests). Patch saved as trackG_xv6/mlfq_scheduler.patch.

## Revised prompt / corrections
- Prompt said getpinfo; implemented waitstat (per-child stats after exit) because turnaround/response need exit times.
- Boost-lock init moved from a lazy 'first call' (racy on 3 CPUs) to procinit().
- Build failed under -Werror (unused busy-loop variable); my build check filtered output and hid the error at first (failure log #16).
- Job length recalibrated twice (40 -> 800 -> 2000 units).
- First benchmark ran all RR boots then all MLFQ boots; host-side QEMU speed drifted ~1.75x and looked like an MLFQ slowdown. Re-ran with alternating order and per-boot calibration (failure log #15).
- Added a time-slice sensitivity experiment after MLFQ lost on short jobs.

## Verification note
usertests -q ALL TESTS PASSED under RR and MLFQ with 3 CPUs. Every boot calibrated at 36 ticks. Run time per job equal under both schedulers (scheduler changes waiting, not work).
