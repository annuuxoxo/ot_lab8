# Track G — xv6 operating-system optimization: MLFQ scheduler

The kernel changes are in **`mlfq_scheduler.patch`** (13 files, +293/−15), against
[mit-pdos/xv6-riscv](https://github.com/mit-pdos/xv6-riscv) commit `e93dabe`. Benchmark driver: `bench.py`.
Results: `results/trackG_xv6/`. Everything runs in QEMU (RISC-V) on the M4; one timer tick ≈ 0.1 s.

```bash
git clone https://github.com/mit-pdos/xv6-riscv.git trackG_xv6/xv6 && cd trackG_xv6/xv6
git checkout e93dabe && git apply ../mlfq_scheduler.patch
make SCHED=MLFQ qemu            # or SCHED=RR (original round-robin, the default)
python trackG_xv6/bench.py bench 6      # RR vs MLFQ, 3 workloads, interleaved, 1 CPU
python trackG_xv6/bench.py quanta 4     # MLFQ time-slice sensitivity
python trackG_xv6/bench.py usertests    # usertests -q under both schedulers, 3 CPUs
```

## OS concept and objective
**Concept:** CPU scheduling, deciding which RUNNABLE process gets the CPU next. Stock xv6 uses round-robin with a
1-tick time slice: every process is treated the same, so an interactive process that runs briefly and then blocks on
I/O waits behind every CPU-bound process.
**Objective:** reduce turnaround and response time for interactive (I/O-bound) processes **without** hurting CPU-bound
throughput or fairness. Metrics per process: turnaround = exit − creation, response = first run − creation,
waiting = turnaround − run time (all in ticks).

## What was changed (kernel feature + statistics system call)
1. **3-level multi-level feedback queue (MLFQ) scheduler** (`kernel/proc.c`, chosen with `make SCHED=MLFQ`):
   - New processes start in queue 0 (highest priority). The scheduler always runs a process from the highest non-empty queue, round-robin within a queue.
   - Time slices are 1 / 2 / 4 ticks. A process that uses its whole slice is demoted one level, so CPU-bound jobs sink. A process that blocks before its slice ends keeps its level, so interactive jobs stay on top.
   - A higher-priority process that becomes runnable preempts the current one at the next tick.
   - **Priority boost:** every 40 ticks all processes go back to queue 0, so CPU-bound jobs cannot starve.
   - The timer interrupt calls `sched_tick()` (`kernel/trap.c`), which charges the tick and decides whether to preempt.
   - **Locking:** the scheduler never holds two process locks at once (deadlock-free). Pass 1 finds the best level, taking each `p->lock` briefly. Pass 2 re-checks state and level under the chosen process's lock before running it. The boost uses its own `boost_lock`. `sched_tick`'s "is a higher-priority process waiting?" scan reads without locks, as a hint only.
2. **Scheduling statistics:** new fields in `struct proc` (creation / first-run / exit tick, run ticks, level) and a new
   system call **`waitstat(int *status, struct procstat *st)`**: wait for a child and get its statistics (`kernel/pstat.h`, `sysproc.c`, `syscall.[ch]`, `usys.pl`, `user.h`).
3. **Benchmark program `user/schedbench.c`** with a calibration mode and three workloads.

## Workloads (1 CPU, so the policy alone decides; 6 interleaved repeats per scheduler)
1. **mixed:** 3 CPU-bound jobs (about 36 ticks of CPU each) + 3 interactive jobs (10 × [about 0.1 tick of CPU, then sleep 1 tick]), started together
2. **short vs long:** 1 long CPU job; 2 ticks later 4 short jobs (about 4.5 ticks each) arrive
3. **equal:** 4 identical CPU-bound jobs (fairness and overhead check)

## Results (`g_schedbench_raw.csv`; mean over repeats; ticks)

| workload | jobs | RR turnaround | MLFQ turnaround | RR response | MLFQ response |
|---|---|---|---|---|---|
| 1 mixed | interactive (I/O) | 33.0 | **13.0 (−61%)** | 3.0 | 3.0 |
| | CPU-bound | 109.8 | 110.1 | 1.0 | 1.0 |
| 2 short vs long | short | **22.4** | 26.0 (+16%) | 2.5 | 1.5 |
| | long | 54.3 | 54.7 | 0 | 0 |
| 3 equal | CPU-bound | 71.6 | 70.6 | 1.5 | 1.5 |

- **Interactive jobs finish 2.5× sooner under MLFQ** (13 vs 33 ticks). Under round-robin, every time an I/O job wakes
  it queues behind three CPU-bound jobs. Under MLFQ it stays in queue 0 (final level 0) and preempts them. CPU-bound
  jobs lose nothing measurable (110.1 vs 109.8 ticks): the interactive jobs use very little CPU.
- **Workload 3:** no throughput cost and the same fairness (Jain index of turnaround: RR 0.9999, MLFQ 0.9991).
- **Workload 2: MLFQ is worse for short jobs here.** A short job needs about 4.5 ticks, but queues 0 and 1 allow only
  1 + 2 = 3 ticks. So the short jobs sink to queue 2 next to the long job, where the slice is 4 ticks, longer than
  round-robin's 1 tick. MLFQ only approximates "shortest job first" for jobs that fit in the top queues.
- Run time per job is the same under both schedulers (e.g. 37.6 ticks for workload 1's CPU jobs), as it should be:
  the scheduler changes waiting, not work.

**Time-slice sensitivity** (`g_quanta_raw.csv`, 4 interleaved repeats): slices 1/2/4 vs 2/4/8 ticks

| | slices 1/2/4 | slices 2/4/8 |
|---|---|---|
| interactive turnaround / response | **13.0 / 3.0** | 16.0 / 6.0 |
| short-job turnaround (workload 2) | 26.0 | **23.0** |
| equal CPU jobs turnaround (workload 3) | 70.6 | **64.6** (fewer context switches) |

Longer slices help short and CPU-bound jobs but double the interactive response time. That's the classic trade-off,
so the slices should be chosen for the workload.

## Correctness (verified after every change)
- **`usertests -q`: ALL TESTS PASSED under both RR and MLFQ, with 3 CPUs** (`g_usertests.csv`, logs `usertests_*.log`), 1.2 min each. That covers fork/exec/pipes/files/memory and concurrency with the new locking.
- Both kernels build with `-Werror`. xv6 boots under both.
- Workload outputs are consistent: every child's run time ≈ its calibrated work, and `waitstat` returns stats for every child.
- **Measurement validity:** the first benchmark ran all RR boots and then all MLFQ boots; QEMU's speed on the host
  drifted by about 1.75× between the batches, making MLFQ look 2× slower (failure log #15). The final runs alternate
  RR/MLFQ on every repeat and calibrate at the start of every boot: all 12 boots calibrated at exactly 36 ticks for the
  same job, so the comparison is fair.

## Limitations
One CPU in the benchmark (MLFQ also passes usertests with 3); simple synthetic workloads; times are QEMU ticks, not
real hardware; no second kernel feature (copy-on-write fork was out of scope for the 5-day plan).
