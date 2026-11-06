# Track I — Quantum algorithms and QML advantage

Run: `python trackI_quantum_algos_qml/run.py` (≈ 15 s) · Tests: `pytest tests/test_trackI.py`
All numbers from `results/trackI_quantum_algos_qml/*.csv`. **Everything is simulated** (Qiskit Aer statevector /
sampling; "noisy" = the depolarizing + readout noise model in `common/quantum.py`). No real hardware was used
(no IBM Quantum hardware access was used), so the noisy runs stand in for hardware.

## Objective
Run every quantum algorithm in the course, write several circuits by hand, compare quantum ML with classical ML on
the same data, compare Grover with classical search, and explain when quantum helps and when it does not.

## Algorithms (`algorithms.py`; hand-written gate by gate: DJ, BV, Grover, QFT, plus QPE and Shor built on my QFT)

| Algorithm | Case | Ideal success | Noisy success |
|---|---|---|---|
| Deutsch–Jozsa (3 + 1 qubits) | constant f | 1.000 | 0.939 |
| | balanced f (parity / x₀) | 1.000 | 0.996 / 0.974 |
| Bernstein–Vazirani | s = 1011 (5 qubits) | 1.000 | 0.899 |
| | s = 110101 (7 qubits) | 1.000 | 0.849 |
| Grover | n = 3, 2 iterations | 0.949 | 0.710 |
| | n = 4, 3 iterations | 0.960 | 0.416 |
| | n = 5, 4 iterations | 0.999 | **0.093** (random = 0.031) |
| QPE | T gate, φ = 1/8, 3 counting qubits | 1.000 (estimate 0.125) | — |
| | φ = 1/3, 3 / 6 counting qubits | 0.68 (estimate 0.375 / 0.328) | — |

- One query decides DJ (classically up to 2ⁿ⁻¹ + 1) and BV (classically n). Noise grows with circuit size: Grover's deeper circuits fall almost to random guessing at 5 qubits.
- QPE gives exact answers only when φ fits in t bits; otherwise more counting qubits are needed for more precision.
- **QFT:** my circuit equals the DFT matrix to 3.8e-15 and Qiskit's `QFTGate` to 2.2e-15.
- **Shor, N = 15, a = 7** (`i1_shor.csv`): measured phases 0, 1/4, 1/2, 3/4 (~25% each). Continued fractions give order r = 4 from 1/4 and 3/4, so gcd(7² ± 1, 15) = **3 and 5**. 51.6% of shots give the factors (theory 50%). The textbook modular-multiplication circuit transpiles to **13,328 CX gates** (depth 28,645): far beyond what noisy hardware can run.
- **Quantum annealing** (`i1_annealing.csv`): exact simulation of H(s) = −(1−s)ΣX + s·H_Max-Cut on the 6-node Track B graph.

  | anneal time T | 0.5 | 1 | 2 | 5 | 10 | 20 | 50 |
  |---|---|---|---|---|---|---|---|
  | P(optimal cut) | 0.04 | 0.06 | 0.12 | 0.28 | 0.54 | 0.88 | **0.99** |

  Random guessing gives 0.031; the minimum spectral gap is 0.56. Slow (adiabatic) evolution finds the optimum, as the adiabatic theorem predicts. Classical simulated annealing (dwave-neal) with only 10 sweeps already finds it in 97% of reads.

## Algorithmic advantage: Grover vs linear search (`i1_grover_vs_linear.csv`, `.png`)

| N | Grover oracle queries | classical queries (measured mean) | CX gates | Grover simulation time | classical time |
|---|---|---|---|---|---|
| 16 | 3 | 8.4 | 84 | 5.2 ms | 0.004 ms |
| 256 | 12 | 124 | 4,080 | 77 ms | 0.012 ms |
| 1,024 | 25 | 528 | 14,500 | 283 ms | 0.047 ms |

**The quadratic advantage is real in oracle queries (25 vs ~512 at N = 1,024)**, but in wall-clock time the simulated
Grover is about 6,000× slower. A classical simulator has to track all 2ⁿ amplitudes (cost grows like N, plus the gates),
and each oracle call needs a multi-controlled gate that decomposes into thousands of CX gates. The advantage only
appears on real hardware, when the oracle is expensive and the error rate is low enough for thousands of CX gates.
That is not true of today's devices (see the noisy Grover row above).

## QML vs classical (`i2_qml_vs_classical.csv`; make_moons, 200 points, 5 train/test splits)
All models except the VQC get the same treatment: hyperparameters chosen by 5-fold CV on the training set only.

| Model | Test accuracy | Train time | Parameters | Circuit runs on hardware |
|---|---|---|---|---|
| Quantum kernel SVM (2-qubit ZZ feature map) | 0.934 ± 0.021 | 0.23 s | 39 support vectors | 14,950 kernel circuits (× shots) |
| Variational quantum classifier (Track C circuit, Adam, untuned) | 0.958 ± 0.013 | 0.21 s | 12 angles | 144,000 (parameter-shift) |
| RBF SVM | **0.962 ± 0.019** | 0.05 s | 34 support vectors | 0 |
| Logistic regression | 0.866 ± 0.017 | 0.01 s | 3 weights | 0 |
| MLP (16 hidden) | 0.914 ± 0.042 | 0.15 s | 65 weights | 0 |

- **No quantum advantage.** The VQC roughly ties a tuned RBF SVM, with fewer parameters, but would need 144k circuit runs × shots on a device. The classical SVM trains in 50 ms on a laptop.
- The quantum kernel scored 0.744 with an untuned feature scale. Tuning the scale (s = 0.25), like the classical models' hyperparameters, raised it to 0.934 (failure log #10). Quantum kernels are very sensitive to how data is encoded.
- Training times are simulator times, which are fast because the circuits have 2 qubits; they say nothing about hardware time.

## When quantum helps, and when it doesn't
- **Helps (in principle):** query complexity (DJ 1 vs 2ⁿ⁻¹+1, BV 1 vs n, Grover √N vs N); structured problems like period finding (Shor).
- **Doesn't help here:**
  - Simulating the circuits costs time and memory that grow exponentially with qubit count.
  - Noise wipes out deep circuits (Grover at 5 qubits, QAOA p > 1, VQE outside chemical accuracy).
  - Barren plateaus kill gradients in deep variational circuits (Track C).
  - Loading data costs circuit runs for every kernel entry or gradient.
  - Problem sizes small enough to simulate are trivial for classical methods.
- **Compared with parallel computing (Track D):** parallel speedups are *measured* on real hardware today: **6.8× on 10 M4 cores** for uneven tasks with load balancing, 7.0× scaled speedup (Gustafson). They are limited by Amdahl's serial fraction (~5%), overhead (tiny tasks ran 2–3× slower than serial) and locks/locality. The quantum "speedups" here are theoretical (Grover's 25 vs 512 queries) or simulated, and their wall-clock times are *slower* than classical (Grover's simulation is 6,000× slower than a Python linear search). Parallelism gives at most a constant factor (≤ number of cores) that is available now. Quantum algorithms promise better scaling (√N, or exponential for Shor) that today's noise and qubit counts don't yet deliver.

## Correctness checks (`tests/test_trackI.py`, all passing)
DJ / BV ideal outputs; Grover success equals sin²((2k+1)θ); QFT equals the DFT matrix; QPE gives exactly 001 for the T gate;
Shor post-processing returns 3 and 5; quantum kernel is symmetric, has unit diagonal and is positive semi-definite.

## Limitations
Simulation only; small problem sizes; a simple noise model; the VQC was not tuned (it would likely gain slightly);
Shor uses the textbook N = 15 circuit, which hard-codes multiplication by a mod 15.
