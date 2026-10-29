# Track C — Quantum optimization

Run: `python trackC_quantum_opt/run.py` (≈ 2 min; needs Track B results) · Tests: `pytest tests/test_trackC.py`
All numbers from `results/trackC_quantum_opt/*.csv`. **Every result is simulated** (Qiskit Aer 0.18, PennyLane 0.45
`default.qubit`). No real quantum hardware was used. "Noisy" means a simulated noise model: depolarizing error
1e-3 (1-qubit gates) / 1e-2 (CX) + 2% readout error, roughly today's superconducting devices (`common/quantum.py`).

## Objective
Apply the Chapter 6 quantum optimization methods (QAOA, VQE, PQC training with parameter-shift / SPSA / QNG),
compare them with the classical baseline from Track B, and measure the practical limits: finite shots, noise,
barren plateaus, embedding QUBOs on annealers.

## Formulations
- **QAOA, Max-Cut:** the same graphs and QUBO as Track B (`results/trackB_discrete/maxcut_graphs.json`). Cost operator C = Σ_edges (1 − Z_iZ_j)/2,
  whose eigenvalue on |x⟩ is cut(x) (checked in tests). The circuit applies p layers of exp(−iγC) (RZZ per edge) and exp(−iβΣX) (RX).
  We maximize ⟨C⟩ over the 2p angles with COBYLA (5 random starts + an INTERP warm start from depth p−1).
- **VQE, H₂** (0.735 Å, 2-qubit parity-reduced Hamiltonian): minimize ⟨ψ(θ)|H|ψ(θ)⟩ with a 6-parameter RY/CNOT ansatz.
- **PQC classifier:** 2-qubit data re-uploading circuit, 12 angles, square loss on 80 make_moons points.

## Results

**QAOA vs classical** (`c1_qaoa.csv`, `c1_qaoa.png`). Approximation ratio = expected cut / optimal cut.

| nodes | p = 1 | p = 2 | p = 3 | P(sample optimum), best p | simulated annealing (Track B) | random |
|---|---|---|---|---|---|---|
| 6 | 0.778 | 0.851 | 0.918 | 0.62 | 1.00 | 0.58 |
| 8 | 0.811 | 0.865 | 0.902 | 0.26 | 1.00 | 0.68 |
| 10 | 0.822 | 0.881 | 0.912 | 0.25 | 1.00 | 0.68 |
| 12 | 0.840 | 0.895 | — | 0.06 | 1.00 | 0.72 |

QAOA beats random guessing and improves with depth, but **classical simulated annealing finds the exact optimum every
time in milliseconds**, while QAOA needs 200–4,700 circuit evaluations. On these sizes QAOA offers no advantage.
The first version used only random restarts and got p = 2 barely better than p = 1 (an optimizer failure, not a
limit of QAOA); warm-starting from depth p−1 (INTERP) fixed it (failure log #8).

**Finite shots** (`c1_shots.csv`, 8-node graph, p = 1)

| shots | std of ⟨C⟩ estimate (measured) | theory σ/√shots | ratio after optimizing with this many shots (5 seeds) |
|---|---|---|---|
| 100 | 0.164 | 0.149 | 0.738 ± 0.050 |
| 1,000 | 0.047 | 0.047 | 0.777 ± 0.049 |
| 10,000 | 0.011 | 0.015 | 0.784 ± 0.052 |

Estimation error follows 1/√shots. Each 10× more shots costs 10× more runtime on hardware, for √10 less noise.
The optimizer copes with noise at ≥ 1,000 shots, but stays below the exact-optimum ratio of 0.811.

**Noise** (`c1_noise.csv`, 8-node graph)

| p | CX gates | ideal | noisy | lost to noise |
|---|---|---|---|---|
| 1 | 38 | 0.811 | 0.787 | 0.024 |
| 2 | 76 | 0.865 | 0.810 | 0.056 |
| 3 | 114 | 0.901 | 0.811 | 0.090 |

**With noise, extra depth stops helping.** p = 2 and p = 3 give the same result as p = 1 (≈ 0.81), because every
added layer adds more CX errors than it adds algorithmic gain.

**VQE** (`c1_vqe.csv`): ideal VQE reaches the exact ground energy −1.857275 Ha (error 4e-8 Ha, 421 energy evaluations).
Under noise (exact density-matrix simulation) the best energy is −1.8287 Ha, an error of **28.6 mHa, far outside chemical accuracy (1.6 mHa)**.

**PQC training: parameter-shift vs SPSA** (`c2_classifier.csv`, 3 seeds, equal budget of 76,800 single-circuit runs)
- My parameter-shift gradient matches PennyLane autodiff to 3.5e-16.
- Final training loss: parameter-shift GD 0.200 ± 0.041, SPSA 0.239 ± 0.057. Test accuracy is 0.975 for both (per seed 1.0 / 0.925 / 1.0; the 40-point test set is small).
- Parameter-shift needs 2 runs per parameter per step (24 here); SPSA needs 2 runs per step, whatever the number of parameters. So SPSA takes 480 steps for the same budget. With only 12 parameters SPSA is slightly worse; its advantage grows with parameter count.

**Quantum natural gradient vs GD** (`c2_qng.csv`, VQE cost, 5 starts, step sizes 0.05–1.6)

| | best step size | median steps to chemical accuracy | circuit runs per step |
|---|---|---|---|
| GD | 0.8 | 6 | 8 |
| QNG (block-diagonal metric) | 0.2 | 6 | 11 |

At the *same* step size QNG needs about 4× fewer steps, but **once each method's step size is tuned, they tie**,
and QNG pays about 40% more circuit runs per step for the metric tensor. An earlier narrower grid (up to 0.4) cut off
GD's best step size and made QNG look 2× faster (failure log #9). QNG also diverged for step sizes ≥ 0.8.

**Barren plateaus** (`c2_barren_plateau.png`): random hardware-efficient circuits, cost ⟨Z₀Z₁⟩, 100 samples per point.
Deep circuits (5n layers) lose gradient variance exponentially, from 0.087 (2 qubits) to 0.0022 (10 qubits), a straight line on the
log plot. Shallow circuits (2 layers) stay at ≈ 0.09. On a device, a gradient that small needs about 1/Var shots just to tell it apart from zero.

**Embedding QUBOs on annealers** (`c2_embedding.csv`, minorminer): a fully connected QUBO on n variables needs chains of physical qubits.

| n | Chimera C8 (512 q): qubits used / max chain | Pegasus P6 (680 q): qubits / max chain |
|---|---|---|
| 16 | 81 / 6 | 40 / 3 |
| 32 | 327 / 13 | 144 / 6 |
| 36 | **fails** | 166 / 6 |
| 40 | **fails** | 222 / 7 |

Physical qubits grow roughly as n². Chimera cannot fit K₃₆ even with 512 qubits. Long chains also need extra
"chain strength" penalties, which squeeze the problem's own energy scale (the same effect as the large penalty in Track B).

## Correctness checks (`tests/test_trackC.py`, all passing)
Cut-operator diagonal equals the classical cut on every bitstring; QAOA expected cut equals ⟨C⟩ from the operator;
QAOA at zero angles gives the random-guess value; INTERP warm start shape and values; VQE reaches the exact ground energy;
parameter-shift equals autodiff.

## Limitations
Small instances (≤ 12 qubits, so everything can be simulated exactly); a simple uniform noise model rather than a
calibrated device model; the QNG comparison uses one tiny problem; circuit-run counts on the simulator differ from
hardware (the simulator computes gradients by backpropagation).
