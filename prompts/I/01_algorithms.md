# I1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 9252bf4

## Original prompt
```
In trackI_quantum_algos_qml/ (Qiskit + Aer):
- Hand-written circuits: Deutsch–Jozsa (constant + balanced) and Bernstein–Vazirani. Tests: correct answer with prob 1.
- Also: Grover (3–4 qubits), QFT (check vs numpy FFT), QPE (T gate → phase 1/8), Shor for 15 (textbook circuit is
  fine), and annealing on a small QUBO with dwave-neal (labelled as classical simulation).
- Run BV with a fake-backend noise model vs ideal.
- Grover vs linear search for n = 2..8 qubits: oracle queries, transpiled gate counts, simulator wall time.
  Explain why the simulation time still grows exponentially.
```

## AI response (excerpt)
Generated `trackI_quantum_algos_qml/algorithms.py`: DJ, BV, Grover (oracle + diffuser by hand), QFT by hand, QPE and Shor (textbook a mod 15 circuit) on top of the hand-written inverse QFT, continued-fraction post-processing; quantum annealing as exact time evolution of a transverse-field Hamiltonian (6 qubits) vs dwave-neal; ideal vs noisy runs; Grover vs linear search.

## Revised prompt / corrections
- Plan said annealing via dwave-neal only; the AI added an exact simulation of quantum annealing (time evolution) so the quantum algorithm itself is demonstrated, with neal as the classical comparison.

## Verification note
Tests: DJ/BV exact outputs, Grover success = sin^2((2k+1)θ), QFT = DFT matrix, QPE T-gate gives 001, Shor post-processing gives 3 and 5. Shor success 51.6% vs 50% theory. Grover plot inspected.
