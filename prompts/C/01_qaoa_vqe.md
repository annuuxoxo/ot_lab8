# C1

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29
- **Commit:** 772039e

## Original prompt
```
In trackC_quantum_opt/ (Qiskit + Aer, simulator only):
1. QAOA for Max-Cut on the Track B graphs, p = 1 and 2, COBYLA. Report the approximation ratio vs brute force and
   vs SA, circuit evaluations and time.
2. VQE on a 2-qubit Hamiltonian (e.g. H2 as a Pauli sum) vs exact diagonalization.
3. QAOA p=1 at 100 / 1000 / 10000 shots (5 seeds) and with a fake-backend noise model vs noiseless.
Test: QAOA bitstring energy == classical QUBO energy.
```

## AI response (excerpt)
Generated `common/quantum.py` (shared noise model: depolarizing 1e-3/1e-2 + 2% readout, transpile to rz/sx/x/cx) and `trackC_quantum_opt/qaoa_vqe.py`: QAOA circuit built by hand (RZZ cost layer, RX mixer), cut operator as SparsePauliOp, COBYLA with restarts, shots study (estimator std + optimizing with shot noise), ideal vs noisy for p = 1-3, VQE for the textbook 2-qubit H2 Hamiltonian, ideal (statevector) vs noisy (density matrix).

## Revised prompt / corrections
- Theoretical shot-noise std first used the spread of cuts over ALL bitstrings (wrong distribution); fixed to the QAOA output distribution.
- QAOA p = 2 was barely better than p = 1 with random restarts only. Added the INTERP warm start from depth p-1 and maxiter 300·p; now the ratio increases with p (failure log #8).

## Verification note
Tests: cut operator diagonal = classical cut; QAOA <C> from probabilities = operator expectation; zero angles = random guess; VQE reaches the exact energy. Shot std matches sigma/sqrt(shots) in the results table.
