# Tensor Networks — MPS / DMRG / TEBD

The strongly-correlated-1D corner of the suite: matrix product states, ground
states by two-site DMRG, real- and imaginary-time evolution by TEBD. Pure
`numpy`/`scipy` (SVD, sparse `eigsh`, `tensordot`) — no external tensor-network
library, matching the repo's "build it from scratch" ethos.

See `Tensor_Networks_Plan.md` for the design and the phase plan.

## Modules

| file | contents |
|---|---|
| `models.py` | local operators; MPO (finite-state-machine) Hamiltonians for TFIM (Pauli convention), Heisenberg spin-½/1 (and mixed-spin chains), 1D Hubbard (Jordan–Wigner, `d=4`); two-site TEBD gates; analytic references (TFIM `e_0`, Bethe-ansatz `¼−ln2`, Haldane gap) |
| `ed.py` | exact-diagonalization oracle — full sparse `H` for `N ≲ 14` (Hubbard via an explicit `2^{2N}` Fock-space build with JW signs), entanglement entropy, local expectations |
| `mps.py` | `MPS` class: mixed canonical form about a movable centre, SVD truncation with discarded-weight report, `⟨O_i⟩`, correlators, entanglement profile, overlaps, `⟨ψ|H|ψ⟩` for an MPO |
| `dmrg.py` | two-site DMRG with cached incremental environments and a warm-started Lanczos local solve; **deflation** (`ortho=[…]`) for excited states in the orthogonal sector |
| `tebd.py` | 2nd-order Trotter gates, real + imaginary time, per-step observable sampling, imaginary-time ground state |
| `validate.py` | the phase-gated checks (below) |

## Validate

```sh
python validate.py --phase all          # ~20 min
python validate.py --phase 2 --quick
```

| phase | checks | result |
|---|---|---|
| 1 | MPS↔ED round-trip fidelity → machine precision at exact `χ`; left-canonical isometry `ΣA†A = I`; entropy matches ED | **PASS** |
| 2 | DMRG = ED to 1e-8 (TFIM, three `g`); energy/site → the free-fermion `e_0(g=1)`; discarded weight bounds the energy error | **PASS** |
| 3 | spin-½ energy/site → Bethe ansatz `¼−ln2`; **Haldane gap → 0.41 J** via a spin-½-capped chain (singlet–triplet splitting, 1/N extrapolation); string order ≫ Néel order; fractional edge spins under a weak field; 4-fold quasi-degenerate open-chain manifold (ED) | **PASS** |
| 4 | TFIM critical block entropy vs the conformal `ln[(N/π)sin(πℓ/N)]` → **central charge `c = ½`**; correlation length grows with `χ` (finite-entanglement scaling) | **PASS** |
| 5 | TEBD quench: correlation-front velocity ≈ Lieb–Robinson `2 min(g,1)`; linear entanglement growth; energy conserved; imaginary-time TEBD = DMRG ground state | **PASS** |
| 6 | 1D Hubbard **Mott plateau width vs the Lieb–Wu charge gap**; spin correlations outlive charge correlations (spin–charge separation); **H-chain dissociation** — DMRG reaches a flat separated-atom limit where restricted HF overshoots | **PASS** |

Media (`media/`): compression curve, DMRG convergence, Haldane-gap extrapolation,
criticality fit, the TEBD correlation **light-cone movie** + entanglement growth,
and the Hubbard Mott / spin–charge / H-chain-dissociation panel.

## Method notes

* **MPO as a finite-state machine** — a small `D_W×D_W` operator-valued matrix per
  site (`D_W = 3` TFIM, `5` Heisenberg, `6` Hubbard). The Hubbard hopping carries
  the Jordan–Wigner string `F` on the left-site operator; the MPO is verified
  Hermitian and spectrum-matched to the Fock-space ED.
* **DMRG environments** are grown incrementally (`O(N χ³ d² D_W)` per half-sweep),
  the local eigenproblem is a matrix-free `LinearOperator` warm-started from the
  current two-site tensor, and the Lanczos tolerance is loosened on early sweeps.
* **Excited states** use deflation: previously converged MPS are projected into the
  current gauge via overlap environments and a penalty `w|φ⟩⟨φ|` is folded into the
  effective Hamiltonian — "DMRG in the orthogonal sector".
* **Haldane gap**: an open spin-1 chain has a 4-fold quasi-degenerate ground
  manifold (the spin-½ edge modes), so the bulk gap is *not* `E_1−E_0` there.
  Capping the chain with a spin-½ at each end binds the edge modes and the bulk
  gap reappears as a clean singlet–triplet splitting that extrapolates to 0.41 J.
* The periodic-boundary / infinite (iDMRG, iTEBD), 2D (PEPS) and finite-`T`
  (purification, METTS) extensions are noted in the plan, not implemented.
