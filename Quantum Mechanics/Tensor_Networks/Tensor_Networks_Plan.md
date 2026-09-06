# Tensor Networks / DMRG — Design Plan

## Context

There is one regime every solver in this repo fails: **strong correlation in
extended systems**. Mean-field methods (`HF_solver/`, `Diatomic_HF_solver/`,
`Molecular_DFT/`) are qualitatively wrong for the Hubbard model, stretched bonds,
Mott insulators, and frustrated magnets. Exact diagonalization dies at ~16–20
spins (`2^N` Hilbert space). `Perturbation_and_Basis_Methods/` only works when a
perturbation is small.

**Matrix Product States (MPS)** compress the ground state of a 1D, short-range-
entangled system into `O(N χ² d)` parameters (bond dimension `χ`, physical
dimension `d`). **DMRG** variationally optimizes an MPS; **TEBD** evolves one in
time. Pure `numpy`/`scipy` — SVD, sparse `eigsh`, `tensordot` — a few hundred
lines, no external tensor-network library (matches the repo's "no pyscf, build
it from scratch" ethos).

Scope: **1D lattice models** — transverse-field Ising, Heisenberg (spin-½ and
spin-1), the 1D Hubbard model, and Hydrogen chains in a minimal basis. Open
boundary conditions. This is the strongly-correlated-physics complement to the
mean-field solvers, and it cross-checks `Quantum_Monte_Carlo/`'s dissociation
benchmarks.

## What this enables (applications)

- **Ground states of correlated 1D systems** — energy per site, local
  observables, correlation functions `⟨S_i · S_j⟩`, structure factors — to
  near-exact accuracy far beyond ED's reach (`N = 100+` sites).
- **The Haldane gap.** The spin-1 Heisenberg chain has a gapped, topologically
  nontrivial ground state with fractionalized spin-½ edge modes — a phase with
  **no mean-field description at all**. DMRG sees it directly (the gap `≈ 0.41 J`,
  the 4-fold-degenerate open-chain edge states, the string order parameter).
- **Quantum phase transitions and CFT diagnostics.** At the transverse-field
  Ising critical point the entanglement entropy of a block scales as
  `S(L) ≈ (c/6) log L`; fitting it extracts the **central charge** `c = ½` — a
  conformal-field-theory quantity read straight off the wavefunction.
- **The Mott transition and spin–charge separation.** The 1D Hubbard model:
  at half filling any `U > 0` opens a charge gap (Mott insulator) while spin
  excitations stay gapless; after a quench, spin and charge correlations spread
  at **different velocities** — the signature that the electron has "split".
  This is precisely the physics LDA gets most wrong.
- **Bond dissociation done right.** A chain of H atoms (or stretched H2) in a
  minimal basis is a standard strong-correlation benchmark. DMRG stays correct
  where restricted HF and LDA fail catastrophically — a direct, quantitative
  cross-check against `Quantum_Monte_Carlo/`'s dissociation curve and the DFT
  trilogy's known failure modes.
- **Real-time quench dynamics (TEBD).** Light-cone spread of correlations
  (Lieb–Robinson bound), entanglement-entropy growth after a quench — and the
  point where that growth forces `χ` past what is tractable, a built-in
  demonstration of the method's own boundary.

## Numerical approach

**MPS**: a chain of rank-3 tensors `A^{s_i}_{a,b}`, kept in mixed canonical form
via successive SVDs. Observables: `⟨O_i⟩` by contracting one site against its
canonical environment (trivial in canonical form); correlators by transfer-matrix
contraction between the two sites.

**MPO (Hamiltonian)**: finite-state-machine construction for a sum of on-site and
nearest-neighbor terms — a small `D_W × D_W` operator-valued matrix per site
(`D_W = 3` for TFIM, `5` for Heisenberg, `6` for Hubbard with Jordan–Wigner).

**DMRG (two-site)**:
1. Sweep left→right and back. At each bond, contract the left and right
   environment blocks with the two local MPO tensors to form the effective
   two-site Hamiltonian as a `LinearOperator`.
2. Solve the local ground state with `scipy.sparse.linalg.eigsh` (Lanczos),
   warm-started from the current MPS.
3. SVD the optimized two-site tensor, truncate to `χ` (or to a singular-value
   cutoff), keep the discarded weight as the error estimate.
4. Update environments. Iterate sweeps until `ΔE` per sweep `< tol`.

**TEBD**: Trotter-split `e^{−iHδt}` (or `e^{−Hτ}` for imaginary time) into
even-bond and odd-bond two-site gates; apply gate → SVD → truncate → move on.
2nd-order Trotter; `δt` and `χ` convergence checked.

**Fermions**: Jordan–Wigner strings folded into the MPO / gates so the Hubbard
model is handled without explicit anticommutation bookkeeping in the MPS.

**Exact-diagonalization oracle** (`ed.py`): full sparse Hamiltonian for `N ≤ 14`,
`scipy.sparse.linalg.eigsh` — the ground-truth check for every DMRG result, and
independently useful/pedagogical.

**Not attempted**: 2D tensor networks (PEPS) — the contraction is exponentially
hard and out of scope; periodic-boundary MPS (open boundaries only, far
better-conditioned); infinite/translation-invariant MPS (iDMRG/iTEBD) — noted as
the natural extension; finite-temperature MPS (purification / METTS) — noted, not
scoped.

## Upstream changes to existing solvers

**None.** This is a clean new corner of the repo — a different paradigm (lattice
models, not continuum grids) with no shared code. Cross-references only:

- Phase 6's Hydrogen-chain / stretched-H2 results are plotted against
  `Quantum_Monte_Carlo/`'s dissociation curve and `Molecular_DFT/` /
  `Diatomic_HF_solver/`'s LDA bond curves (data comparison in a notebook, no code
  dependency).

## File layout

```
Quantum Mechanics/Tensor_Networks/
    Tensor_Networks_Plan.md
    ed.py              # Phase 1: sparse exact diagonalization oracle, N <= 14
    mps.py             # Phase 1: MPS class, canonical form, <O_i>, correlators, entanglement entropy
    mpo.py             # Phase 2: MPO construction for TFIM / Heisenberg / Hubbard
    dmrg.py            # Phase 2-3: two-site variational sweep
    tebd.py            # Phase 5: Trotter gates, real + imaginary time
    models.py          # lattice Hamiltonian parameters, JW strings
    Validation.ipynb              # Phase 1-2: MPS round-trip, DMRG vs ED
    Spin_Chains.ipynb             # Phase 3-4: Heisenberg, Haldane gap, criticality
    Quench_Dynamics.ipynb         # Phase 5: TEBD light cone, entanglement growth
    Hubbard_and_Hydrogen.ipynb    # Phase 6: Mott physics, H-chain dissociation
    media/
```

## Phases

**Phase 1 — ED oracle + MPS basics** (`ed.py`, `mps.py`)
- Sparse `H` for TFIM / Heisenberg (`N ≤ 14`); MPS class with canonicalization,
  local expectation values, overlaps, bipartite entanglement entropy.
- Validate: compress an ED ground state into an MPS and back — the fidelity
  `1 − |⟨ψ_MPS|ψ_ED⟩|²` falls monotonically with `χ` and hits ~1e-12 once `χ`
  reaches the exact Schmidt rank; canonical-form identities
  (`Σ A†A = I`) hold to ~1e-13.

**Phase 2 — MPO + DMRG for the transverse-field Ising chain** (`mpo.py`, `dmrg.py`)
- Two-site DMRG; sweep to convergence.
- Validate: ground-state energy vs ED (`N = 12`) to ~1e-8; vs the exact
  analytic infinite-chain energy per site
  `e_0 = −(1/π)∫₀^π √(1 + g² − 2g cos k) dk / 2` (Pfaffian solution) as
  `N → 100`; discarded weight tracks the actual energy error.

**Phase 3 — Heisenberg chains: spin-½ and the Haldane gap** (`Spin_Chains.ipynb`)
- Spin-½: energy per site vs the Bethe-ansatz value `¼ − ln 2 ≈ −0.4431`.
- Spin-1: measure the **Haldane gap** (energy of the first excited state via a
  second DMRG run in the orthogonal sector, or a small finite-`N` extrapolation)
  vs the accepted `0.4105 J`; show the 4-fold quasi-degenerate open-chain ground
  manifold (the spin-½ edge modes) and a nonzero **string order parameter** with
  vanishing Néel order.

**Phase 4 — Criticality and the central charge** (`Spin_Chains.ipynb`)
- TFIM at `g = 1`.
- Validate: block entanglement entropy `S(L)` vs `log[(N/π) sin(πL/N)]` is linear
  with slope `c/6`, giving `c = 0.50 ± 0.02`; the correlation length extracted
  from `⟨σ^x_i σ^x_j⟩` diverges with `χ` (finite-entanglement scaling with the
  known exponent).

**Phase 5 — TEBD real-time dynamics** (`tebd.py`, `Quench_Dynamics.ipynb`)
- Global quench in the TFIM (`g: g_0 → g_1` at `t = 0`).
- Validate: connected correlations `⟨σ^z_i σ^z_j⟩_c` spread inside a light cone
  with the Lieb–Robinson velocity `v = 2 min(g, 1)` (max group velocity of the
  Bogoliubov dispersion); entanglement entropy grows **linearly** in time until
  truncation error blows up — the method's wall, shown explicitly. Imaginary-time
  TEBD reproduces Phase 2's DMRG ground-state energy independently. MP4 of the
  correlation light cone.

**Phase 6 — Hubbard model + Hydrogen chains** (`Hubbard_and_Hydrogen.ipynb`)
- 1D Hubbard at half filling: charge gap opens for any `U > 0` (Mott insulator)
  while the spin sector stays gapless — validate the charge gap vs the exact
  Lieb–Wu Bethe-ansatz result at a couple of `U` values; show spin and charge
  correlations spreading at different velocities after a quench.
- Hydrogen chain / stretched H2 in a minimal (1 orbital/atom) basis: DMRG energy
  vs bond length, plotted against `Quantum_Monte_Carlo/`'s dissociation curve and
  the DFT trilogy's LDA bond curves — DMRG dissociates to the correct
  separated-atom limit where restricted HF and LDA do not. Figure to `media/`.

## Progress

- [ ] Phase 1 — ED oracle, MPS class, round-trip test
- [ ] Phase 2 — MPO + DMRG for TFIM vs ED and analytic
- [ ] Phase 3 — Heisenberg spin-½ / spin-1, Haldane gap, edge modes
- [ ] Phase 4 — TFIM criticality, central charge `c = ½`
- [ ] Phase 5 — TEBD quench, light cone, entanglement growth
- [ ] Phase 6 — Hubbard Mott physics + Hydrogen-chain dissociation

## Verification

1. Phase 1: MPS↔ED fidelity → machine precision at exact `χ`; canonical
   identities hold.
2. Phase 2: DMRG energy matches ED (1e-8) and the analytic infinite-chain
   result; discarded weight bounds the error.
3. Phase 3: spin-½ energy vs Bethe ansatz; spin-1 Haldane gap `≈ 0.41 J` with
   edge modes and string order.
4. Phase 4: entanglement entropy slope gives `c = ½`.
5. Phase 5: light-cone velocity `= 2 min(g,1)`; linear entanglement growth;
   imaginary-time TEBD agrees with DMRG.
6. Phase 6: Hubbard charge gap vs Lieb–Wu; Hydrogen chain dissociates correctly
   vs QMC and against the failing mean-field curves.
