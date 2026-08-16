# TDSE Solver — Design Plan

## Context

`Quantum Mechanics/TDSE_Solver/` currently holds 5 loose legacy scripts moved there during the quantum-projects reorg (grouped because they all touch time-dependent wavefunction dynamics). Inspecting them turned up real problems, not just cosmetic ones:

- `1D Schrodinger Equation (Not my implementation).py` — genuine time propagation via a **dense matrix exponential** (`scipy.sparse.linalg.expm` on the full Hamiltonian). Works at N=256 in 1D but this approach doesn't scale to 2D/3D at all (the matrix is N^d × N^d).
- `2D Schrodinger Equation.py` / `2D Schrodinger Equation Solutions.py` — despite the "Schrödinger Equation" framing, these don't do time evolution at all: they build a sparse 2D Hamiltonian and call `scipy.sparse.linalg.eigs` for the lowest eigenstates. The first of the pair is also a broken draft (its `hamiltonian()` builds a 1D-sized operator despite being labeled 2D). TensorFlow is imported only to re-do a dense `expm` that's never actually reached on this path.
- `3D Schrodinger Equation Solver.py` — same eigs-based approach extended to 3D via a Kronecker-sum Laplacian, capped at N=32 per axis (32³ unknowns) because dense/sparse diagonalization at higher resolution isn't practical this way. Also imports TensorFlow unnecessarily.
- `Improved 2D Schrodinger Equation (Not my implenetation).py` — genuine Crank–Nicolson time propagation for a 2D double-slit, but the (Ni × Ni) system matrix is built by an explicit Python `for` loop over every interior grid point and stored as a **dense** NumPy array before converting to sparse — correct method, wasteful and non-scaling construction.

Net effect: no dimension-generic architecture, an unused TensorFlow dependency, two scripts that are secretly stationary-state solvers mislabeled as time-dependent ones, and no validation against any analytic result anywhere in the family.

This plan consolidates all of it into one real, dimension-generic time-dependent Schrödinger equation (TDSE) solver, in the same house style as `Fluid Mechanics/MAC_Grid_Solver/` and `HF_solver/`: its own `*_Plan.md`, plain procedural modules (no classes), physics/numerics derived inline, phases gated by concrete validation checks, and MP4 animations saved to `media/`. Per the user's request, the goal is one shared architecture that the *other* two new quantum projects (`Bound_States_1D/`, `Perturbation_and_Basis_Methods/`) can also draw on later (e.g. a shared sparse-Hamiltonian-assembly convention), though this plan only scopes `TDSE_Solver/` itself.

## Numerical approach

**Units**: atomic units (ħ = m = 1), so `iψ_t = Hψ = (-½∇² + V)ψ` is the whole equation to discretize — mirrors the `Re`-based non-dimensionalization convention already used in `MAC_Grid_Solver`.

**Primary method — symmetric (Strang) split-operator / split-step spectral method** (Feit, Fleck & Steiger 1982):

```
e^{-iHΔt} ≈ e^{-iVΔt/2} · e^{-iTΔt} · e^{-iVΔt/2}
```

2nd-order accurate in Δt, unconditionally stable, and exactly unitary (norm-conserving to machine precision) whenever no absorbing layer is active. This is the standard algorithm for numerical quantum dynamics and is what actually gives one shared architecture across 1D/2D/3D:

- **Potential half-step**: elementwise multiply by `exp(-iVΔt/2)` in real space. The exact same line of code works in 1D/2D/3D since `V` is just an ndarray of the grid's shape.
- **Kinetic full-step**: `T = -∇²/2` is diagonal in the right spectral basis, so it's applied as a transform → elementwise phase multiply → inverse transform, generic over dimension:
  - **Periodic / open domain** → FFT (`numpy.fft.fftn`/`ifftn`), eigenvalue `|k|²/2` with `k` from `fftfreq` per axis, combined as `kx²+ky²+kz²` via broadcasting.
  - **Hard-wall (ψ=0 at the boundary, infinite box)** → type-I discrete sine transform (`scipy.fft.dstn`/`idstn`), eigenvalue `(nπ/L)²/2` per mode per axis.
  
  Both cases share one function shaped like `MAC_Grid_Solver`'s `operators.laplacian_matrix(..., bc=...)`: one generic operator parameterized by a boundary-condition choice, reused everywhere instead of re-derived per notebook.

**Open/scattering boundaries** (tunneling, double-slit): add a **complex absorbing potential (CAP)** — a smooth imaginary term `-iη(r)` switched on near the domain edges — to `V` before the potential half-step, so outgoing probability flux is absorbed instead of reflecting or periodically wrapping around. This is the standard companion to spectral propagation for scattering problems. It deliberately breaks exact norm conservation (the lost norm *is* the flux that exited), so validation must track norm loss as a diagnostic rather than require it stay at 1.

**Dropped from the legacy scripts**: TensorFlow entirely (plain `numpy`/`scipy.fft` covers the whole job, no dense `expm` needed); dense matrix-exponential propagation (doesn't scale past 1D); building a Crank–Nicolson system matrix via a Python loop over grid points.

**Kept, repurposed**: the `eigs`-based stationary-state machinery is genuinely useful, just not as the main propagator — it becomes a small validation/initial-condition utility (`stationary_states.py`), used to (a) generate a physically meaningful starting wavefunction and (b) provide the single strongest correctness test of the propagator: an eigenstate should evolve by nothing but the exact phase `e^{-iEt}`.

## File layout

```
Quantum Mechanics/TDSE_Solver/
    TDSE_Solver_Plan.md
    grid.py                   # Phase 1: ndim-generic position + reciprocal(k)/mode grids
    propagator.py              # Phase 1-2: kinetic step (FFT or DST, chosen by boundary) + potential half-step + full Strang step
    potentials.py               # Phase 2-3: rectangular barrier, double-slit barrier, soft-Coulomb well, harmonic well, CAP layer
    observables.py               # Phase 3: probability density, norm, <x>/<p>, region-integrated transmission probability
    stationary_states.py          # Phase 4: sparse ndim Kronecker-sum Hamiltonian + eigsh lowest-k (validation/initial-condition tool)
    Validation.ipynb               # Phases 1-4 internal checks
    Tunneling_1D.ipynb              # Phase 5: rectangular barrier, transmission probability vs analytic formula
    Double_Slit_2D.ipynb             # Phase 6: interference pattern vs analytic fringe spacing
    Bound_Wavepacket_3D.ipynb         # Phase 7: soft-Coulomb wavepacket, energy-conservation check
    media/
```

## Phases

Each phase stops for review before moving on (same workflow as `MAC_Grid_Solver`).

**Phase 1 — Grid & kinetic propagator** (`grid.py`, core of `propagator.py`)
- ndim-generic position grids and reciprocal/mode grids for both the FFT and DST cases.
- Kinetic step for both boundary choices.
- Validate: applying the kinetic propagator to a plane wave (periodic case) or box eigenmode (hard-wall case) reproduces the exact analytic phase `e^{-i k²Δt/2}` to machine precision; norm is conserved to machine precision with `V=0`.

**Phase 2 — Potential half-step & full split-operator step** (`propagator.py`, `potentials.py`)
- Assemble the full Strang step: `V/2 → T → V/2`.
- Validate: a free Gaussian wavepacket's width growth matches the closed-form `σ(t) = σ0·√(1+(t/2σ0²)²)`; halving Δt shows the expected ~2nd-order convergence toward that curve; propagating a stationary eigenstate (from Phase 4, built early for this test) changes only by the phase `e^{-iEt}`, not shape.

**Phase 3 — Observables & absorbing boundary** (`observables.py`, CAP in `potentials.py`)
- Probability density, norm, `<x>`, `<p>`, region-integrated transmission probability; CAP layer construction and norm-loss diagnostic.
- Validate: a wavepacket absorbed by a CAP at the domain edge shows no spurious reflection back into the interior; norm loss is ~0 with the CAP off (isolates the CAP, not the propagator, as the loss source).

**Phase 4 — Stationary states utility** (`stationary_states.py`)
- Sparse ndim Kronecker-sum Hamiltonian (same Laplacian-assembly idea as `MAC_Grid_Solver`, adapted for a scalar field), `eigsh` for the lowest k eigenstates.
- Validate: eigenvalues for an infinite square well and a harmonic well match their known analytic formulas.

**Phase 5 — Benchmark: 1D tunneling** (`Tunneling_1D.ipynb`)
- Rectangular barrier; reuse the legacy script's (already-correct) analytic transmission-probability formula as ground truth.
- Measure transmission probability (long-time `|ψ|²` integrated past the barrier, with a CAP acting as an absorbing detector) across a range of incident energies and compare to the analytic curve.
- Animate probability density vs. time to `media/` as MP4.

**Phase 6 — Benchmark: 2D double-slit** (`Double_Slit_2D.ipynb`)
- Two-slit barrier (cleaned-up version of the legacy `double_slit_potential_barrier` shape), incident Gaussian wavepacket.
- Compare measured far-field fringe spacing to the analytic double-slit formula `Δy ≈ λL/d` using the wavepacket's central de Broglie wavelength `λ = 2π/k0`.
- Animate the interference pattern developing to `media/` as MP4.

**Phase 7 — Benchmark: 3D bound wavepacket** (`Bound_Wavepacket_3D.ipynb`)
- Soft-Coulomb-like well (from the legacy script), wavepacket initialized off-center (or as a superposition of two low eigenstates from Phase 4) so it actually moves.
- Validate via conservation of `<H>` over time (no explicit time dependence ⇒ exactly conserved) — the primary quantitative check, since there's no simple closed-form trajectory to compare against directly.
- Visualize a slice/isosurface animation to `media/` as MP4.

## Progress

- [x] Phase 1 — `grid.py`, kinetic propagator core — done, 10/10 checks pass
- [x] Phase 2 — potential half-step + full split-operator step — done, 6/6 checks pass, confirmed 2nd-order convergence
- [x] Phase 3 — `observables.py` + absorbing boundary — done, 5/5 checks pass, CAP reflection/leakage ~1e-8
- [x] Phase 4 — `stationary_states.py` — done, 9/9 checks pass, confirmed 2nd-order FD convergence and correct 2D/3D degeneracies
- [x] Phase 5 — `Tunneling_1D.ipynb` — done, measured T(E) tracks analytic curve to within 0.009 (expected wavepacket energy-spread systematic); MP4 saved
- [x] Phase 6 — `Double_Slit_2D.ipynb` — done, fringe spacing matches analytic formula to 0.3% near center; MP4 saved
- [x] Phase 7 — `Bound_Wavepacket_3D.ipynb` — done, <H> conserved to 4e-7 relative; dipole-oscillation period matches independent eigensolver prediction to 0.95%; MP4 saved

All 7 phases complete. 46/46 validation checks pass across Validation.ipynb + the three benchmark notebooks.

## Verification

Each phase's checks build on the last — don't trust a later phase until the earlier ones pass:

1. Phase 1: kinetic-step phase and norm checks pass to machine precision.
2. Phase 2: free-particle spreading matches the analytic formula; convergence order is ~2; eigenstate phase test passes.
3. Phase 3: CAP absorbs cleanly with no interior reflection; norm loss is attributable only to the CAP.
4. Phase 4: eigenvalues match analytic square-well/harmonic-well formulas — this also retroactively validates Phase 2's eigenstate-phase test.
5. Phase 5: measured 1D transmission probability tracks the analytic tunneling curve across incident energies.
6. Phase 6: interference fringe spacing matches the analytic double-slit formula; pattern is visually a clean diffraction pattern.
7. Phase 7: `<H>` stays constant (to numerical tolerance) over the full 3D run.
