# Perturbation & Basis Methods — Design Plan

## Context

`Quantum Mechanics/Perturbation_and_Basis_Methods/` currently holds 4 loose legacy
notebooks moved there during the quantum-projects reorg. Inspecting them found
real, concrete problems — not just style issues:

- `1D Pertubaton Theory.ipynb` — computes 1st-order energy/wavefunction
  corrections in the infinite-square-well basis, but has a genuine
  forward-reference bug: cell 2 uses `Hp` (the perturbation array) before it's
  ever defined (`Hp` isn't created until cell 3). It only "works" if cells are
  run out of order. There is also no validation anywhere — the corrected
  wavefunction/energy are never checked against anything independent.
- `Particle in a 1D box.ipynb` / `Simple Quantum Harmonic Oscillator.ipynb` —
  both decompose a Gaussian wavepacket into an eigenbasis (box, then harmonic
  oscillator) via a Python loop calling `scipy.integrate.quad` once per mode
  (slow, doesn't scale), and both evolve the coefficients with
  `exp(+1j*E*t)` — the wrong sign; the correct time evolution is
  `exp(-1j*E*t)`. As written, "forward in time" plays the dynamics backwards.
- `Hydrogen Atom.ipynb` — calls `scipy.special.sph_harm`, which **no longer
  exists** in the installed scipy (1.17.1); confirmed directly (`AttributeError:
  module 'scipy.special' has no attribute 'sph_harm'. Did you mean:
  'sph_harm_y'?`). The replacement, `sph_harm_y(n, m, theta, phi)`, also swapped
  argument order and the theta/phi convention (old: theta=azimuthal,
  phi=polar; new: theta=polar, phi=azimuthal) — the legacy code's variable
  naming (`THE` spanning 0-pi fed into the old function's "theta" slot, which
  expected 0-2pi) suggests it likely had this backwards even under the old API,
  just invisible because it only ever plotted `m=0`. Beyond that, the notebook
  is purely a static angular-shape plot — no radial equation, no dynamics, not
  really "the hydrogen atom" yet.

This plan consolidates all four into one project built around a single theme —
representing and evolving quantum states in an **analytic eigenbasis**, the
natural complement to `TDSE_Solver`'s grid/spectral approach — plus perturbation
theory built directly on that basis machinery, in the same house style as
`HF_solver`/`MAC_Grid_Solver`/`TDSE_Solver` (own `*_Plan.md`, procedural modules,
phase-gated build with a validation gate per phase, MP4s to `media/`).

## Numerical approach

**`basis.py`** — one generic `BasisSet` (eigenfunctions sampled on a shared
grid + their energies) covering both the box and harmonic-oscillator eigenbases
through shared projection/reconstruction/evolution code. This directly fixes
both real bugs above: projection becomes one vectorized Simpson-rule integral
(`scipy.integrate.simpson`) instead of a per-mode `quad` loop, and evolution
uses the correct `exp(-i*E*t)`.

**`hydrogen.py`** — real hydrogen wavefunctions `psi_nlm(r,theta,phi) =
R_nl(r)*Y_l^m(theta,phi)`, using the *current* `scipy.special.sph_harm_y` (its
docstring gives theta=polar/colatitude in `[0,pi]`, phi=azimuthal in
`[0,2*pi]` — verified directly, not assumed) and `scipy.special.eval_genlaguerre`
for the radial part, `E_n=-1/(2n^2)` (atomic units, Z=1), plus the real
`2p_x`/`2p_y`/`2p_z` combinations. Replaces the broken/cosmetic legacy notebook
with genuinely working, checkable physics.

**`perturbation.py`** — 1st/2nd-order non-degenerate perturbation theory
(energy and wavefunction corrections) built on any `BasisSet`, validated
against direct diagonalization of the truncated `H0+H'` matrix in the same
basis (`numpy.linalg.eigh`) — the legacy notebook never validated its result
against anything. The same matrix-diagonalization idea, restricted to a
degenerate subspace, gives degenerate perturbation theory, used for the
hydrogen Stark effect (Phase 4).

## File layout

```
Quantum Mechanics/Perturbation_and_Basis_Methods/
    Perturbation_and_Basis_Methods_Plan.md
    basis.py                    # Phase 1: BasisSet, box_basis(), harmonic_basis(), project(), reconstruct(), evolve()
    hydrogen.py                   # Phase 1: hydrogen radial/angular/full wavefunctions, energies, real p-orbitals
    perturbation.py                 # Phase 3-4: matrix elements, 1st/2nd order corrections, exact diagonalization, degenerate PT
    Validation.ipynb                  # Phase 1-2 checks
    Nondegenerate_Perturbation_Theory.ipynb  # Phase 3
    Hydrogen_Stark_Effect.ipynb                # Phase 4
    media/
```

Legacy notebooks archived to `Perturbation_and_Basis_Methods/legacy/` once the
replacement content is validated, matching `TDSE_Solver/legacy/`.

## Phases

Each phase stops for review before moving on (same workflow as `TDSE_Solver`).

**Phase 1 — Basis sets & hydrogen wavefunctions** (`basis.py`, `hydrogen.py`)
- `box_basis(x, L, num_modes)`: `phi_n=sqrt(2/L)*sin(n*pi*x/L)`, `E_n=(n*pi/L)^2/2`.
- `harmonic_basis(x, omega, num_modes)`: Hermite-function eigenstates, `E_n=omega*(n+1/2)`.
- `project`/`reconstruct`/`evolve`: vectorized Simpson-rule projection, correct `exp(-i*E*t)` evolution.
- `hydrogen.py`: `R_nl(r)` via `eval_genlaguerre`, `psi_nlm` via `sph_harm_y`, `E_n=-1/(2n^2)`; real `2p_x`/`2p_y`/`2p_z`.
- Validate: box/HO eigenfunctions are orthonormal (Gram matrix ~ identity) to numerical-integration precision; their energies match the analytic formulas (already independently confirmed in `TDSE_Solver/stationary_states.py`'s Phase 4 validation — good cross-project consistency without any shared code); hydrogen radial functions are normalized and orthogonal across `n` for fixed `l`; hydrogen energies match `E_n=-1/(2n^2)`.

**Phase 2 — Basis expansion & time evolution** (`basis.py`, exercised in `Validation.ipynb`)
- Project a Gaussian wavepacket onto each basis, reconstruct, evolve in time.
- Validate: truncation error (`1-sum(|c_n|^2)`) shrinks as more modes are kept; box-basis reconstruction matches the target to that truncation-limited tolerance; for the harmonic oscillator, a Gaussian with width exactly matched to the oscillator's natural length (`sigma=1/sqrt(omega)`) is an exact coherent state, so `<x>(t)` must follow the *exact classical trajectory* `x0*cos(omega*t)` with no spreading — a strong closed-form check with no free parameters.

**Phase 3 — Non-degenerate perturbation theory** (`perturbation.py`, `Nondegenerate_Perturbation_Theory.ipynb`)
- Box basis plus a localized bump perturbation (same physical idea as the legacy notebook, now fixed and validated).
- 1st-order energy/wavefunction correction, 2nd-order energy correction.
- Validate: across a range of perturbation strengths `lambda`, compare PT predictions to direct diagonalization of the truncated `H0+H'` matrix; confirm PT error shrinks as `lambda->0` at roughly the expected order, and show explicitly (not hide) that PT breaks down once `lambda` is no longer small.

**Phase 4 — Degenerate perturbation theory: hydrogen Stark effect** (`perturbation.py`, `Hydrogen_Stark_Effect.ipynb`)
- The 4-fold-degenerate `n=2` hydrogen shell (`2s`, `2p_x`, `2p_y`, `2p_z`) under a uniform field, `H'=E_field*z`.
- Compute the 4x4 `H'` matrix numerically from `hydrogen.py`'s actual wavefunctions (no hardcoded textbook constants) and diagonalize within the degenerate subspace.
- Validate: 2 of the 4 states shift by `+/-3*E_field` (atomic units, `a0=1`) and the other 2 (`2p_x`, `2p_y`) are unshifted — the standard linear Stark effect result for hydrogen `n=2`, reached by direct computation rather than assumed; the shifted eigenvectors should come out proportional to `(2s +/- 2p_z)/sqrt(2)`.
- Visualize the energy-level splitting vs. field strength (a Stark diagram) to `media/`.

## Progress

- [x] Phase 1 — `basis.py`, `hydrogen.py` — done, 9/9 checks pass
- [x] Phase 2 — basis expansion & time evolution validation — done, 5/5 checks pass; coherent-state <x>(t) matches exact classical trajectory to 4.8e-9
- [x] Phase 3 — `Nondegenerate_Perturbation_Theory.ipynb` — done, 5/5 checks pass; error scales as V0^2 (1st order) / V0^3 (2nd order) exactly as theory predicts, breakdown at strong coupling shown explicitly
- [x] Phase 4 — `Hydrogen_Stark_Effect.ipynb` — done, 5/5 checks pass; computed splitting matches the textbook +/-3 (atomic units) result and (2s+/-2p_z)/sqrt2 eigenvectors exactly

All 4 phases complete. 28/28 validation checks pass across Validation.ipynb + the two benchmark notebooks.

## Verification

1. Phase 1: orthonormality and energy checks pass to numerical-integration precision.
2. Phase 2: truncation error shrinks with more modes kept; coherent-state `<x>(t)` matches the exact classical trajectory with no free parameters.
3. Phase 3: perturbation theory matches exact diagonalization for small `lambda` with the correct order-of-`lambda` error scaling, and visibly diverges for large `lambda`.
4. Phase 4: computed Stark splitting matches the analytic `+/-3*E_field` result and the correct eigenvector structure.
