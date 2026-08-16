# Molecular DFT (full 3D real-space) — Design Plan

## Context

Third and final phase of this session's DFT initiative: `HF_solver/` built
an atomic (radial-grid) Kohn-Sham LDA solver, `Diatomic_HF_solver/` built a
two-center (prolate-spheroidal-grid) version for diatomics. Both exploited
an exact symmetry (spherical, then axial) to reduce a 3D problem to 1D/2D
finite-difference grids. A general molecule has no such symmetry to
exploit -- this project drops the symmetry trick entirely and solves the
full 3D problem on a uniform Cartesian grid, the architecture every real
production DFT code (VASP, Quantum ESPRESSO, Octopus, ABINIT, ...)
actually uses for exactly this reason.

**Scope decision, made up front**: this is genuinely a different, much
larger numerical undertaking than the previous two projects (a 3D grid
with `N` points per axis has `N^3` total points, vs. `N` or `N^2` before --
direct/shift-invert sparse eigensolvers and sparse-direct Poisson solves,
both used successfully in the earlier projects, do not scale to this).
Initial build targets **all-electron light atoms/molecules only** (H, He,
H2) -- enough to validate the full pipeline (grid, kinetic operator,
Poisson solver, XC, SCF loop, iterative eigensolver) end-to-end against
exact/already-validated references. Pseudopotentials (needed for anything
heavier, e.g. O for H2O) are explicitly out of scope for this initial
build -- a real, separate piece of physics/numerics (core-electron
removal, an appropriately screened effective ion potential), not a small
extension, and flagged as future work rather than attempted speculatively.

## Numerical approach

**Grid and boundary conditions**: a uniform 3D Cartesian grid inside a
cubic box, with **periodic boundary conditions** (the "supercell"
approach every plane-wave-adjacent real-space DFT code uses for isolated
systems) -- the molecule sits in vacuum padding large enough that its
density has decayed to negligible amplitude before reaching the box edge,
so periodic images barely interact. This is a real, standard, well
-documented approximation (not a shortcut being invented here), with a
known, checkable failure mode (box too small) -- convergence with box size
is tested explicitly, not assumed.

**Kinetic energy and Poisson solve, both via FFT**: periodic boundary
conditions make the Laplacian and the Poisson equation both diagonal in
Fourier space (`-nabla^2 psi -> |G|^2 * psi_G`, same `G`-vectors for both),
so `scipy.fft`'s 3D FFT gives an exact (to floating-point precision, not
a finite-difference approximation), `O(N log N)` implementation of each --
reusing the identical grid/`G`-vector machinery for both operators.
Considered reusing `Electromagnetism/Poisson_Solver/poisson.py` (sparse
finite-difference Laplacian + direct sparse solve) first, but its
`spsolve`-based approach is 2D-only in practice -- 3D sparse-direct solves
suffer far worse fill-in and would be prohibitively expensive at any grid
size worth using here, whereas FFT is `O(N log N)` regardless of
dimension. Not reused; a fresh FFT-based approach is used instead (see
`fft_ops.py`).

**Eigensolver**: a 3D grid Hamiltonian is far too large for the
shift-invert direct sparse factorization the atomic/diatomic solvers used
(that approach's cost scales badly with 3D fill-in, same reasoning as the
Poisson solve above). Instead, `H` is applied as a function (kinetic via
FFT + potential via pointwise multiply in real space) through a
`scipy.sparse.linalg.LinearOperator`, solved with an iterative
eigensolver (`eigsh`/`lobpcg`) that only ever needs matrix-vector
products, never an explicit matrix -- the standard architecture for
large-scale real electronic-structure codes.

**Exchange-correlation**: Slater exchange (`ALPHA_LDA`) + PZ81 correlation,
copied from `HF_solver/potentials.py` (not imported cross-folder, per this
repo's self-contained-project convention, already followed by
`Diatomic_HF_solver`).

**SCF loop**: same shape as `HF_solver/scf.py`/`Diatomic_HF_solver/diatomic_driver.py`
-- build `V_eff` (nuclear + Hartree + XC) from the current density,
diagonalize, aufbau-fill occupations (from the first iteration only, then
held fixed, same stability rationale as both earlier solvers), rebuild
density from occupied orbitals, mix, check convergence.

## File layout

```
Quantum Mechanics/Molecular_DFT/
    Molecular_DFT_Plan.md
    grid3d.py                          # uniform 3D grid + FFT G-vector (frequency) utilities
    fft_ops.py                          # FFT-based kinetic energy operator + Poisson solver
    potentials3d.py                     # nuclear (softened Coulomb) potential + Slater/PZ81 XC on the 3D grid
    scf3d.py                            # SCF driver (build V_eff -> diagonalize -> mix -> converge)
    Kinetic_and_Poisson_Validation.ipynb  # Phase 1-2
    Atom_SCF_Validation.ipynb             # Phase 3-4: H, He
    H2_Validation.ipynb                   # Phase 5
    media/
```

## Phases

- [ ] **Phase 1 — 3D grid + FFT kinetic energy operator (`grid3d.py`, `fft_ops.py`)**.
  Uniform grid, `G`-vectors via `np.fft.fftfreq`-style construction. `apply_kinetic(psi)` = `ifftn(0.5*G2*fftn(psi))`.
  **Validation**: exact plane-wave eigenvalues `E_G = 0.5*|G|^2` for a periodic box with no potential -- the diatomic/atomic solvers' "zero potential reproduces the exact free/hydrogenic answer" acid test, here adapted to the periodic free-particle case.
- [ ] **Phase 2 — FFT Poisson solver (`fft_ops.py`)**. `V_H` from `rho` via `V_H_G = 4*pi*rho_G/|G|^2` (with the `G=0` component -- the average potential, undefined for a genuinely periodic charge distribution -- set to zero, standard convention). **Validation**: a smooth test charge distribution (e.g. a Gaussian) has a known closed-form potential; compare directly, and separately confirm box-size convergence (does the isolated-system approximation actually hold once the density is negligible at the boundary).
- [ ] **Phase 3 — Hydrogen atom SCF (`potentials3d.py`, `scf3d.py`)**. One electron, softened nuclear Coulomb potential (a real 1/r singularity isn't resolvable on a finite grid without softening, unlike the earlier solvers' radial-grid substitutions that handled the cusp analytically). **Validation**: exact `-0.5` Ha (with a small, convergence-checked deviation from grid softening/spacing -- not exact to machine precision the way the radial solver was, and that gap is itself part of what's validated/characterized here).
- [ ] **Phase 4 — Helium atom SCF**. Two electrons, exercises Hartree+XC together for the first time in 3D. **Validation**: compare directly against `HF_solver`'s already-validated radial LDA result for He (`-2.834` Ha) -- a project-native cross-check the earlier two solvers didn't have available, since this is the third leg of the same initiative.
- [ ] **Phase 5 — H2 molecule SCF (`H2_Validation.ipynb`)**. Two nuclei, no symmetry exploited this time (unlike `Diatomic_HF_solver`'s prolate-spheroidal reduction) -- genuinely tests the full, general 3D machinery. **Validation**: compare against `Diatomic_HF_solver`'s already-validated H2 LDA result (`-1.135544` Ha at `R=1.4` Bohr) -- another project-native cross-check -- and confirm a real bonding minimum falls out of a bond-length scan, same acid test as `Diatomic_HF_solver` Phase 5.

## Known limitations (anticipated, to confirm/document once built)

- **All-electron, light species only** -- no pseudopotentials in this build, so nothing heavier than He/H2 is in scope (see Scope decision above).
- **Periodic/supercell boundary conditions** -- an approximation to a truly isolated molecule, controlled by box-size convergence, not exact.
- **Softened nuclear cusp** -- unlike the radial/prolate-spheroidal solvers' analytic cusp handling, a 3D Cartesian grid needs the bare `1/r` singularity softened to stay resolvable, trading some accuracy for tractability; characterized empirically in Phase 3.
- **Same non-relativistic, LDA-level XC caveats as the rest of this initiative** -- inherited, not new.

## Verification

1. Run Phase 1's notebook: confirm exact free-particle plane-wave eigenvalues before trusting the kinetic operator for anything else.
2. Confirm Phase 2's Poisson solver against a closed-form Gaussian-charge potential, and check convergence with box size.
3. Run Phase 3's hydrogen SCF and confirm convergence near `-0.5` Ha, characterizing the softening-driven gap.
4. Run Phase 4's helium SCF and compare directly against `HF_solver`'s already-validated `-2.834` Ha LDA result.
5. Run Phase 5's H2 SCF (single geometry, then a bond-length scan) and compare against `Diatomic_HF_solver`'s already-validated `-1.135544` Ha result and known equilibrium bond length.
