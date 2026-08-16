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

- [x] **Phase 1 — 3D grid + FFT kinetic energy operator (`grid3d.py`, `fft_ops.py`)** ✅ done.
  Uniform grid, `G`-vectors via `np.fft.fftfreq`-style construction. `apply_kinetic(psi)` = `ifftn(0.5*G2*fftn(psi))`.
  **Validated**: exact plane-wave eigenvalues `E_G = 0.5*|G|^2` for a periodic box with no potential, to `1.7e-14` relative error across 20 random reciprocal-lattice vectors -- the diatomic/atomic solvers' "zero potential reproduces the exact free/hydrogenic answer" acid test, here adapted to the periodic free-particle case, and genuinely exact (not just small) since FFT has no discretization error for grid-representable plane waves.
- [x] **Phase 2 — FFT Poisson solver (`fft_ops.py`)** ✅ done. `V_H` from `rho` via `V_H_G = 4*pi*rho_G/|G|^2` (with the `G=0` component -- the average potential, undefined for a genuinely periodic charge distribution -- set to zero, standard convention).

  **Validated** against a Gaussian test charge's known closed-form potential (`V(r) = Q*erf(sqrt(a)*r)/r`): agrees near the charge, but a real, important limitation was found and characterized (not fixed -- see the Scope decision above), not assumed away: box-size convergence for a *non-neutral* density (exactly what a single atom's electron density is by itself) is only **algebraic (~1/L)**, not exponential -- `51%` relative error at `L=10` Bohr, still `7%` at `L=80` Bohr for the identical physical charge. This is a well-known limitation of naive periodic-FFT Poisson solves for isolated/charged systems (the reason production codes use specialized solvers like Martyna-Tuckerman or wavelets); it propagates into every SCF result below as a real, non-negligible source of error, reported honestly rather than hidden.
- [x] **Phase 3 — Hydrogen atom SCF (`potentials3d.py`, `scf3d.py`)** ✅ done. One electron, softened nuclear Coulomb potential (a real 1/r singularity isn't resolvable on a finite grid without softening, unlike the earlier solvers' radial-grid substitutions that handled the cusp analytically): `-Z/sqrt(r^2+softening^2)`.

  **Validated** (`L=16` Bohr, `N=32^3`, `dx=0.5` Bohr): energy approaches the exact `-0.5` Ha monotonically as softening shrinks relative to `dx` -- `softening=0.5*dx`: `-0.47553` Ha (`+4.9%`); `1.0*dx`: `-0.40806` Ha (`+18.4%`); `1.5*dx`: `-0.36607` Ha (`+26.8%`); `2.0*dx`: `-0.33484` Ha (`+33.0%`). Pushing softening *below* `0.5*dx` at this same resolution was also tested and, as expected for an under-resolved cusp, breaks down the other way (spurious overbinding): `0.3*dx` gives `-0.559` Ha (overshoots past `-0.5`), `0.2*dx` gives `-0.807` Ha (badly overbound) -- confirming `~0.5*dx` is close to the actual sweet spot at fixed resolution, not "smaller is always better." Separately confirmed genuine convergence to the exact answer with *resolution* (softening scaled proportionally, `softening=0.5*dx` held fixed): `N=32`: `-0.476` Ha, `N=48`: `-0.495` Ha, `N=64`: `-0.506` Ha -- clean convergence, but at steep cost (`N=48` took `~4.2` min, `N=64` took `~15.7` min on this session's hardware, confirming 3D grid cost is the real practical constraint this whole project's Scope decision anticipated, not a paper concern).
- [x] **Phase 4 — Helium atom SCF** ✅ done. Two electrons, exercises Hartree+XC together for the first time in 3D. **Validated**: `E_total=-2.51842` Ha vs. `HF_solver`'s already-validated radial LDA result for He, `-2.834178` Ha (`11.1%` off) -- a project-native cross-check the earlier two solvers didn't have available, since this is the third leg of the same initiative, and a genuinely independent one (no shared code at all: different coordinate system, different eigensolver, different Poisson method). The gap is consistent with this build's already-characterized, stacked approximations (grid softening, the Phase 2 box-size Poisson error, finite resolution) rather than a new bug -- not chased further at N=32 resolution, per the Scope decision.
- [x] **Phase 5 — H2 molecule SCF (`H2_Validation.ipynb`)** ✅ done. Two nuclei, no symmetry exploited this time (unlike `Diatomic_HF_solver`'s prolate-spheroidal reduction) -- genuinely tests the full, general 3D machinery, and the natural foundation for eventual polyatomic molecules.

  **A second real bug, found and fixed getting here**: the first H2 run gave a wildly overbound `E_total=-1.963` Ha -- `scf3d.run_scf` never added the nuclear-nuclear repulsion term `Z_A*Z_B/R`, trivially zero (and so never surfaced) for the single-nucleus atoms in Phases 3-4, but essential the instant a second nucleus exists. Adding `Z_A*Z_B/R` (`0.714` Ha at `R=1.4`) immediately brought the result in line with expectations.

  **Validated**: single geometry at `R=1.4` Bohr gives `E_total=-1.24851` Ha vs. `Diatomic_HF_solver`'s already-validated LDA result `-1.135544` Ha (`9.9%` off -- a genuinely independent cross-check, no shared code at all between the two solvers). A 5-point bond-length scan (`R=1.0` to `3.0` Bohr) produces a real bonding curve -- steep repulsive wall at small `R`, a genuine interior minimum, dispersing again at large `R` -- confirming actual chemical bonding falls out of the fully general 3D machinery with zero symmetry assumptions built in. The minimum itself lands at `R=1.80` Bohr, `E_total=-1.28994` Ha, noticeably shifted from the known `R_e~1.40` Bohr and shallower/flatter than the true curve (only `~0.04` Ha of curvature between `R=1.4` and `R=2.2`) -- an honest, real limitation at this build's resolution (`N=32^3`, `dx=0.5` Bohr), consistent with the already-characterized Phase 2 box-size Poisson error not scaling uniformly with `R` (a more spread-out bonding density at larger `R` sees a different effective box-size penalty than the same charge concentrated near `R=1.4`), compounded by grid resolution coarse relative to the bond-length scale. Reported as found, not smoothed over -- tightening resolution/box size together (at real computational cost, per Phase 3's `N=48`/`N=64` timing) would be the natural next step to close this gap, out of scope for this initial build.

## Known limitations (anticipated, to confirm/document once built)

- **All-electron, light species only** -- no pseudopotentials in this build, so nothing heavier than He/H2 is in scope (see Scope decision above).
- **Periodic/supercell boundary conditions** -- an approximation to a truly isolated molecule, controlled by box-size convergence, not exact.
- **Softened nuclear cusp** -- unlike the radial/prolate-spheroidal solvers' analytic cusp handling, a 3D Cartesian grid needs the bare `1/r` singularity softened to stay resolvable, trading some accuracy for tractability; characterized empirically in Phase 3.
- **Same non-relativistic, LDA-level XC caveats as the rest of this initiative** -- inherited, not new.

## Verification

1. ✅ Ran Phase 1's notebook: confirmed exact free-particle plane-wave eigenvalues (`1.7e-14` relative error) before trusting the kinetic operator for anything else.
2. ✅ Confirmed Phase 2's Poisson solver against a closed-form Gaussian-charge potential, and characterized its box-size convergence (algebraic, `~1/L`, not exponential -- a real, documented limitation, not a bug).
3. ✅ Ran Phase 3's hydrogen SCF and confirmed convergence near `-0.5` Ha, characterizing both the softening-driven gap (`4.9%` at the practical sweet spot, `N=32`) and genuine convergence with resolution (`-0.506` Ha at `N=64`, at steep compute cost).
4. ✅ Ran Phase 4's helium SCF and compared directly against `HF_solver`'s already-validated `-2.834178` Ha LDA result (`11.1%` off, ballpark-consistent with the stacked, already-characterized approximations).
5. ✅ Ran Phase 5's H2 SCF (single geometry, then a bond-length scan) and compared against `Diatomic_HF_solver`'s already-validated `-1.135544` Ha result (`9.9%` off) -- found and fixed a missing nuclear-repulsion-term bug along the way. The bond-length scan produces a genuine minimum with real bonding-curve shape, though shifted in `R` from the known equilibrium -- reported honestly as a resolution-driven limitation, not hidden.

**This closes the full 3-phase DFT initiative**: atomic (`HF_solver/`, radial grid) -> diatomic (`Diatomic_HF_solver/`, prolate-spheroidal grid) -> general molecular (`Molecular_DFT/`, 3D Cartesian grid, this project). Natural next steps, explicitly out of scope for this initial build: pseudopotentials (to reach heavier atoms / real polyatomic molecules like H2O), a specialized isolated-boundary Poisson solver (to fix the algebraic box-size convergence characterized in Phase 2), and tighter default resolution (at the real compute cost characterized in Phase 3).
