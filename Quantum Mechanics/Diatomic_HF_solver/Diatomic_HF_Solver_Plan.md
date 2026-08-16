# Diatomic Hartree-Fock-Slater Solver — Design Plan

## Context

This is the natural sequel to `Quantum Mechanics/HF_solver/` (see its `HF_Solver_Plan.md`), which solves the atomic case: a spherically-symmetric central-field mean-field (Hartree + Slater/Xα exchange) problem, reduced from 3D to a 1D radial grid via separation into spherical-harmonic (n, l) channels and solved with a log-grid finite-difference generalized eigenproblem (`scipy.sparse.linalg.eigsh`, shift-invert).

A diatomic molecule (two nuclei, A and B, separated by fixed distance R) is **not** spherically symmetric, so that exact reduction doesn't carry over. It **is** axially symmetric about the internuclear axis, which is the basis of a different, but structurally analogous, exact reduction: **prolate spheroidal coordinates**.

```
mu = (r_A + r_B) / R      in [1, infinity)
nu = (r_A - r_B) / R      in [-1, 1]
phi = azimuthal angle about the internuclear axis, in [0, 2*pi)
```

`r_A`, `r_B` are the electron's distances to nuclei A and B. In these coordinates the bare two-center nuclear potential `-Z_A/r_A - Z_B/r_B` is exactly separable — this is the classic method used to solve H2+ exactly (Bates, Ledsham & Stewart 1953), and it is also the basis of a real, established numerical method for many-electron diatomics: finite-difference Hartree-Fock in (mu, nu) coordinates (Laaksonen/Sundholm-style "fully numerical" diatomic codes, still used today for high-precision diatomic benchmarks). This is not a novel scheme being invented for this repo — it is the standard alternative to Gaussian-basis Roothaan-Hall for diatomics specifically, chosen here (as the atomic solver chose radial-grid over Gaussian-basis) to keep using finite-difference/grid machinery rather than switching to a completely different numerical paradigm (multi-center Gaussian integrals), and because it reuses much of what `HF_solver/` already built.

**Scope decision made with the user up front**: this solver handles **diatomics only** (two nuclei). General polyatomic molecules would require abandoning the exact coordinate separation entirely (no axis of symmetry to exploit) and moving to a genuinely different method (Gaussian-basis Roothaan-Hall, or a full 3D real-space grid) — explicitly out of scope for this project, which is about extending the existing grid/finite-difference approach as far as an exact symmetry will take it, not about building a general quantum chemistry package.

The atomic solver's l (orbital angular momentum) is replaced by **lambda**, the azimuthal quantum number about the molecular axis (`|m|` in the atomic sense) — the *only* angular momentum-like quantity conserved for a two-center potential. Molecular orbitals are labeled sigma (lambda=0), pi (lambda=1), delta (lambda=2), phi (lambda=3), by direct analogy to s/p/d/f. Homonuclear diatomics (Z_A = Z_B) get one more exact symmetry for free: inversion parity (gerade/ungerade), which is this project's analog of the atomic solver's l-degeneracy check — a strong, independent correctness signal, not required for the solver to function.

## What carries over from `HF_solver/` largely unchanged

- **Overall SCF architecture**: build effective potential from current density -> diagonalize -> read off fixed occupations -> rebuild density -> mix -> check convergence (`scf.py`'s `run_scf` loop shape).
- **Slater/Xalpha exchange**: `V_x(r) = -3*alpha*(3*rho(r)/(8*pi))**(1/3)` is a purely local functional of the density at a point — completely indifferent to geometry, so `potentials.slater_exchange_potential`'s functional form ports over unchanged (just evaluated on a 2D grid instead of 1D).
- **Sparse generalized-eigenproblem solving approach**: `scipy.sparse.linalg.eigsh` with shift-invert (`sigma`) is still the right tool; the matrix is just bigger and has a different (non-tridiagonal) sparsity pattern.
- **Total energy bookkeeping**: the `sum_nl(N_nl*eps_nl) - E_H - E_x/3` structure (Euler's-theorem-derived double-counting correction, `HF_Solver_Plan.md` Phase 6) still holds, since it depends only on the Xalpha functional's scaling behavior, not on geometry. One new additive term is required: **nuclear-nuclear repulsion** `Z_A*Z_B/R`, which has no atomic analog (a single nucleus doesn't repel itself).

## What has to change

- **Grid**: 1D log-radial grid -> 2D grid in `(mu, nu)`. `mu` needs a grid that resolves both nuclear cusps (`mu -> 1`) and the long-range tail (`mu -> mu_max`), analogous to the atomic solver's Z-dependent `r_min`; `nu` is naturally bounded, a uniform or Gauss-Lobatto-like grid on `[-1, 1]` should suffice.
- **Eigensolver**: the atomic 1D tridiagonal `(H, M)` pencil becomes a 2D 5-point-stencil sparse matrix pencil on the `(mu, nu)` grid, one per lambda-channel (instead of one per l-channel). Larger (N_mu * N_nu square) but still sparse and still amenable to shift-invert `eigsh` — no new eigensolver strategy needed, just a bigger, differently-shaped operator.
- **Hartree potential**: the atomic solver's O(N) shell-theorem trick (`potentials.hartree_potential`) is a 1D spherical-symmetry shortcut that has no 2D equivalent. Fix: expand `1/|r - r'|` in Legendre polynomials of `nu` (the standard two-center multipole expansion) — this decouples the 2D Poisson problem into a series of coupled 1D-in-`mu` ODEs, one per multipole order `L`, truncated at some `L_max` chosen by a convergence check (residual multipole contribution below tolerance). This is the single largest new piece of numerical machinery this project requires; everything else is a re-shaping of existing ideas.
- **Shell filling**: no Madelung-style hardcoded table needed. Molecular orbitals fill by straightforward aufbau on whatever energies the eigensolver returns per lambda-channel (simpler than the atomic case, which needed `shells.py`'s exceptions table specifically because Madelung order sometimes disagrees with true energy order — for diatomics we just sort the computed energies directly).
- **Total energy**: add the `Z_A*Z_B/R` nuclear repulsion term noted above.

## File layout

Self-contained sibling folder to `HF_solver/`, same convention (design doc first, `media/` for validation plots):

```
Quantum Mechanics/Diatomic_HF_solver/Diatomic_HF_Solver_Plan.md   # this design document
Quantum Mechanics/Diatomic_HF_solver/prolate_coords.py            # coordinate transforms, Jacobian, r_A/r_B <-> (mu,nu,phi)
Quantum Mechanics/Diatomic_HF_solver/diatomic_scf.py               # 2D (mu,nu) finite-difference eigensolver, per lambda-channel
Quantum Mechanics/Diatomic_HF_solver/multipole_potential.py        # two-center Hartree potential via Legendre multipole expansion + Slater/Xa exchange reused from potentials.py
Quantum Mechanics/Diatomic_HF_solver/molecular_shells.py           # aufbau filling by computed energy (no Madelung table needed), sigma/pi/delta labeling, g/u parity
Quantum Mechanics/Diatomic_HF_solver/diatomic_driver.py            # SCF loop (same shape as HF_solver/scf.py's run_scf) + total energy incl. nuclear repulsion
Quantum Mechanics/Diatomic_HF_solver/H2+ Validation.ipynb          # Phase 1: bare two-center Coulomb problem vs. exact H2+ energies
Quantum Mechanics/Diatomic_HF_solver/Diatomic SCF Validation.ipynb # Phases 2-4: multipole Hartree check, H2 SCF, N2/O2/F2
Quantum Mechanics/Diatomic_HF_solver/media/                        # saved plot outputs per phase, same convention as HF_solver/media/
```

`potentials.py`'s `slater_exchange_potential` and the alpha presets (`ALPHA_LDA`/`ALPHA_SLATER`/`ALPHA_SCHWARZ`) should be imported from `HF_solver/` rather than duplicated, once this folder actually starts depending on that one (needs either a shared path/package setup or a light copy — decide at implementation time, not speculatively now).

## Phase breakdown

- [ ] **Phase 1 — Prolate spheroidal coordinates + bare two-center eigensolver (`prolate_coords.py`, `diatomic_scf.py`)**. Coordinate transforms and the Jacobian; build the 2D `(H, M)` pencil for `V = -Z_A/r_A - Z_B/r_B` only (no Hartree/exchange yet, exactly mirroring how `HF_solver/atomic_scf.py`'s Phase 2 validated against pure Coulomb before any SCF machinery existed); solve per lambda-channel via `eigsh`. **Validation**: exact H2+ energies are tabulated in the literature (Bates-Ledsham-Stewart and many later sources) for a range of internuclear distances R — matching those is this phase's acid test, the direct analog of the atomic solver's `E_nl = -Z**2/(2n**2)` check.
- [ ] **Phase 2 — Two-center multipole Hartree potential (`multipole_potential.py`)**. Legendre expansion of `1/|r-r'|` in two-center coordinates; per-multipole 1D-in-`mu` ODE solves; truncation/convergence check on `L_max`. **Validation**: needs a two-center density with a known closed-form self-Coulomb energy, or cross-check against a brute-force 2D numerical double integral for a simple test density (slower, but exact) at small grid size, the same role `SCF Validation.ipynb`'s hydrogen-1s closed-form check played for the atomic Hartree potential.
- [ ] **Phase 3 — Molecular orbital filling (`molecular_shells.py`)**. Aufbau on computed lambda-channel energies (no Madelung table); sigma/pi/delta labeling; gerade/ungerade parity for homonuclear cases. **Validation**: known ground-state MO configurations (e.g. N2: `1sigma_g^2 1sigma_u^2 2sigma_g^2 2sigma_u^2 1pi_u^4 3sigma_g^2`) matched by construction once aufbau ordering is confirmed against literature MO diagrams.
- [ ] **Phase 4 — Full SCF loop + total energy (`diatomic_driver.py`)**. Same mixing/convergence shape as `HF_solver/scf.py`'s `run_scf`; add `Z_A*Z_B/R` nuclear repulsion to the total energy. **Validation**: H2 first (2 electrons, simplest possible many-electron case, well-documented literature non-relativistic HF energy to compare against, same role He played for the atomic solver); then N2/O2/F2 to exercise pi (and, for anything heavier, delta) orbitals and confirm the multipole Hartree solve holds up for a less trivial density.
- [ ] **Phase 5 — Validation & visualization**. Bond-length energy scan (`E_total` vs. `R`) to confirm a genuine minimum (a bonding curve) falls out of the SCF — the molecular analog of the atomic solver's shell-peak radial density as "does this look like real physics" validation; MO energy-level diagrams; 2D density plots in the `(mu, nu)` plane and/or converted back to real-space `(x, z)` cross-sections reusing the atomic solver's `imshow` cross-section convention.

## Known limitations (anticipated, to confirm/document once built)

- **Diatomics only** — no polyatomic extension path from this method (see Scope decision above); a genuinely different solver would be needed for 3+ nuclei.
- **Fixed nuclear geometry per run** — R is an input, not solved for; a bonding curve requires multiple full SCF runs at different R (same pattern as any Born-Oppenheimer PES scan), not a single relaxation.
- **Same non-relativistic, configuration-averaged, Xalpha-exchange caveats as `HF_solver/`** — inherited, not new: no relativistic corrections, no full term-symbol/multiplet treatment, Xalpha is an approximation to true non-local HF exchange.
- **Multipole truncation error** in the Hartree potential is a new approximation this project introduces that the atomic solver didn't have (its shell-theorem Hartree potential was exact, not truncated) — needs its own convergence check per Phase 2, and is likely the main new source of numerical error to characterize.

## Verification

1. Run the Phase 1 notebook first: confirm exact H2+ energy match across a range of R (this phase's version of the atomic solver's Radial SCF Validation acid test) before trusting anything downstream.
2. Confirm the Phase 2 multipole Hartree potential against a brute-force reference for at least one test density.
3. Confirm Phase 3's aufbau filling reproduces known ground-state MO configurations for N2, O2, F2.
4. Run Phase 4's SCF for H2 and confirm convergence + a total energy in the right ballpark of literature non-relativistic HF for H2.
5. Run a bond-length scan (Phase 5) for at least one molecule and confirm a genuine energy minimum at a physically reasonable bond length (e.g. H2's ~1.4 Bohr).
