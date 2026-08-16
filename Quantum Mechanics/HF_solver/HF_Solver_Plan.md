# Atomic Hartree-Fock-Slater Solver — Design Plan

## Context

The repo has an empty stub at `Quantum Mechanics/Hartree_Fock.ipynb` (just an import cell) and no other quantum-chemistry code anywhere — `Quantum Mechanics/Hydrogen Atom.ipynb` only plots spherical harmonics `sph_harm(m,l,theta,phi)`, with no radial wavefunction or multi-electron treatment. The goal is to build a working atomic electronic-structure solver from scratch (no `pyscf`/`psi4` installed or wanted), starting with orbital generation/visualization and building up to a full self-consistent-field (SCF) calculation for atoms across the periodic table, including transition metals and lanthanides/actinides (d/f shells).

Two architectural decisions were made with the user up front:
- **Radial-grid / finite-difference approach**, not a Gaussian-basis Roothaan-Hall approach — exploits atomic spherical symmetry, and matches this repo's existing precedent of building a finite-difference Hamiltonian and calling `scipy.linalg.eigh_tridiagonal` (seen in `Quantum Mechanics/Alpha Decay.ipynb`: uniform grid, `main_diag`/`off_diag` arrays, `eigh_tridiagonal(main_diag, off_diag, select='v', select_range=...)`, `np.trapz` normalization).
- Since exact non-local Hartree-Fock exchange isn't tractable on a radial grid, exchange is treated via the **Slater/Xα local-exchange approximation** (Hartree-Fock-Slater method) — a standard, well-documented simplification used by classic atomic-structure codes (Herman-Skillman, Desclaux).
- Must support the **full periodic table** (s/p/d/f), which requires a **configuration-averaged central-field treatment** for open shells (fractional/integer occupation per (n,l) shell, spherically averaged density) rather than a full term-symbol/multiplet treatment.

House style to match throughout: plain procedural functions (no classes), standard `numpy`/`scipy`/`sympy`/`matplotlib` header, `scienceplots` styling where convenient (optional — only installed in the global Python, not `torch.venv`; skip it rather than block on installing it). Run/test using `torch.venv`'s Python (has numpy/scipy/sympy/matplotlib already).

## Progress

- [x] Phase 1 — Analytic hydrogen-like orbitals — `hydrogenic.py` implemented; normalization verified exactly (integral=1.000000) for n up to 4; angular/radial/cross-section plots visually confirmed and saved to `media/`. Had to use `scipy.special.sph_harm_y` (not `sph_harm`, removed in installed SciPy 1.17) and `np.trapezoid` (not `np.trapz`, removed in installed NumPy 2.4).
- [x] Phase 2 — Radial finite-difference eigensolver — `atomic_scf.py` implemented; validated in `Radial SCF Validation.ipynb` to ~1e-5 to 4e-5 relative energy error across Z=1..100 and l=0..3, with l-degeneracy confirmed. **Deviated from the original plan**: the "rescale to a plain symmetric tridiagonal `eigh_tridiagonal` problem" approach was numerically unstable — dividing by `r` to eliminate the generalized eigenproblem creates matrix entries spanning ~1/r_min^2, which overflows float64's usable dynamic range once r_min is small enough to resolve heavy-atom cores (silently gave wrong/garbage eigenvalues, sometimes even positive, for r_min ≲ 1e-5). Fixed by keeping the problem in its natural generalized form `H w = E M w` and solving via `scipy.sparse.linalg.eigsh(..., M=M, sigma=..., which="LM")` (shift-invert), which factorizes `H - sigma*M` at natural scale instead of forming an explicit rescaling. Also switched grid defaults to Z-dependent (`default_grid(Z)`: `r_min=1e-5/Z`, `r_max=150`, `N=4000`) since core-orbital size scales as 1/Z.
- [x] Phase 3 — Effective potential — `potentials.py` implemented (`hartree_potential`, `slater_exchange_potential`, `effective_potential`). Validated exactly: for the hydrogen 1s density, `E_H = 0.5*integral(V_H*rho*4*pi*r**2 dr)` reproduces the classic analytic self-Coulomb energy `5/16 Ha = 0.3125` to 9 significant figures, and `V_H(r)*r -> N` at large r as required. See `SCF Validation.ipynb`.
- [x] Phase 4 — Shell filling (Madelung + exceptions) — `shells.py` implemented (`madelung_order`, `ground_state_configuration`, `format_configuration`, `EXCEPTIONS`). Spot-checked against known reference configurations (He, Ne, Ar, Fe, Cr, Cu, Pd, La, Gd, U) — all match exactly, including electron count. See `SCF Validation.ipynb`.
- [x] Phase 5 — Full SCF loop — `scf.py` implemented (`run_scf`). Converges cleanly for He/Ne/Ar (closed shell), Cr/Cu (Madelung exceptions, confirmed the hardcoded configuration is actually used), and Fe/Gd (spot-checked separately, confirming l_max=3/f-orbital support end-to-end — Gd's SCF run takes ~22 minutes at the default N=4000 grid, noted as a performance consideration, not a correctness issue, for whatever Phase 7 cases get chosen). See below for energy numbers and the ARPACK robustness fix this phase required, and `SCF Validation.ipynb` + `media/phase3_6_scf_validation.png` for the runnable validation.
- [x] Phase 6 — Total energy — `compute_total_energy` implemented in `scf.py` ahead of schedule since Phase 5's convergence check needs it; derivation and formula documented below.
- [x] Phase 7 — Validation & visualization — `Hartree_Fock.ipynb` implemented: sanity checks, an animated 4-panel SCF-convergence movie (`scf_movie.py`, `media/scf_convergence_Ar.mp4`), radial density / 2D cross-section / 3D angular-shape plots reusing `hydrogenic.py` with numeric `R_nl(r)` swapped in, plus a new emission-spectra extension (`spectra.py`, `media/emission_spectra.png`) — see below.

## File layout

Everything lives inside a single self-contained folder, `Quantum Mechanics/HF_solver/`, rather than scattered across the top-level `Quantum Mechanics/` folder:

```
Quantum Mechanics/HF_solver/HF_Solver_Plan.md         # this design document
Quantum Mechanics/HF_solver/hydrogenic.py             # Phase 1: analytic hydrogen-like orbitals + viz helpers
Quantum Mechanics/HF_solver/atomic_scf.py             # Phase 2: log-grid radial eigensolver
Quantum Mechanics/HF_solver/potentials.py             # Phase 3: Hartree + Slater/Xa effective potential
Quantum Mechanics/HF_solver/shells.py                 # Phase 4: Madelung shell filling + exceptions table
Quantum Mechanics/HF_solver/scf.py                    # Phase 5-6: SCF driver + total energy (+ optional per-iteration history recording)
Quantum Mechanics/HF_solver/scf_movie.py              # Phase 7: 4-panel SCF-convergence animation from recorded history
Quantum Mechanics/HF_solver/spectra.py                # Phase 7 extension: frozen-potential single-active-electron emission spectra
Quantum Mechanics/HF_solver/Radial SCF Validation.ipynb   # validates Phase 2 against analytic hydrogen results
Quantum Mechanics/HF_solver/SCF Validation.ipynb      # validates Phases 3-6 (potential, shells, SCF, energy)
Quantum Mechanics/HF_solver/Hartree_Fock.ipynb        # Phase 7 — runs SCF for chosen atoms, builds the SCF movie + orbital/density visuals + emission spectra
Quantum Mechanics/HF_solver/media/                    # saved plot/animation outputs from every phase (see convention below)
```

(The original plan sketched Phases 2-6 as all living in one `atomic_scf.py`; in practice each phase got its own module — smaller files, and it keeps `atomic_scf.py`'s scope to just the eigensolver.)

The pre-existing `Quantum Mechanics/Hartree_Fock.ipynb` stub and `Quantum Mechanics/Hydrogen Atom.ipynb` are left untouched at the top level (not moved/deleted) — they're only referenced here as style/precedent, not built upon in place. All new code and notebooks for this project are self-contained under `HF_solver/`, so notebooks there should import the local modules via a relative/same-folder import (e.g. `import hydrogenic`, `import atomic_scf`) rather than reaching back into the parent `Quantum Mechanics/` folder.

**Media convention**: every phase that produces a plot saves a representative PNG to `HF_solver/media/` (e.g. `angular_shapes.png`, `radial_density.png`, `density_slices.png` for Phase 1) so the validation/demo plots are preserved as reviewable artifacts alongside the code, not just transient notebook/matplotlib output.

This follows the one existing precedent for a shared helper module (`Electromagnetism/numerical.py`), since the solver machinery (grid, matrix builder, Poisson quadrature, exchange, Madelung table, SCF loop) is substantial and reused across notebooks — unlike the small inline blocks in `Alpha Decay.ipynb`/`2 Delta Wells.ipynb`.

## Phase 1 — Analytic hydrogen-like orbitals (`hydrogenic.py`) ✅ done

- `radial_wavefunction(n, l, r, Z=1)`: analytic `R_nl(r)` via `scipy.special.genlaguerre` (associated Laguerre polynomials) and the standard normalization constant.
- `psi_nlm(n, l, m, r, theta, phi, Z=1)`: combines `radial_wavefunction` with `scipy.special.sph_harm` (same call already used in `Hydrogen Atom.ipynb`) to build the full 3D orbital.
- `radial_density(n, l, r, Z=1)`: returns `r**2 * R_nl(r)**2` for radial distribution plots.
- Visualization helpers matching house style: angular-shape `plot_surface` (as in `Hydrogen Atom.ipynb`), 1D line plots of `r**2 R_nl(r)**2`, and a 3D isosurface/cross-section view of `|psi_nlm|**2` for a few illustrative (n,l,m).
- Test: verify `∫ R_nl(r)**2 r**2 dr = 1` numerically (`np.trapz`) for a handful of (n,l,Z), and visually sanity-check shapes against known orbital pictures (s spherical, p dumbbell, d cloverleaf, etc.).

## Phase 2 — Radial finite-difference eigensolver (`atomic_scf.py`) ✅ done

- **Log-spaced grid**: `x = ln(r)`, uniform step `h`, `r_i = r_min * exp(i*h)` — needed because core orbitals (r ~ 1/Z) and outer valence shells (n up to 6-7 for heavy atoms) span many orders of magnitude in r; a uniform r-grid can't resolve both without an unmanageable point count. `default_grid(Z, r_max=150, n_points=4000)` scales `r_min = 1e-5/Z` since core-orbital size shrinks as 1/Z.
- **Substitution**: `u(r) = r*R(r)`, then `u(r) = sqrt(r)*w(x)`. This eliminates the first-derivative term from the `x=ln(r)` transform and turns the centrifugal term into `(l+1/2)^2` instead of `l(l+1)` (a known, correct artifact of this substitution — not a bug), giving the generalized eigenproblem `H w = E M w` with, per l-channel:
  - `main_diag_H[i] = 2/h**2 + (l+0.5)**2 + 2*r_i**2*V(r_i)`, `off_diag_H[i] = -1/h**2`
  - `M_diag[i] = 2*r_i**2`
- **Solved as a generalized eigenproblem, not rescaled to a standard one.** The original plan called for dividing through by `r` to get a plain symmetric tridiagonal problem solvable by `eigh_tridiagonal` (matching `Alpha Decay.ipynb`'s style) — this turned out to be numerically unstable: it makes matrix entries span ~1/r_min^2, which overflows float64's usable dynamic range once r_min is small enough to resolve heavy-atom cores (confirmed empirically: silently wrong/garbage eigenvalues, sometimes even positive, for r_min ≲ 1e-5). Fixed by keeping `H`/`M` at natural scale and solving via `scipy.sparse.linalg.eigsh(H, k=n_states, M=M, sigma=..., which="LM")` (shift-invert, `sigma` estimated from `-r[0]*V[0]` as a proxy for the effective nuclear charge) — this factorizes `H - sigma*M` directly, which never develops the same dynamic-range blowup.
- `log_grid(r_min, r_max, N)`, `default_grid(Z, ...)`, `build_radial_matrices(r, l, V, h)`, `solve_radial_channel(r, l, V, h, n_states=8)` (returns the lowest ~8 states per l-channel — enough to cover n up to 7-8 for l=0..3), `radial_function_from_u(u, r)`.
- **No shooting/Numerov method** — direct diagonalization returns the whole ordered low-lying spectrum per l-channel with no manual energy bracketing or node-counting, which is essential for automating this across arbitrary Z inside an SCF loop.
- **Validated** in `Radial SCF Validation.ipynb`: with `V(r) = -Z/r` only, eigenvalues match the analytic hydrogen-like result `E_nl = -Z**2/(2*n**2)` to ~1e-5 to 4e-5 relative error across Z=1..100 and l=0..3; the "accidental" l-degeneracy (same n, different l) holds to the same precision; numeric `R_nl(r)` overlays the analytic curve almost exactly (see `media/phase2_validation.png`).

## Phase 3 — Effective potential (`potentials.py`) ✅ done

- **Hartree (electron-electron Coulomb) potential**, via the shell theorem applied to the spherically-averaged density (efficient O(N) evaluation, not an O(N²) double integral): `hartree_potential(r, rho)` computes `Q(r) = cumulative_trapezoid(4*pi*r'**2*rho(r'))` (enclosed charge) and `S(r) = cumulative_trapezoid(4*pi*r'*rho(r'))`, returning `V_H(r) = Q(r)/r + (S(r_max) - S(r))`.
- **Slater/Xα exchange**: `slater_exchange_potential(r, rho, alpha)` implements `V_x(r) = -3*alpha*(3*rho(r)/(8*pi))**(1/3)`. Three presets exposed as module constants: `ALPHA_LDA = 2/3` (theoretically exact LDA value), `ALPHA_SLATER = 1.0` (classic Slater), `ALPHA_SCHWARZ = 0.7` (default — Schwarz-optimized empirical compromise). Documented as a tunable approximation, not a fixed constant.
- `effective_potential(r, rho, Z, alpha)` combines `-Z/r + V_H + V_x`.
- **Validated** against the one case with an exact closed-form answer: for the hydrogen 1s density (`rho = R_10(r)**2/(4*pi)`, `N=1`), `E_H = 0.5*integral(V_H*rho*4*pi*r**2, dr)` reproduces the textbook self-Coulomb energy `5/16 Ha = 0.3125` to 9 significant figures, and `V_H(r)*r -> N` at large r as expected from the shell theorem.

## Phase 4 — Shell filling (`shells.py`) ✅ done

- `madelung_order(n_max=8, l_max=3)`: generates `(n, l)` shells sorted by `(n+l, n)` — the standard Madelung fill order.
- `shell_capacity(l) = 2*(2l+1)`, and `ground_state_configuration(Z, exceptions=EXCEPTIONS)` fills shells along Madelung order up to `Z` electrons, then swaps in a hardcoded exception's *valence* shells when `Z` is one of the 20 well-known real-world Madelung anomalies (Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, La, Ce, Gd, Pt, Au, Ac, Th, Pa, U, Np, Cm, Lr), re-deriving the noble-gas core underneath via ordinary Madelung filling (`core_Z = Z - sum(valence overrides)`). Lr flagged lower-confidence per the original plan (relativistic effects outside scope).
- `format_configuration(config)` renders a config dict as e.g. `"1s2 2s2 2p6 3s2 3p6 3d5 4s1"` for readability.
- Occupations are held **fixed** for a given Z throughout the SCF run, as planned.
- **Validated**: spot-checked `ground_state_configuration` against known reference configurations for He, Ne, Ar, Fe (normal Madelung), Cr, Cu, Pd (exceptions with 5s0/5s1 quirks), La, Gd (lanthanide exceptions), and U (actinide exception) — all match exactly (including total electron count) against hand-derived NIST ASD configurations.

## Phase 5 — Full SCF loop (`scf.py`, `run_scf`) ✅ done

Implemented exactly as planned: build `V_eff` from current density → diagonalize each occupied l-channel via `solve_radial_channel` (only the l-channels actually present in the atom's configuration, with `n_states` per channel sized to the highest occupied n for that l) → read off `N_nl` from the fixed Phase 4 configuration → rebuild the spherically-averaged density `rho(r) = (1/(4*pi*r**2))*sum_nl(N_nl*u_nl(r)**2)` → linearly mix with the previous iteration's density → check convergence (`|ΔE_total| < 1e-6` Ha and `∫|Δrho|*4*pi*r**2 dr < 1e-5`, sustained for 2 consecutive iterations; 200-iteration cap with a `warnings.warn` if not reached, via a `for...else`).

**Deviation from the original plan — ARPACK robustness fix**: `solve_radial_channel`'s shift-invert `eigsh` call, which was rock-solid for the pure-Coulomb Phase 2 validation, occasionally threw `ArpackNoConvergence` mid-SCF-run (observed concretely: Cu's p-channel, `k=2`, against the crude exponential seed density used on iteration 1). Root cause is that ARPACK's Lanczos iteration is more sensitive to a poorly-conditioned starting point than to the physics itself — a seed density far from self-consistency. Fixed by wrapping the `eigsh` call in a retry loop (`atomic_scf.solve_radial_channel`) that widens the Krylov subspace (`ncv`) and perturbs `sigma` by 20% on each of up to 4 attempts, raising only if all attempts fail.

**Validated**: `run_scf` converges cleanly (well within 200 iterations) and integrates to the exact electron count (`Q(r_max) = N` to machine precision) for:
  - He: `E_total = -2.76637 Ha` (lit. non-rel. HF: -2.86168 Ha)
  - Ne: `E_total = -128.03490 Ha` (lit.: -128.547 Ha)
  - Ar: `E_total = -525.89503 Ha` (lit.: -526.818 Ha)
  - Cr: `E_total = -1042.21990 Ha`, confirmed using the exception configuration `3d5 4s1` (not naive Madelung `3d4 4s2`)
  - Cu: `E_total = -1638.28291 Ha`, confirmed using the exception configuration `3d10 4s1` (not naive Madelung `3d9 4s2`)

All within 0.2-3.5% of literature non-relativistic HF energies, consistent with Xα being a local-exchange approximation rather than exact HF (expected to be somewhat less binding, especially for lighter/fewer-electron atoms where exchange is a proportionally larger share of the energy).

## Phase 6 — Total energy (`scf.compute_total_energy`) ✅ done (implemented ahead of schedule)

Implemented inside `scf.py` rather than `atomic_scf.py` as originally sketched, because Phase 5's convergence check directly needs `E_total` every iteration:
```
E_total = sum_nl( N_nl * epsilon_nl ) - E_H - E_x/3

E_H = 0.5 * integral( V_H(r) * rho(r) * 4*pi*r**2 dr )
E_x = 0.75 * integral( V_x(r) * rho(r) * 4*pi*r**2 dr )
```
`E_x` here is computed directly from `V_x` and `rho` (rather than the `rho**(4/3)` integral form) using the identity `integral(V_x*rho*4*pi*r**2 dr) = (4/3)*E_x`, which follows from Euler's theorem since `E_x[rho]` scales as `rho**(4/3)` under `rho -> lambda*rho`. This is algebraically equivalent to the originally-planned form but avoids computing the `(3/(8*pi))**(1/3)` prefactor a second time (it is already baked into `V_x` from Phase 3). The `1/3` (not `1/2`) double-counting correction for exchange follows from the same scaling argument, distinct from the Hartree term's simple pairwise `1/2`.

## Phase 7 — Validation & visualization (`Hartree_Fock.ipynb`) ✅ done

- **Sanity checks**: (a) `Q(r_max) = N` every iteration; (b) zeroing `V_H`/`V_x` reproduces exact hydrogen-like energies (regression test reusing Phase 2's check); (c) SCF energy converges monotonically-ish, plotted per iteration.
- **SCF algorithm movie** (the main ask that kicked off this phase — the SCF loop is hard to build intuition for from numbers alone): `run_scf` gained an opt-in `record_history=True` flag (`scf.py`) that snapshots `rho`, `V`, `orbital_energies`, `E_total`, `dE`, `dn` every iteration with no effect on the physics or default call signature. `scf_movie.make_scf_animation` turns that into a 4-panel animation — density `4*pi*r**2*rho(r)`, potential `r*V_eff(r)`, per-l orbital energy levels, and the `E_total` convergence trace with a moving "you are here" marker — one frame per SCF iteration. Density/potential panels use a log-r axis and the levels panel a symlog energy axis; both are necessary because core shells (r~1/Z, |eps| up to ~100 Ha) and valence shells (r~few Bohr, |eps|~0.1-1 Ha) span orders of magnitude and are invisible together on linear axes. Rendered for Ar (`media/scf_convergence_Ar.mp4`, 45 frames) via matplotlib's ffmpeg writer — `imageio-ffmpeg` was pip-installed into `torch.venv` since no system ffmpeg was present; `matplotlib.rcParams['animation.ffmpeg_path']` must point at `imageio_ffmpeg.get_ffmpeg_exe()` for `FFMpegWriter.isAvailable()` to pick it up. Falls back to an animated `.gif` (Pillow writer, always available) if ffmpeg isn't set up.
- **More visual aids**: `hydrogenic.py`'s `real_orbital`/`plot_radial_density`/`plot_orbital_density_slice` gained an optional `R_func` callable that overrides the analytic `radial_wavefunction(n,l,r,Z)` — exactly the "reusing hydrogenic.py's plotting code with the numerically-computed R_nl(r) swapped in" the original plan called for, done via one added parameter rather than a parallel set of numeric-orbital plotting functions. Used for Ar's 1s/2s/2p/3s/3p radial densities, a 2D cross-section, and 3D angular shapes.
- **Reference comparison**: literature non-relativistic HF total energies (He: -2.86168 Ha, Ne: -128.547 Ha, Ar: -526.818 Ha) are trend/order-of-magnitude checks only, per `SCF Validation.ipynb`.
- **Known limitations documented in the notebook**: no relativistic corrections (Z>80 qualitative only); configuration average, not full term-symbol treatment; Xα is an approximation to true HF exchange with a tunable `alpha`.

## Phase 7 extension — single-active-electron emission spectra (`spectra.py`)

Not in the original plan — a "logical leap" the user asked for on top of Phase 7: since the solver already produces self-consistent orbital energies, freeze the converged `V_eff` and treat one electron promoted out of the atom's valence shell (`spectra.valence_shell`, the last shell filled along Madelung order) as a single active electron moving through that frozen mean field's *excited* eigenstates (`spectra.excited_levels`, just `solve_radial_channel` with more `n_states`). It emits a photon falling to any shell with a Pauli-allowed vacancy via a dipole-allowed (`Delta l = +-1`) transition (`spectra.dipole_transitions`) — the same construction that gives the hydrogenic Lyman/Balmer series, generalized to a screened potential, and the standard zeroth-order picture behind alkali/Rydberg spectra (e.g. Na's D-line as 3p->3s).

**Validated against known resonance lines** (`media/emission_spectra.png`, generated by a scratch script in this session, folded into `Hartree_Fock.ipynb`):
  - Na (`[Ne] 3s1`): predicted 3p->3s at 667.5 nm vs. experimental D-line 589.3 nm (+13%)
  - K (`[Ar] 4s1`): predicted 4p->4s at 874.1 nm vs. experimental D-line 766.5 nm (+14%)
  - Ar (`[Ne] 3s2 3p6`): predicted 4s->3p at 131.1 nm vs. experimental VUV resonance line 104.8 nm (+25%)

All three land in the right qualitative spectral region (Na/K: visible/near-IR; Ar: deep UV) and all three are consistently ~13-25% *too long a wavelength* (too small an energy gap) — a systematic, explainable Koopmans'-theorem-like error: the frozen ground-state core isn't relaxed around the excited electron+hole, which always under-estimates the true excitation energy. This is a genuine physical finding, not noise.

**Filtering needed to get clean output**: the local Xα potential's missing `-1/r` asymptotic tail (self-interaction error, same family of issue as DFT/LDA) leaves a few barely-bound "Rydberg" levels only ~1e-4 Ha deep that are numerical artifacts of the finite box, not real high-n states — without filtering these show up as spurious multi-micron "transitions". `dipole_transitions`'s `max_wavelength_nm=1500` cutoff removes them; this mattered more in practice than the `bound_cutoff` energy threshold.

**Known limitations**: no orbital relaxation (see Koopmans' error above); no spin-orbit coupling (single line per transition, not the fine-structure doublets real spectra show, e.g. Na D1/D2); single-configuration/single-active-electron only (no shake-up, no correlation, no full term-symbol structure). Documented in `spectra.py`'s module docstring and in `Hartree_Fock.ipynb`.

## Phase 8 — Genuine Kohn-Sham LDA (`potentials.pz81_correlation`, `scf.run_scf(method="lda")`) ✅ done

Despite the project's name, everything through Phase 7 is **not** actually
Hartree-Fock (no exact non-local exchange) and **not** actually Kohn-Sham
DFT either (Slater/Xα local exchange only, with a tunable `alpha` and no
correlation functional at all). This phase adds the one piece needed to
make the second of those true: a real correlation functional, paired with
exchange fixed at its exact theoretical value (`ALPHA_LDA=2/3`, already
present as a preset, just never used with a matching correlation term) --
turning the "tunable Xα local-exchange model" into genuine, parameter-free
Kohn-Sham LDA.

- **`potentials.pz81_correlation(rho)`**: the Perdew-Zunger (1981)
  parametrization of the correlation energy per electron `eps_c(rs)` (a fit
  to Ceperley-Alder quantum Monte Carlo data for the unpolarized electron
  gas, via the Wigner-Seitz radius `rs = (3/(4*pi*rho))**(1/3)`), plus its
  potential `V_c = eps_c - (rs/3)*d(eps_c)/d(rs)`. Two branches (`rs<1`
  logarithmic, `rs>=1` a rational Padé-like form), each independently
  curve-fit to the QMC data.
- **`potentials.lda_xc_potential`/`effective_potential_lda`**: combine
  `slater_exchange_potential(r, rho, ALPHA_LDA)` with `pz81_correlation`.
  Kept as new, separate functions rather than modifying
  `effective_potential`/`slater_exchange_potential` in place, so every
  existing Xα-based result above is untouched.
- **`scf.run_scf(Z, method="lda")`**: new opt-in `method` parameter
  (`"xalpha"`, the previous and still-default behavior, vs `"lda"`).
  `compute_total_energy` gained an analogous optional `eps_c`/`V_c` pair:
  since PZ81 correlation isn't a pure power law of `rho` the way Slater
  exchange is, there's no equivalent closed-form Euler's-theorem
  double-counting shortcut (the existing `-E_x/3` term) -- the general
  Kohn-Sham double-counting correction `E_c[rho] - integral(V_c*rho)` is
  evaluated directly instead when `method="lda"` supplies `eps_c`/`V_c`.
- **Bug found and fixed while validating**: far out on the radial grid the
  density underflows to exact float64 `0.0`; `pz81_correlation`'s
  `rs = (.../rho)**(1/3)` then hits a literal division by zero (`rs=inf`),
  and the low-density branch's `V_c` formula evaluates a literal `0*inf`,
  producing `NaN`. That `NaN` then poisoned the effective-potential matrix
  passed to `solve_radial_channel`, making `scipy.sparse.linalg.eigsh`'s
  internal `splu` factorization fail with `"Factor is exactly singular"` --
  first seen running Ne, not He (He's grid/density profile happened not to
  underflow within its range). Physically `rho=0` just means
  `eps_c, V_c -> 0` (the true limit of both branches as `rs -> infinity`);
  fixed by clipping `rho` to a `1e-300` floor before computing `rs`, far
  below any physically meaningful density.

**Validated**:
- *Correlation functional itself*: `eps_c`, `V_c` negative everywhere
  tested (`rs=0.5` to `20`); continuous across the `rs=1` branch boundary
  to the fit's own precision (~3e-5 Ha discontinuity, a known, tiny
  artifact of the published PZ81 parametrization -- its two branches were
  independently curve-fit and only approximately match at the boundary by
  construction, not a bug to chase to exact zero).
- *Self-interaction error (hydrogen, Z=1, one electron)*: exact HF is
  exact for one electron (exchange exactly cancels self-Hartree, already
  confirmed via the zeroed-potential check in `Radial SCF Validation.ipynb`).
  LDA has no such exact cancellation -- a real, textbook DFT limitation,
  cleanly isolated here with no many-electron physics to confound it:
  exact-LDA-exchange-only gives `E=-0.40652 Ha` (SIE `+0.09348 Ha` vs. the
  exact `-0.5 Ha`); adding PZ81 correlation improves this to `E=-0.44588
  Ha` (SIE `+0.05412 Ha`) -- correlation partially, not fully, compensates
  the self-interaction error, consistent with the DFT literature.
- *Closed-shell atoms (He, Ne, Ar)*: LDA total energies land closer to (but
  still short of) literature non-relativistic HF than Xα(Schwarz) does for
  every atom tested -- `He: LDA=-2.83418 vs Xα=-2.76637 vs HF=-2.86168`;
  `Ne: LDA=-128.22351 vs Xα=-128.03490 vs HF=-128.54700`; `Ar:
  LDA=-525.92506 vs Xα=-525.89503 vs HF=-526.81800` Ha. Reported honestly
  as a known DFT phenomenon (fortuitous partial cancellation between LDA's
  exchange underestimate and correlation overestimate), not evidence that
  LDA's individual exchange or correlation pieces are more accurate than
  exact HF exchange -- the hydrogen SIE check above is the more diagnostic,
  uncompensated measurement of LDA's actual error character.
- All three LDA runs integrate to the exact electron count
  (`Q(r_max) = N` to 6 decimal places) and reproduce the unchanged Xα
  numbers exactly when re-run through `method="xalpha"` (regression check).

See `SCF Validation.ipynb` section 5 for the runnable validation.

## Verification

1. ✅ Run `Radial SCF Validation.ipynb` first: confirm exact hydrogen energy match and l-degeneracy (Phase 2's acid test) before trusting anything downstream.
2. ✅ In `scf.py`, run `run_scf(Z)` for He, Ne, Ar and confirm convergence within the stated tolerances and `Q(r_max) ≈ N` each iteration. See `SCF Validation.ipynb` and `media/phase3_6_scf_validation.png`.
3. ✅ Run `run_scf` for Cr and Cu and confirm the hardcoded exception configurations are actually used (print/inspect occupations). Also spot-checked Fe (normal Madelung d-block) and Gd (f-block exception, confirms l_max=3/f-orbital support end-to-end) directly — not folded into the notebook since Gd takes ~22 minutes at the default N=4000 grid (noted as a performance consideration for Phase 7, not a correctness issue).
4. ✅ Ran `Hartree_Fock.ipynb` end-to-end for Ar (movie + full visualization suite) and Na/Ar/K (emission spectra vs. literature resonance lines) — see the Phase 7 sections above for what was produced and how it checks out.
5. ✅ Ran `SCF Validation.ipynb` section 5 end-to-end: `pz81_correlation` sign/continuity checks, hydrogen self-interaction-error comparison, and `method="lda"` vs `method="xalpha"` vs literature HF for He/Ne/Ar — see Phase 8 above. Confirmed `method="xalpha"` (the default) still reproduces every pre-existing number exactly (regression check).
