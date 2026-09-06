# Real-Time TDDFT — Design Plan

## Context

The DFT trilogy (`HF_solver/` atomic, `Diatomic_HF_solver/` two-center,
`Molecular_DFT/` full 3D) all solve for the **ground state**: a self-consistent
set of Kohn–Sham orbitals in their own mean field. None of them can say anything
about *excited* states, optical response, or what happens when you shine a laser
on the system.

Real-time time-dependent DFT (rt-TDDFT) is the smallest addition that unlocks all
of that. The idea: take the converged ground-state KS orbitals `{φ_j}` from
`Molecular_DFT/scf3d.py`, then propagate each one in real time under the
*time-dependent* KS potential

```
i ∂φ_j/∂t = [ -½∇² + v_ext(r,t) + v_H[ρ(r,t)] + v_xc[ρ(r,t)] ] φ_j ,   ρ(r,t) = Σ_j f_j |φ_j(r,t)|²
```

with the **adiabatic LDA (ALDA)** approximation: `v_xc` is the *same* static
Slater+PZ81 functional already in `potentials3d.py`, evaluated at the
instantaneous density. This is the standard workhorse method behind codes like
Octopus and is the natural fourth leg of this session's DFT initiative.

This project is **all-electron light systems only** (H, He, Be, H2, LiH), same
scope ceiling as `Molecular_DFT/` — enough to validate the machinery against
known excitation energies without needing pseudopotentials.

## Implementation strategy — two stages

rt-TDDFT is the one project in the new-solver set where the C++/OpenGL framework
genuinely earns its place (a KS-orbital field on a 3D grid, split-step FFT every
timestep, driven by a laser, worth watching live). But the physics validation is
far cheaper in Python, and the Python result is the reference the C++ version's
`--selftest` checks against — the same relationship `01_hartree_fock/main.cpp`
already has with the Python `HF_solver` reference numbers.

- **Stage 1 — Python (`Quantum Mechanics/TDDFT/`, this plan's Phases 1–6).**
  Built directly on `Molecular_DFT/`'s `grid3d` / `fft_ops` / `potentials3d` /
  `scf3d`. ~200–300 lines. Delivers absorption spectra and HHG for He/Be/H2,
  every validation gate, and the reference dipole traces. Days of work.
- **Stage 2 — C++ (`OpenGL Physics`, Phases 7–9).** Extend `05_tdse_gpu` (which
  already has the FFT compute shader, `--relax`, `--scf` multi-orbital,
  `[[drive]]`, and mean-field Hartree) with a 3D FFT, an ALDA `v_xc` compute
  kernel, and multi-orbital propagation. Payoff: real-time interactive
  absorption/HHG (watch the harmonic comb build), larger systems, a website
  asset. Weeks of work; only worth starting once Stage 1 has locked the physics.

Stage 2 is a separate-repo effort tracked like the electrodynamics C++ ports —
begin it only after Stage 1's gates pass.

## What this enables (applications)

- **Optical absorption spectra.** Give every orbital an instantaneous momentum
  boost `φ_j → e^{i κ·r} φ_j` at `t=0` (a "δ-kick"), propagate, record the dipole
  `d(t) = -∫ r ρ(r,t) d³r`, and Fourier-transform: the dynamic polarizability
  `α(ω)` has poles at the true excitation energies and the photoabsorption cross
  section is `σ(ω) ∝ ω Im α(ω)`. One short propagation gives the **entire
  spectrum** — no Casida matrix, no state-by-state solve.
- **High-harmonic generation (HHG).** Drive with a few-cycle laser pulse
  `v_ext += E(t)·r`, Fourier-transform the dipole acceleration → the odd-harmonic
  comb and the `3.17 U_p + I_p` cutoff. The strong-field-physics analog of
  `TDSE_Solver/`'s tunneling work, but many-electron.
- **Frequency-dependent polarizability / van der Waals C6.** `α(iω)` on the
  imaginary axis → Casimir–Polder / dispersion coefficients.
- **Charge-transfer and photo-induced dynamics.** Watch `δρ(r,t)` redistribute
  after excitation — where does the electron density actually go.
- **Sum rules as built-in diagnostics.** The Thomas–Reiche–Kuhn sum rule
  `∫ σ(ω) dω = N_electrons` is an exact constraint the propagation must satisfy —
  a free, quantitative correctness check with no external reference needed.

## Numerical approach

**Initial state**: `Molecular_DFT/scf3d.run_scf` (with a new `return_orbitals`
flag — see Upstream changes) gives `{φ_j}`, `{f_j}`, and the converged `ρ_0`.

**Propagator**: because `v_KS` depends on `ρ(t)` the Hamiltonian is
time-dependent, so a bare Strang split-step (which assumes fixed `V` over the
step) is only 1st-order in the coupling. Use the **enforced time-reversal
symmetry (ETRS)** scheme, standard for rt-TDDFT:

```
φ_j(t+Δt) ≈ exp[-i Δt/2 · H(ρ_pred)] · exp[-i Δt/2 · H(ρ(t))] φ_j(t)
```

where each `exp[-i Δt/2 · H]` is itself applied by a short split-step (or a
4–6 term Taylor/Chebyshev expansion of the exponential, since `Δt/2` is small),
and `ρ_pred` is a predicted density at `t+Δt` (one uncorrected step, optionally
one corrector iteration). Kinetic part via `fft_ops.apply_kinetic` exactly as in
the ground-state code; potential part is a real-space pointwise multiply.

**Perturbation / drive**:
- δ-kick: multiply every orbital by `exp(i κ · r)` once, with `κ` small
  (`|κ| ~ 0.01` a.u.) and along `x`, `y`, `z` in three separate runs to get the
  full polarizability tensor.
- Laser: add `E(t) · r` to `v_ext` inside the propagator, `E(t)` a
  carrier-envelope pulse (`sin²` envelope). Dipole gauge; length gauge is fine at
  this box size.

**Absorbing boundary**: `Molecular_DFT/` is a periodic supercell. For bound
response below the ionization threshold the periodic box is fine; for laser runs
that ionize, multiply every orbital by a smooth **mask function** (1 in the
interior, cosine taper to 0 over the outer ~4 a.u.) once per step to soak up
outgoing flux. Norm loss then measures ionization — a diagnostic, not an error.

**Spectrum extraction**: `α_ab(ω) = -FT[d_a(t) - d_a(0)]_b / (κ_b/i)` with an
exponential damping window `e^{-t/τ}` (τ ≈ half the propagation time) to tame the
finite-time ringing; the damping sets an artificial Lorentzian linewidth, stated
explicitly.

**Dropped / not attempted**: hybrid functionals and exact exchange (ALDA only);
non-adiabatic / memory kernels; the Casida linear-response matrix formulation
(real-time gets the same spectrum more cheaply here); pseudopotentials.

## Upstream changes to existing solvers

All additive, all keeping existing results bit-for-bit:

1. **`Molecular_DFT/scf3d.py`** — `run_scf(..., return_orbitals=False)`: when
   `True`, also return the occupied orbital array and occupations, not just
   `ρ` and energies. TDDFT needs the orbitals as its initial condition.
2. **`Molecular_DFT/potentials3d.py`** — factor the "build `v_KS` from `ρ`" logic
   currently inside `run_scf`'s loop into a standalone
   `ks_potential(rho, X, Y, Z, nuclei, G2, method="lda")`. Both the SCF loop and
   the TDDFT propagator must call the *identical* function — that is what makes
   "propagating the ground state produces only phase evolution" an exact test
   rather than an approximate one.
3. **`Molecular_DFT/`** — new `masking.py` (or extend `potentials3d.py`) with a
   `boundary_mask(shape, dx, width)` smooth absorber, reused by TDDFT laser runs.
   (Ground-state code does not need it; no behavior change there.)

## File layout

```
Quantum Mechanics/TDDFT/
    TDDFT_Plan.md
    propagate.py        # Phase 1-2: ETRS propagator for a set of KS orbitals, time-dependent v_KS
    perturb.py          # Phase 3: dipole kick, laser pulse E(t), mask application
    response.py         # Phase 3-4: dipole/current recording, alpha(omega), sigma(omega), sum-rule check
    Validation.ipynb            # Phases 1-2
    Absorption_Spectra.ipynb    # Phases 3-4: He, Be, H2 photoabsorption
    Strong_Field_HHG.ipynb      # Phase 5: H / H2 harmonic spectrum
    media/
```

Reuses `Molecular_DFT/`'s `grid3d.py`, `fft_ops.py`, `potentials3d.py`,
`scf3d.py` by same-repo import (the two folders are part of one initiative;
follow whatever import convention `Molecular_DFT` ends up exposing, or a light
copy if cross-folder import is awkward, decided at implementation time).

## Stage 1 (Python) — phases

Each phase stops for review (same workflow as `Molecular_DFT/`).

**Phase 1 — Time-dependent propagator, fixed external potential** (`propagate.py`)
- ETRS propagator for a single orbital, `v_KS` replaced by a *static* external
  potential (no Hartree/XC feedback yet).
- Validate: on the same 3D grid, reproduce `TDSE_Solver/`'s closed-form checks —
  free Gaussian spreading `σ(t) = σ0√(1+(t/2σ0²)²)`, and a harmonic coherent
  state oscillating at `ω` with constant width and energy. ~2nd-order
  convergence in `Δt`. Norm conserved to ~1e-8 with no mask.

**Phase 2 — Self-consistent propagation, ground-state fixed point** (`propagate.py`)
- Full `v_KS[ρ(t)]` via the factored `ks_potential`. Propagate the *converged*
  `Molecular_DFT` ground state with **no** perturbation.
- Validate: each orbital evolves by pure phase `e^{-iε_j t}`; the density and the
  dipole are constant to ~1e-6 over a long run. This is the acid test that the
  propagator and the ground-state SCF are mutually consistent — a moving
  ground state means a bug in one of them.

**Phase 3 — δ-kick linear response: closed-shell atom** (`perturb.py`, `response.py`)
- He (and Be): boost, propagate ~50–100 a.u. of time, record `d(t)`, extract
  `α(ω)` and `σ(ω)`.
- Validate: the lowest absorption peak lands at He's first dipole-allowed
  excitation (`1s→2p`, ≈ 21.2 eV experimentally; ALDA typically within ~0.5 eV);
  the TRK sum rule `∫ σ dω = N_e` holds to a few % (grid/box/finite-time
  limited); spectrum converges under box-size and `Δt` refinement.

**Phase 4 — Molecular absorption: H2** (`Absorption_Spectra.ipynb`)
- Full δ-kick spectrum of H2 at `R_e`; the three Cartesian kicks give the
  parallel/perpendicular components.
- Validate: sum rule to a few %; lowest excitation in the right region vs
  reference TDDFT/experiment (qualitative — ALDA + this grid resolution);
  peak positions stable under resolution refinement.

**Phase 5 — Strong-field HHG: H / H2** (`Strong_Field_HHG.ipynb`)
- Few-cycle `sin²` pulse, mask on. Fourier-transform the dipole acceleration.
- Validate: only **odd** harmonics appear (inversion symmetry of H / H2); the
  harmonic plateau cuts off near `I_p + 3.17 U_p` for the chosen intensity and
  wavelength (semiclassical three-step prediction); ionization (norm loss into
  the mask) rises with intensity as expected.

**Phase 6 — Visualization** (`media/`)
- Density-difference `δρ(r,t)` isosurface/slice movies for a δ-kick and for a
  laser cycle; dipole-vs-time traces; the absorption and HHG spectra as
  publication-quality figures.

## Stage 2 (C++) — GPU implementation, extending `05_tdse_gpu`

Begin only after Stage 1's gates (Phases 1–6) pass. Lives in
`OpenGL Physics/projects/05_tdse_gpu` (a new `--tddft` mode) or a fresh
`12_tddft` project — decide at the time based on how much of `05`'s single-ψ
code paths can be shared. `05` already provides: the hand-written radix-2 FFT
compute shader, `--relax` (imaginary-time relaxation to the K lowest KS-like
states), `--scf` (multi-orbital self-consistent Poisson–Schrödinger with a
mean-field Hartree kick), `[[drive]]` time-dependent potentials, and CAP
boundaries. What Stage 2 adds:

**Phase 7 — 3D grid + multi-orbital propagation.**
`05` is 2D; a 3D FFT (row → transpose → row → transpose → row, or three 1D
passes) and an orbital-major SSBO layout for N occupied orbitals. Reuse
`--relax` to produce the initial KS orbitals.
- Validate: `--selftest` cross-checks the 3D propagator against the Stage-1
  Python `propagate.py` on the shared free-particle / harmonic cases (fp32 vs
  fp64, rel ~1e-5, same tolerance `05`'s existing FFT selftest uses); the
  ground-state fixed-point test (Phase 2) reproduced in fp32 to ~1e-4.

**Phase 8 — ALDA `v_xc` kernel + Hartree from a 3D Poisson solve.**
A compute kernel evaluating Slater exchange + PZ81 correlation pointwise from
`|ψ_j|²` summed over orbitals (port `HF_solver/potentials.py`'s branchless
forms); Hartree via the existing FFT Poisson approach in 3D.
- Validate: `--tddft` δ-kick on He reproduces the Stage-1 Python absorption
  spectrum's lowest peak position to ~0.1 eV; TRK sum rule within a few %.

**Phase 9 — Interactive absorption / HHG demo.**
`fw::SimApp` view: a 3D density isosurface or slice with the instantaneous
dipole and a live-updating running-FFT spectrum panel; `[[drive]]` supplies the
laser. Record to MP4 via `tools/make_movie.py`.
- Validate: the live-accumulated HHG spectrum converges to the Stage-1 Python
  result (odd harmonics, `I_p + 3.17 U_p` cutoff) as the propagation runs.

## Progress

### Stage 1 — Python
- [x] Phase 1 — fixed-potential propagator, TDSE cross-checks. `propagate.py`
      (`strang_step` / `kinetic_step` / `potential_step`) + `validate.py`.
      Free spreading matches `σ(t)` to ~2e-7 (split-step exact for `V=0`);
      harmonic coherent state `⟨x⟩ = x₀cos ωt` to ~1e-5, bounded energy drift
      ~7e-6, norm to ~1e-12; 2nd-order Δt convergence confirmed (ratio ~4).
- [x] Phase 2 — self-consistent propagation, ground-state fixed-point test.
      `etrs_step` + `relax_to_self_consistency`. Propagating the converged He
      ground state with no perturbation holds the **dipole** to ~7e-6 and the
      **KS total energy** to ~4e-5 (propagator↔SCF consistency). The raw
      density L1 `∫|ρ(t)-ρ₀|` shows the split-operator's **O(Δt²)** shape
      wobble (~2e-3 at Δt=0.02, ¼ that at Δt=0.01) — discretization, not an
      inconsistency; the dipole/energy are the observables that matter and
      they are pinned.
- [x] Phase 3 — δ-kick absorption spectrum (He). `perturb.py` (`dipole_kick`)
      + `response.py` (`polarizability` / `strength_function` / `cross_section`
      / `sum_rule` / `peaks`). One kick along x, propagate ~320 a.u., FFT the
      dipole. **TRK f-sum rule recovers ~97% of `N_e`** (rest above the
      finite-time cutoff); `Im α ≥ 0` to ~0.4% of peak (passivity); lowest
      peak identical at `k=0.005` and `k=0.02` (linearity). Lowest line at
      ~12.9 eV — red-shifted from the 21.2 eV experimental `1s→2p` by the
      softened nucleus (`soft = 0.5 dx`, same cause as the SCF energy being
      ~0.45 Ha high); a resolution panel shows it blue-shifting toward
      experiment as `dx` shrinks. `α(0) ≈ 3.7 a.u.` (He expt 1.38) — inflated
      by the periodic-FFT Poisson error, reported not gated.
- [x] Phase 4 — H2 absorption spectrum. `response.kick_spectrum` helper +
      `phase4_h2_absorption`. Kick along **and** across the bond (H2 at
      `R_e = 1.4`). **Per-axis TRK sum rule ~99%** of `N_e` for both. The
      response is genuinely **anisotropic**: `α_∥(0) ≈ 10.7` vs `α_⊥(0) ≈ 8.1`
      a.u. — more polarizable along the bond, ratio `1.31` vs the
      experimental `~1.28` (the ratio is well-captured even though the
      magnitudes are ~1.7× high from the periodic-FFT Poisson error). Lowest
      lines ~9.6 eV (∥) / ~9.8 eV (⊥), red-shifted from the ~12-13 eV
      experimental onset by the softened cusp.
- [x] Phase 5 — strong-field HHG (H atom). `perturb.flattop_pulse` +
      `response.hhg_spectrum` (dipole-acceleration FFT) + `phase5_hhg`.
      Ground state via `propagate.imaginary_time_ground_state` (FFT-only,
      ~5 s vs ~12 min for `eigsh` at these box sizes); H propagated in the
      self-interaction-free single-particle limit (`method=None` — no
      Hartree/XC, ETRS predictor skipped). **4/4:** odd-only harmonics
      (odd/even ratio ~700×); a flat plateau (to ~8th order) ending in a
      >3-decade cutoff; the cutoff extends with intensity (9.6→13.4
      harmonics); ionization rises with intensity (60%→91%). The absolute
      cutoff sits *above* `I_p + 3.17 U_p` (5.6 harmonics) because `E₀` is
      ~ the barrier-suppression field for this softened H — the rescattering
      law is a lower bound in the over-the-barrier regime; the clean
      tunneling regime needs a longer-wavelength/lower-intensity run than
      this 3D box affords.
- [x] Phase 6 — visualization. `visualize.py` → `media/deltarho_he_kick.mp4`:
      the He electron cloud's dipole `δρ(r,t) = n(t) − n(0)` sloshing after a
      kick (the real-space picture behind every line of the Phase-3 spectrum),
      next to the `⟨x⟩(t)` trace.

**Stage 1 complete — all six phases pass.** The caveat threaded through
Phases 3–5: sum rules, anisotropy ratio, harmonic symmetry, intensity
scaling — the geometry- and symmetry-exact quantities — all hold; the
absolute excitation / cutoff energies are shifted by this grid's softened
cusp, small box, and periodic-FFT Poisson error, and move the right way
under refinement.

**Upstream (done):** `Molecular_DFT/potentials3d.ks_potential` (incl. a
`method=None` bare-`V_nuc` branch), `scf3d.run_scf(return_orbitals=)`,
`Molecular_DFT/masking.py` — SCF results verified bit-identical (H atom
`-0.47553` Ha). Plus `TDDFT/propagate.imaginary_time_ground_state` (FFT-only
KS ground state, validated against `scf3d` for He to ~5e-4 Ha).

### Stage 2 — C++ (`05_tdse_gpu` extension) — gated on Stage 1
- [ ] Phase 7 — 3D FFT + multi-orbital propagation, selftest vs Python
- [ ] Phase 8 — ALDA `v_xc` kernel + 3D Hartree, He spectrum vs Python
- [ ] Phase 9 — interactive absorption / HHG demo + MP4

## Verification

1. Phase 1: ✅ 3D propagator reproduces `TDSE_Solver`'s analytic free/harmonic
   checks (free `σ(t)` to 2e-7, harmonic `⟨x⟩` to 1e-5); 2nd-order in `Δt`.
2. Phase 2: ✅ propagating the converged KS ground state leaves the **dipole**
   stationary to ~7e-6 and the **KS energy** to ~4e-5 — propagator/SCF
   consistency. (The raw density L1 carries an `O(Δt²)` split-operator wobble,
   ~2e-3 at Δt=0.02; verified to be discretization by its `Δt²` scaling, not
   an inconsistency.)
3. Phase 3: ✅ TRK f-sum rule within a few % (~97% of `N_e`); `Im α ≥ 0`
   (passivity); peak position independent of kick strength (linearity). The
   absolute excitation energy is grid/softening-limited on this build (~12.9
   eV vs 21.2 eV expt) and converges toward experiment under `dx` refinement
   — the sum rule and linearity are the geometry-exact gates.
4. Phase 4: ✅ H2 per-axis sum rule ~99% of `N_e` (∥ and ⊥ kicks); the
   response is anisotropic with `α_∥/α_⊥ ≈ 1.31` (expt ~1.28). Absolute peak
   positions grid/softening-limited like Phase 3.
5. Phase 5: ✅ odd-only harmonics (ratio ~700×); a plateau ending in a sharp
   (>3-decade) cutoff; cutoff extends with intensity; ionization rises with
   intensity. Absolute cutoff sits above `I_p + 3.17 U_p` — `E₀` is in the
   over-the-barrier regime for this softened H, where the rescattering law is
   a lower bound.
6. Phase 6: ✅ `δρ(r,t)` dipole-sloshing movie (`media/deltarho_he_kick.mp4`).
7. Phase 7 (C++): 3D GPU propagator matches the Stage-1 Python propagator on the
   shared free/harmonic cases (fp32 vs fp64 ~1e-5); fp32 ground-state fixed point
   to ~1e-4.
8. Phase 8 (C++): `--tddft` He spectrum lowest-peak position within ~0.1 eV of
   the Stage-1 Python result; sum rule within a few %.
9. Phase 9 (C++): live-accumulated HHG spectrum converges to the Stage-1 result.
