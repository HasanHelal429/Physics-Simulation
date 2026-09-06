# Real-Time TDDFT (Stage 1, Python)

Propagates the Kohn-Sham orbitals from `../Molecular_DFT/` in real time under
the adiabatic-LDA time-dependent KS potential, to get excited-state and
light-matter response the ground-state DFT trilogy cannot reach — optical
absorption spectra (δ-kick + Fourier the dipole), high-harmonic generation
(laser drive), and density-response dynamics.

See `TDDFT_Plan.md` for the full design, the two-stage strategy (this is
Stage 1; Stage 2 is a GPU port), and the phase-by-phase validation gates.

## Status

| phase | what | state |
|---|---|---|
| 1 | bare propagator vs `TDSE_Solver/` closed forms | **done** |
| 2 | self-consistent propagation, ground-state fixed point | **done** |
| 3 | δ-kick absorption spectrum (He) | **done** |
| 4 | H2 absorption spectrum | not started |
| 5 | strong-field HHG (H, H2) | not started |
| 6 | visualization | not started |

## Layout

```
propagate.py   the propagator: strang_step (fixed V, Phase 1),
               etrs_step (density-dependent V_KS, Phase 2+), density(),
               observables, relax_to_self_consistency(), propagate() driver
perturb.py     dipole_kick (Phase 3), sin2_pulse / dipole_field (Phase 5)
response.py    polarizability alpha(w), strength_function S(w),
               cross_section sigma(w), TRK sum_rule, peak-finder
validate.py    headless Phase 1-3 checks -> media/ figures + PASS/FAIL table
media/         validation figures
```

Depends on `../Molecular_DFT/` (`grid3d`, `fft_ops`, `potentials3d`, `scf3d`)
— added to `sys.path` at import. These upstream additions were made for it:

- `potentials3d.ks_potential(rho, V_nuc, G2, method, alpha)` — the KS effective
  potential, factored out of `scf3d.run_scf`'s loop so the SCF and the
  propagator build `V_eff` from *identical* code (that identity is what makes
  Phase 2's fixed-point test exact).
- `scf3d.run_scf(..., return_orbitals=True)` — also returns the converged
  occupied orbitals + occupations.
- `masking.py` — `boundary_mask()`, a smooth absorbing boundary for Phase 5.

## Run

```sh
python validate.py --phase all          # ~5-6 min (a Helium SCF dominates)
python validate.py --phase 1 --quick    # ~30 s, propagator checks only
```

## Method (Phases 1-2)

**Propagator.** Kinetic step `exp(-i G²Δt/2)` exactly in Fourier space
(`propagate.kinetic_step`), potential step `exp(-iVΔt)` pointwise. Phase 1's
fixed potential → a symmetric Strang split-step (2nd order, exactly unitary).
Phase 2+ → the **ETRS** scheme: a predictor step with `V₀ = V_KS[ρ(t)]`
estimates `ρ(t+Δt)`, then the real step uses `V₁ = V_KS[ρ_pred]` on the
left half-kick — needed because `V_KS` depends on the evolving density.

**Phase 1 validation** (`--quick`, ~30 s): on the 3D periodic grid,

- free Gaussian wavepacket spreads as `σ(t) = σ₀√(1+(t/2σ₀²)²)` — matches to
  ~2e-7 (split-step is exact for `V=0`); `⟨r⟩ = k₀t`; norm to 5e-14;
- harmonic coherent state: `⟨x⟩(t) = x₀cos ωt` to ~1e-5, constant width,
  bounded energy drift ~7e-6 (no secular growth), norm to 1e-12;
- the harmonic case's final-time error falls ~4× per Δt halving (2nd order).

**Phase 2 validation** (~1.5 min quick / ~5 min full): converge the
`Molecular_DFT` Helium ground state, refine it to a tight KS fixed point
(`relax_to_self_consistency`), then propagate it with **no** perturbation.
The dipole stays pinned to ~7e-6 and the KS total energy to ~4e-5 — proving
the propagator and the SCF are mutually consistent (a moving ground state
would mean a bug in one of them). The raw density L1 `∫|ρ(t)-ρ₀|` picks up
the split-operator's `O(Δt²)` shape wobble (verified: halving Δt → ¼ the
wobble), which is discretization, not inconsistency.

**Phase 3 — δ-kick absorption (He).** Boost every orbital by `exp(i k x)`
(`k = 0.01`), propagate ~320 a.u., record `d_x(t) = ∫x n d³r`, and
`response.polarizability` Fourier-transforms `d_x(t)-d_x(0)` (× an
`e^{-t/τ}` damping window) into `α(ω)`; `S(ω) = (2ω/π) Im α`. Checks:

- **TRK f-sum rule** `∫₀^∞ S(ω) dω = N_e` — the rigorous internal test;
  recovers ~96% of `N_e = 2` (the rest is above the finite-time cutoff).
- **passivity**: `Im α ≥ 0` to within the ~0.5%-of-peak between-line ripple.
- **linearity**: the lowest peak is identical at `k = 0.005` and `k = 0.02`.
- the lowest line sits in the bound-excitation region, **red-shifted** from
  the 21.2 eV experimental `1s→2p` because the softened nucleus
  (`soft = 0.5 dx`) under-binds the `1s` (same reason the SCF energy is
  ~0.45 Ha high). The full run's resolution panel shows it blue-shifting
  toward experiment as `dx` shrinks.

This grid's known limitations — softened cusp, small box, periodic-FFT
Poisson error (documented in `Molecular_DFT/fft_ops.solve_poisson`) — set
the quantitative accuracy; the sum rule and linearity are geometry-exact
and pass regardless.
