# Born–Oppenheimer Nuclear Dynamics — Design Plan

## Context

`Diatomic_HF_solver/` and `Molecular_DFT/` both produce **Born–Oppenheimer
potential energy curves** `E(R)` — the electronic energy as a function of fixed
nuclear separation (the bond-length scans already exist in their validation
notebooks). Those curves *are* the potential the nuclei move in. Nothing in the
repo currently does anything with them beyond reading off the minimum.

This project closes that loop: solve the **nuclear** Schrödinger equation on the
internuclear coordinate `R` with reduced mass `μ = M_A M_B / (M_A + M_B)`,

```
[ -1/(2μ) d²/dR² + ħ²J(J+1)/(2μR²) + E(R) ] χ(R) = E_vib χ(R)
```

reusing `TDSE_Solver/`'s split-step propagator and `stationary_states.py`'s
finite-difference eigensolver essentially unchanged. It is the bridge between the
electronic-structure half of the repo and the wavepacket-dynamics half — two
finished solver families, joined by a thin layer.

## What this enables (applications)

- **Vibrational spectra.** Bound states of the PES well give `v = 0, 1, 2, …`
  energies. The `0→1` spacing is `ω_e`; the shrinking of successive spacings is
  the anharmonicity `ω_e x_e`; the `v=0` energy above the well bottom is the
  zero-point energy. Directly comparable to spectroscopic constants (H2:
  `ω_e ≈ 4401 cm⁻¹`).
- **Rotational structure and isotope effects.** The centrifugal term gives the
  rotational constant `B_e ∝ ⟨1/R²⟩`; swapping `μ` (H2 → D2 → HD) shifts every
  level by a known factor — a parameter-free check.
- **Franck–Condon factors.** Overlaps `|⟨χ_{v'}^{upper} | χ_0^{lower}⟩|²` between
  vibrational states on two different electronic PESs set the intensity envelope
  of a vibronic band (the "progression" seen in molecular absorption/emission).
- **Photodissociation dynamics.** Launch a wavepacket onto a repulsive or
  above-threshold PES, propagate, and watch it fly apart: dissociation lifetime,
  kinetic-energy release, branching ratios.
- **Predissociation / tunneling decay.** Quasi-bound levels trapped behind a
  rotational barrier (`J` large) or an avoided-crossing barrier leak out with a
  finite lifetime — the exact physics of `Bound_States_1D/`'s alpha-decay
  notebook, now in a molecular context.
- **Vibrational wavepacket motion and revivals.** A coherent superposition
  sloshes in the well; anharmonicity dephases it and then (partially) revives it.

## Numerical approach

**Potential**: the electronic solvers return `E` at a discrete set of `R`. Fit /
spline to a continuous `E(R)`:
- Interior: `scipy.interpolate.CubicSpline` through the computed points.
- Extrapolation past the scan range: a **Morse fit**
  `E(R) ≈ D_e [1 - e^{-a(R-R_e)}]² + E_∞` (physically correct asymptotics on both
  ends, and it gives closed-form vibrational levels for the Phase-1 validation).

**Stationary levels**: finite difference on a uniform `R`-grid, tridiagonal
`H` with `-1/(2μ) d²/dR²` and diagonal `E(R) + ħ²J(J+1)/(2μR²)`, solved with
`scipy.linalg.eigh_tridiagonal` (exactly `Bound_States_1D/`'s method) or
`TDSE_Solver/stationary_states.lowest_states` with the new `mass` argument.

**Dynamics**: `TDSE_Solver/propagator.strang_step` on the 1D `R`-grid with
`mass=μ`. For dissociation, a complex absorbing potential
(`potentials.absorbing_boundary`) at large `R` acts as the detector; absorbed
norm integrated against outgoing momentum gives the KER distribution.

**Two-surface problems** (Franck–Condon dynamics, predissociation with coupling):
a 2-channel propagator — diabatic potentials `V_1(R)`, `V_2(R)` and coupling
`V_{12}(R)`. The split-step generalizes cleanly: the potential half-step becomes
a pointwise `2×2` matrix exponential (`expm` of a `2×2` is closed-form, no
scipy needed).

**Reduced mass in atomic units**: nuclear masses in electron-mass units
(`m_p ≈ 1836.15 m_e`), so `μ` for H2 is `≈ 918`. The kinetic term is
`~2000×` weaker than the electronic case — vibrational spacings come out
`~0.01` Ha, rotational `~1e-4` Ha, which sets the grid and `dt` scales.

**Not attempted**: full ro-vibrational coupling beyond the parametric-`J`
treatment; non-adiabatic dynamics with more than 2 surfaces; polyatomic normal
modes (this is the diatomic / single-reaction-coordinate case only, matching
`Diatomic_HF_solver`'s scope).

## Upstream changes to existing solvers

1. **`TDSE_Solver/propagator.py`** — add a `mass=1.0` parameter to
   `kinetic_eigenvalues`, `kinetic_step`, `strang_step` (`k²/2 → k²/(2·mass)`).
   Default `1.0` reproduces every existing TDSE result unchanged.
2. **`TDSE_Solver/stationary_states.py`** — same `mass=1.0` parameter on
   `laplacian_matrix` / `hamiltonian` / `lowest_states`.
3. **`TDSE_Solver/potentials.py`** — new
   `potential_from_samples(grid, R_samples, E_samples, fill="morse")`: splines a
   tabulated PES onto the solver grid with physical extrapolation. Small,
   genuinely reusable.
4. **`Diatomic_HF_solver/diatomic_driver.py`** (and analogously
   `Molecular_DFT/scf3d.py`) — promote the bond-scan loop currently written
   inline in the validation notebooks to a function
   `scan_pes(Z_A, Z_B, R_values, method="lda", **kw) -> (R_array, E_array)` that
   also writes `media/pes_<system>.csv`. No physics change; just makes the PES a
   clean importable product.

## File layout

```
Quantum Mechanics/Nuclear_Dynamics/
    Nuclear_Dynamics_Plan.md
    pes.py             # Phase 1: load/spline/Morse-fit a tabulated E(R); reduced masses; spectroscopic-constant extraction
    vibrational.py     # Phase 2-3: nuclear grid + reduced-mass H, bound levels, <O>, rotational term
    dynamics.py        # Phase 4-6: 1-surface and 2-surface split-step propagation, FC factors, dissociation analysis
    Validation.ipynb              # Phase 1: Morse closed-form check
    H2_Vibration_Rotation.ipynb   # Phase 2-3: H2/D2 levels vs spectroscopic constants
    Wavepacket_and_FranckCondon.ipynb  # Phase 4-5
    Photodissociation.ipynb       # Phase 6
    media/
```

Imports `TDSE_Solver/`'s `grid.py`, `propagator.py`, `stationary_states.py`,
`observables.py`, `potentials.py`; consumes PES CSVs from `Diatomic_HF_solver/`
/ `Molecular_DFT/` (data files, not code).

## Phases

**Phase 1 — Nuclear grid + reduced-mass kinetic operator** (`pes.py`, `vibrational.py`)
- Build `H` on the `R`-grid with `mass=μ`; ingest a Morse potential analytically.
- Validate: computed levels match the closed-form Morse spectrum
  `E_v = ω_e(v+½) - ω_e x_e (v+½)²` (with `ω_e`, `ω_e x_e` from `D_e`, `a`, `μ`)
  to ~1e-6 relative for the bound levels; number of bound states matches the
  Morse count `⌊(2D_e/ω_e - 1)/2⌋`.

**Phase 2 — Real PES → vibrational levels** (`vibrational.py`, `H2_Vibration_Rotation.ipynb`)
- Spline the H2 `Diatomic_HF_solver` LDA bond curve; solve `J=0` levels.
- Validate: `v=0→1` spacing vs H2 `ω_e ≈ 4401 cm⁻¹` and `ω_e x_e ≈ 121 cm⁻¹`
  (expect a few % — the PES itself is LDA, honest comparison in the same spirit
  as the rest of the repo); dissociation energy `D_0` = (well depth) − ZPE vs the
  experimental `4.48 eV`.

**Phase 3 — Rotational structure & isotopes** (`vibrational.py`)
- Add the centrifugal term; compute `B_e` from `⟨1/R²⟩_{v=0}` and the
  vibration–rotation coupling `α_e` from `B_v` vs `v`.
- Validate: `B_e` vs H2's `≈ 60.85 cm⁻¹`; re-run with `μ_{D2}`, `μ_{HD}` and
  confirm levels scale as `μ^{-1/2}` (vibration) and `μ^{-1}` (rotation) to the
  precision of the shared PES.

**Phase 4 — Vibrational wavepacket dynamics** (`dynamics.py`)
- Coherent superposition of the lowest few levels; propagate with `mass=μ`.
- Validate: `⟨R⟩(t)` oscillates near `ω_e`; energy conserved to ~1e-8; the
  packet dephases over ~(anharmonic) timescale and shows a partial revival at the
  predicted revival time `T_rev ≈ 2π/(ω_e x_e)`. MP4.

**Phase 5 — Franck–Condon factors** (`dynamics.py`, `Wavepacket_and_FranckCondon.ipynb`)
- Two PESs: the H2 ground curve and a model excited curve (shifted `R_e`, altered
  `D_e`). Compute `|⟨χ_{v'} | χ_0⟩|²` for `v' = 0..10`.
- Validate: the FC envelope peaks at the `v'` whose classical turning point lines
  up with the lower state's `R_e` (the reflection/vertical-transition principle);
  `Σ_{v'} FC = 1` over a complete-enough upper manifold. MP4 of the vertical
  transition and the resulting wavepacket motion on the upper surface.

**Phase 6 — Photodissociation** (`dynamics.py`, `Photodissociation.ipynb`)
- Wavepacket placed above the dissociation asymptote (or vertically onto a
  purely repulsive model curve); CAP at large `R` as the detector.
- Validate: total absorbed norm → 1 (everything dissociates); the KER extracted
  from the absorbed flux equals `E_photon − D_0` (energy conservation); for a
  rotationally trapped quasi-bound case, the decay is exponential with a lifetime
  that lengthens as `J` decreases. MP4.

## Progress

- [ ] Phase 1 — reduced-mass grid, Morse closed-form validation
- [ ] Phase 2 — H2 vibrational levels vs spectroscopic constants
- [ ] Phase 3 — rotational constants + isotope scaling
- [ ] Phase 4 — vibrational wavepacket, revivals
- [ ] Phase 5 — Franck–Condon factors, two-surface dynamics
- [ ] Phase 6 — photodissociation, KER

## Verification

1. Phase 1: computed levels reproduce the exact Morse spectrum and bound-state
   count.
2. Phase 2: H2 `ω_e`, `ω_e x_e`, `D_0` within a few % of experiment (PES-limited).
3. Phase 3: `B_e` matches experiment; D2/HD levels scale with the correct
   `μ`-powers.
4. Phase 4: energy conservation; revival at `2π/(ω_e x_e)`.
5. Phase 5: FC sum rule; envelope peak at the vertical-transition `v'`.
6. Phase 6: KER `= E_photon − D_0`; exponential predissociation lifetime vs `J`.
