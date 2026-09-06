# Dirac / Relativistic Solver — Design Plan

## Context

`HF_solver/` repeatedly flags what non-relativistic quantum mechanics cannot do:
"Z > 80 qualitative only, no relativistic corrections"; the emission-spectra
extension produces "a single line per transition, not the fine-structure doublets
real spectra show, e.g. Na D1/D2". `Bound_States_1D/` and `TDSE_Solver/` tunnel
non-relativistically. The Dirac equation is the fix, and it fits this repo's
existing machinery well: the free Dirac Hamiltonian is diagonal in momentum
space, so `TDSE_Solver/`'s split-step Fourier method carries over directly, and
the radial Dirac problem slots into `HF_solver/atomic_scf.py`'s log-grid
generalized-eigenproblem pattern.

Three solvers, increasing in ambition:

1. **1D time-dependent Dirac** (2-spinor, 1+1 D) — split-step Fourier. The
   conceptual demos: Zitterbewegung, Klein tunneling.
2. **Radial Dirac–Coulomb** (stationary, hydrogen-like) — the coupled
   large/small-component ODE system, finite-difference on a log grid. Exact fine
   structure.
3. **Dirac–Fock–Slater** (self-consistent, many-electron atoms) — the radial
   Dirac solver wrapped in `HF_solver/`'s SCF loop. Relativistic atomic
   structure across the periodic table.

Scope: hydrogen-like ions and closed-/simple-open-shell neutral atoms. **QED
effects (Lamb shift, vacuum polarization, self-energy) are explicitly out of
scope** — Dirac is the relativistic *single-particle* theory; the Lamb shift is
the next layer beyond it and is noted, not attempted.

## Language: Python primary, one optional C++ visual demo

Everything here is **Python** (`Quantum Mechanics/Dirac_Solver/`), sibling to
`HF_solver/` and reusing its SCF machinery — the radial and self-consistent parts
(Phases 3–6) are one-shot solves where research iteration speed matters and the
C++/OpenGL framework adds nothing.

The **1D time-dependent Dirac** work (Phases 1–2: Zitterbewegung, Klein
tunneling) is also validated in Python first — it is the same split-step
structure `TDSE_Solver/` already uses, just with a 2-spinor and a 2×2 k-space
step. *Optionally*, once the Python version has locked the physics (dispersion
limits, Zitterbewegung frequency, non-decaying Klein transmission), the Klein-
tunneling / Zitterbewegung demos are a natural `05_tdse_gpu` addition — a
2-spinor mode reusing its domain-colored view, so real-time Klein tunneling can
sit next to real-time non-relativistic tunneling. That C++ demo is a
nice-to-have visual, not on the critical path, and tracked like the other C++
ports.

If Dirac–Fock–Slater (Phase 4) proves out, porting it into a
convention-cleaned `01_hartree_fock` as a `--relativistic` mode is the natural
follow-up — by then that project has decks + selftests to make the port safe.

## What this enables (applications)

- **Fine structure.** The `2p_{1/2}` / `2p_{3/2}` splitting in hydrogen
  (`≈ 0.365 cm⁻¹`, `~11 GHz`) that non-relativistic QM misses entirely —
  reproduced from first principles, scaling as `(Zα)⁴`.
- **Spin–orbit doublets in spectra.** Feed j-resolved Dirac–Fock–Slater valence
  levels to `HF_solver/spectra.py` and the Na D-line comes out as **two** lines
  (D1 589.6 nm, D2 589.0 nm, `~17 cm⁻¹` split) instead of one.
- **Relativistic orbital contraction / expansion.** `s` and `p_{1/2}` orbitals
  contract, `d` and `f` expand (indirect screening) — the mechanism behind the
  color of gold, the liquidity of mercury, the inert `6s²` pair in Tl/Pb/Bi, and
  the `HF_solver` "Z > 80 qualitative only" caveat.
- **Relativistic total-energy corrections.** For `Z ≈ 80–100` these reach tens of
  hartree — the systematic gap between `HF_solver`'s heavy-atom numbers and
  reality, now quantified.
- **Klein tunneling / Klein paradox.** A wavepacket transmitted with `O(1)`
  probability through a barrier taller than `E + mc²`, where the non-relativistic
  `TDSE_Solver/` predicts near-total reflection. Same setup as the tunneling
  notebook, opposite result — and the mechanism behind graphene's
  Klein-tunneling junctions.
- **Zitterbewegung.** The `2mc²/ħ` trembling motion from positive/negative-energy
  interference in a localized packet.

## Numerical approach

Units: atomic units with `c = 1/α ≈ 137.036` (Hartree atomic units), so
relativistic effects are controlled by `Zα = Z/c`.

**1D time-dependent Dirac** (`dirac_1d.py`):
- 2-spinor `ψ = (ψ_1, ψ_2)`, Hamiltonian `H = c α_x p_x + β m c² + V(x)` with
  Pauli matrices for `α_x`, `β` in 1+1 D.
- Split-step: the free part `c α_x p_x + β mc²` is a `2×2` Hermitian matrix in
  `k`-space with eigenvalues `±E_k = ±√(c²k² + m²c⁴)` — the kinetic half-step is
  an exact `2×2` unitary `exp(-i H_free(k) Δt)` per `k` mode (closed form via the
  `2×2` rotation formula, no `expm`). The potential half-step is a local `2×2`
  (diagonal for a scalar potential).
- Norm `∫(|ψ_1|² + |ψ_2|²) dx` conserved to machine precision (analytically
  unitary, same as the non-relativistic split-step).

**Radial Dirac–Coulomb** (`dirac_radial.py`):
- Quantum number `κ` (= `−(l+1)` for `j = l+½`, `+l` for `j = l−½`). Coupled
  first-order radial system for the large `G(r)` and small `F(r)` components:
  ```
  dG/dr = −(κ/r) G + (1/c)[ 2mc² + E − V ] F
  dF/dr = +(κ/r) F − (1/c)[ E − V ] G
  ```
- Discretize on `HF_solver/atomic_scf.py`'s log grid. The known hazard is
  **variational collapse / spurious states** from an unbalanced discretization
  of the first-order operator; the robust fix used here is a symmetric
  finite-difference scheme on a *staggered* grid for `G` and `F` (or, as a
  fallback, the well-tested Salvat–Fernández-Varea RADIAL-style predictor–
  corrector shooting with node counting). Derived inline in the module docstring.
- Solve as a generalized eigenproblem for `E` with shift-invert, mirroring the
  atomic non-relativistic solver.

**Dirac–Fock–Slater** (`dirac_fock_slater.py`):
- `HF_solver/scf.py`'s loop shape unchanged: build `V_eff` from the current
  density → diagonalize each `κ`-channel with the radial Dirac solver → fill
  occupations by relativistic subshell `(n, l, j)` → rebuild the spherically
  averaged density `ρ = Σ (N_{nκ}/4πr²)(G² + F²)` → mix → converge.
- Slater/Xα exchange (`potentials.slater_exchange_potential`) is a **local
  functional of `ρ`** — geometry- and relativity-indifferent, ports over
  unchanged. Same for PZ81 correlation.
- Total energy: `Σ N_{nκ} ε_{nκ} − E_H − E_x/3` as in the non-relativistic case;
  the eigenvalues `ε` now include the rest-mass-subtracted relativistic
  kinetic energy automatically.

**Not attempted**: the Breit interaction (magnetic + retardation electron–
electron corrections); QED (Lamb shift etc.); full `jj`-coupled multiplet
structure (configuration-averaged, same simplification as `HF_solver/`);
negative-energy-sea / pair-production dynamics in the time-dependent part
(single-particle interpretation only, note the Klein-paradox subtlety).

## Upstream changes to existing solvers

All additive; `HF_solver/`'s existing non-relativistic numbers are untouched.

1. **`HF_solver/shells.py`** — `ground_state_configuration_jj(Z)`: split each
   `(n, l>0)` shell into `j = l ± ½` subshells with capacities `2j+1`, filled in
   relativistic-Madelung order. The non-relativistic `ground_state_configuration`
   stays the default.
2. **`HF_solver/spectra.py`** — no code change needed: `dipole_transitions`
   already takes an arbitrary list of levels. Feeding it `j`-resolved
   Dirac–Fock–Slater levels produces the fine-structure doublets automatically.
   Worth a note in the module docstring and a new comparison cell.
3. **`HF_solver/Hartree_Fock.ipynb`** (or a new notebook) — a relativistic vs
   non-relativistic total-energy comparison vs `Z`, showing the correction
   growing as `(Zα)²`, and the `1s` `⟨r⟩` contraction ratio.
4. **`HF_solver/potentials.py`** — no change; `slater_exchange_potential` /
   `pz81_correlation` are imported (or lightly copied, per repo convention) by
   the Dirac–Fock–Slater module as-is.

## File layout

```
Quantum Mechanics/Dirac_Solver/
    Dirac_Solver_Plan.md
    dirac_1d.py             # Phase 1-2: 2-spinor split-step Fourier, scalar/vector potentials
    dirac_radial.py         # Phase 3: coupled G/F radial system on a log grid, per kappa
    dirac_fock_slater.py    # Phase 4-5: SCF loop over kappa-channels + relativistic density + total energy
    rel_shells.py           # relativistic subshell filling (or import HF_solver/shells.py's new fn)
    Validation.ipynb              # Phase 1, 3: dispersion, Zitterbewegung, Sommerfeld formula
    Klein_Tunneling.ipynb        # Phase 2
    Relativistic_Atoms.ipynb     # Phase 4-6: DFS energies, contraction, Na D-lines, gold
    media/
```

## Phases

Phases 1–2 are Python (`dirac_1d.py` + notebooks); an optional C++ visual demo
in `05_tdse_gpu` follows only after they pass. Phases 3–6 are Python throughout.

**Phase 1 — 1D time-dependent Dirac** (`dirac_1d.py`)
- Free 2-spinor split-step.
- Validate: group velocity of a wavepacket `→ c` as `⟨p⟩ → ∞` and `→ p/m` in the
  non-relativistic limit; norm conserved to ~1e-13; **Zitterbewegung** of a
  boosted, spatially localized Gaussian oscillates at angular frequency
  `≈ 2mc²/ħ` with the predicted amplitude `~ħ/(mc)`.

**Phase 2 — Klein tunneling** (`Klein_Tunneling.ipynb`)
- Gaussian incident on a potential step with `V > E + mc²`.
- Validate: transmission is `O(1)` and **does not decay** as the barrier grows
  (contrast the non-relativistic exponential suppression from `TDSE_Solver/`'s
  tunneling notebook, run side by side); the transmitted packet's negative-
  energy character is visible in the small/large component ratio. MP4 of both
  cases together.

**Phase 2b (optional) — C++ interactive Klein-tunneling demo** (`05_tdse_gpu`)
- Only after Phases 1–2 pass. A 2-spinor mode in `05_tdse_gpu` reusing its
  domain-colored view and `[[potential]]` step blocks; the payoff is a live
  side-by-side with the non-relativistic tunneling deck.
- Validate: `--selftest` reproduces the Python `dirac_1d.py` transmission
  coefficient for the supercritical step to fp32 tolerance.

**Phase 3 — Radial Dirac–Coulomb** (`dirac_radial.py`)
- Hydrogen-like ions, point nucleus, `V = −Z/r`.
- Validate: bound-state energies match the **exact Sommerfeld formula**
  `E_{nκ} = mc²[1 + (Zα / (n − |κ| + √(κ² − Z²α²)))²]^{-1/2} − mc²`
  across `Z = 1..100`, `n = 1..4` to ~1e-6 relative; the `2p_{1/2}`/`2p_{3/2}`
  fine-structure splitting matches the `(Zα)⁴` analytic result; **no spurious /
  variational-collapse states** in the spectrum (explicit node-count check).

**Phase 4 — Dirac–Fock–Slater SCF** (`dirac_fock_slater.py`, `Relativistic_Atoms.ipynb`)
- Closed-shell atoms: He, Ne, Ar, Kr, Xe, Rn.
- Validate: total energies vs literature Dirac–Slater / Dirac–Fock values (a few
  % as expected for Slater exchange, same gap character as the non-relativistic
  solver's Xα comparisons); the relativistic correction
  `E_DFS − E_HFS` grows with `Z` roughly as `Z⁴` per the leading `(Zα)²`
  per-electron scaling times `Z`; `1s` `⟨r⟩` contracts vs the non-relativistic
  value by the expected `√(1 − (Zα)²)`-ish factor.

**Phase 5 — Fine-structure spectra** (`Relativistic_Atoms.ipynb`)
- Dirac–Fock–Slater for Na and K; feed the `j`-resolved valence levels to
  `HF_solver/spectra.py`.
- Validate: the alkali D-line comes out as a **doublet** — Na `3p_{3/2}→3s`
  and `3p_{1/2}→3s` split by `≈ 17 cm⁻¹` (K `≈ 58 cm⁻¹`), in the right ratio
  even if the absolute wavelength keeps the non-relativistic solver's
  Koopmans-type systematic offset. Compare directly to the single non-
  relativistic line.

**Phase 6 — Heavy-atom showcase** (`Relativistic_Atoms.ipynb`)
- Au (or Hg): relativistic vs non-relativistic `6s` and `5d` orbital energies
  and radii.
- Validate/illustrate: the `6s` contraction and stabilization that underlies
  gold's color (the `5d → 6s` absorption edge shifted into the visible) and the
  inert-pair effect; qualitative agreement with textbook relativistic-chemistry
  numbers. Figures to `media/`.

## Progress

- [ ] Phase 1 — 1D Dirac split-step, dispersion + Zitterbewegung (Python)
- [ ] Phase 2 — Klein tunneling vs non-relativistic tunneling (Python)
- [ ] Phase 2b — optional C++ interactive Klein demo in `05_tdse_gpu` (gated on 1–2)
- [ ] Phase 3 — radial Dirac–Coulomb vs Sommerfeld formula (Python)
- [ ] Phase 4 — Dirac–Fock–Slater SCF, relativistic energies + contraction
- [ ] Phase 5 — alkali D-line fine-structure doublets
- [ ] Phase 6 — gold / mercury relativistic showcase

## Verification

1. Phase 1: relativistic dispersion limits; norm conservation; Zitterbewegung at
   `2mc²/ħ`.
2. Phase 2: non-decaying transmission for `V > E + mc²`, in direct contrast with
   the non-relativistic notebook.
3. Phase 3: eigenvalues match the exact Sommerfeld formula across `Z`; fine-
   structure splitting `∝ (Zα)⁴`; no spurious states.
4. Phase 4: DFS total energies within a few % of literature; relativistic
   correction and `1s` contraction scale correctly with `Z`.
5. Phase 5: alkali D-lines resolve into doublets with the correct splitting.
6. Phase 6: gold `6s` relativistic stabilization reproduced qualitatively.
