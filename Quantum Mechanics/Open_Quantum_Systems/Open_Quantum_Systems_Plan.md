# Open Quantum Systems / Lindblad — Design Plan

## Context

Every quantum solver in the repo so far is **closed, pure-state, and unitary**:
`TDSE_Solver/` and `05_tdse_gpu` propagate a wavefunction, `Perturbation_and_Basis_Methods/`
evolves basis coefficients, the DFT solvers find pure ground states. Real systems
couple to an environment — and that coupling is where decoherence, dissipation,
thermalization, spontaneous emission, finite linewidths, and measurement all come
from.

The minimal framework is the **density matrix** `ρ` and the **Lindblad master
equation**:

```
dρ/dt = −i[H, ρ]  +  Σ_k ( L_k ρ L_k†  −  ½ { L_k† L_k , ρ } )
```

The first term is ordinary unitary evolution; each **jump operator** `L_k`
encodes one dissipative channel (photon emission, dephasing collision, bath
absorption). This is small-Hilbert-space physics — two-level atoms, a few Fock
states — so it is fast, dense-matrix, and analytically checkable almost
everywhere. Deliberately a different scale from the many-electron solvers: the
payoff is conceptual coverage, not raw compute.

## What this enables (applications)

- **Decoherence — the quantum-to-classical transition.** Take the `TDSE_Solver/`
  double-slit and add a "which-path" dephasing channel: the interference
  contrast decays continuously as the coupling grows, and in the strong-coupling
  limit the fringe pattern collapses to the classical two-bump sum. The
  canonical demonstration of *why we don't see superpositions of macroscopic
  objects*.
- **T1 / T2 relaxation.** A driven two-level system (a qubit, an NMR spin, an
  atom in a laser) shows Rabi oscillations that damp to a steady state. The
  distinction between **energy relaxation** (T1, population decay) and **pure
  dephasing** (T2 ≤ 2·T1, phase randomization without energy loss) is the
  foundation of NMR, ESR, and qubit coherence budgets.
- **Natural linewidths and line shapes.** An atom radiating into the vacuum
  produces a **Lorentzian** emission line of width `γ` (the natural linewidth) —
  `HF_solver/spectra.py` currently draws zero-width stick lines; this is where
  the width comes from physically.
- **Resonance fluorescence / the Mollow triplet.** A strongly driven two-level
  atom's emission spectrum splits into **three** peaks at `ω_L` and
  `ω_L ± Ω_Rabi` — a famous, exactly-solvable benchmark.
- **Cavity QED — Jaynes–Cummings with loss.** One atom + one photon mode + cavity
  decay + atomic decay → vacuum Rabi splitting, the strong-coupling / weak-
  coupling boundary. The model behind circuit QED and single-atom lasers.
- **Thermalization.** Couple to a thermal bath with rates in detailed balance and
  `ρ` relaxes to the Gibbs state `e^{−H/kT}/Z` — checkable against the analytic
  thermal state to machine precision.
- **Quantum trajectories / Monte Carlo wavefunction.** The stochastic unravelling
  of the master equation: individual runs show discrete **quantum jumps**
  (a photon detection here, another there); averaging many trajectories
  reproduces `ρ(t)`. Both a computational method (cheaper for large Hilbert
  spaces) and the theory of continuous measurement.

## Numerical approach

**Representation**: `ρ` as a dense `N×N` complex matrix in a truncated basis —
energy eigenstates for atoms, Fock states `|0⟩…|n_max⟩` for a cavity mode,
tensor products for composite systems. `N` is small (2, 3, ~20 for a cavity,
~40 for Jaynes–Cummings).

**Time evolution**:
- Vectorize `ρ → |ρ⟩⟩` (column-stack); the right-hand side becomes a linear
  `N²×N²` **Liouvillian superoperator** `L` built from Kronecker products
  (`−i(H⊗I − I⊗Hᵀ)` for the commutator, standard forms for the dissipator).
- Time-independent `H`: `ρ(t) = unvec( expm(L t) |ρ_0⟩⟩ )`, or
  `scipy.integrate.solve_ivp` for a trajectory.
- Time-dependent drive: RK4 / `solve_ivp` on `d|ρ⟩⟩/dt = L(t)|ρ⟩⟩`.

**Steady state**: the null vector of `L` (`scipy.sparse.linalg.eigs(L, k=1,
sigma=0)` or a constrained linear solve with `Tr ρ = 1`).

**Spectra**: the **quantum regression theorem** — two-time correlations
`⟨A(t+τ)B(t)⟩` evolve under the same `L` as `ρ`, so the emission spectrum is
`S(ω) = Re ∫₀^∞ ⟨σ⁺(τ)σ⁻(0)⟩_ss e^{iωτ} dτ`.

**Quantum trajectories**: propagate a pure state under the non-Hermitian
`H_eff = H − (i/2) Σ L_k† L_k`; at each step the norm decay gives the total jump
probability; draw a uniform random number to decide whether a jump occurs and
(if so) which channel; renormalize; repeat; average observables over many
trajectories.

**Physical checks baked in**: `ρ` stays Hermitian, `Tr ρ = 1` (to machine
precision — a drift means a bug), and all eigenvalues of `ρ` stay in `[0, 1]`
(complete positivity). These are asserted every run.

**Not attempted**: non-Markovian / memory-kernel master equations (Lindblad is
Markovian by construction); deriving `L_k` microscopically from a specific bath
spectral density (the rates are taken as given parameters); large many-body open
systems (that is the `Tensor_Networks/` project's territory, via open-system
MPS).

## Upstream changes to existing solvers

Essentially none — this project is standalone. Two small optional touch-points:

1. **`HF_solver/spectra.py`** — `plot_spectrum` / `plot_grotrian` gain an
   optional `linewidths` argument to render **Lorentzians** instead of sticks,
   with the widths supplied by a Lindblad calculation of each transition's `γ`.
   Cosmetic, opt-in, default unchanged.
2. **`TDSE_Solver/`** — the double-slit potential geometry
   (`potentials.double_slit_barrier`) and initial wavepacket are reused as the
   setup for the Phase-5 decoherence demo (data/config reuse, or a coarse grid
   Lindblad — see Phase 5). No change to `TDSE_Solver/` itself.

## File layout

```
Quantum Mechanics/Open_Quantum_Systems/
    Open_Quantum_Systems_Plan.md
    operators.py       # spin / ladder / Fock operators, tensor products, common Hamiltonians
    lindblad.py        # Liouvillian assembly, expm / solve_ivp propagation, steady state
    trajectories.py    # Phase 6: Monte Carlo wavefunction unravelling
    spectra.py         # Phase 6: quantum regression theorem -> emission spectrum
    Validation.ipynb            # Phases 1-2: closed-system + spontaneous emission
    Bloch_and_Thermal.ipynb     # Phases 3-4: optical Bloch equations, Gibbs relaxation
    Decoherence.ipynb           # Phase 5: double-slit fringe collapse
    CavityQED_and_Mollow.ipynb  # Phase 6: Jaynes-Cummings, Mollow triplet, trajectories
    media/
```

## Phases

**Phase 1 — Density-matrix infrastructure + closed-system check** (`operators.py`, `lindblad.py`)
- Build `ρ`, the commutator Liouvillian, propagation, observables.
- Validate: with **no** jump operators, a two-level system driven on resonance
  shows exact Rabi oscillations `P_e(t) = sin²(Ωt/2)`; `ρ` stays pure
  (`Tr ρ² = 1`), Hermitian, unit-trace to ~1e-12.

**Phase 2 — Spontaneous emission** (`lindblad.py`, `Validation.ipynb`)
- One channel `L = √γ σ⁻`.
- Validate: excited-state population decays exactly as `e^{−γt}`; the steady
  state is the ground state; coherences decay at `γ/2`; `Tr ρ = 1` throughout.

**Phase 3 — Optical Bloch equations** (`Bloch_and_Thermal.ipynb`)
- Driven + damped two-level: Rabi drive `Ω`, detuning `Δ`, decay `γ`, optional
  pure dephasing `γ_φ` via `L = √(γ_φ/2) σ_z`.
- Validate: `T1 = 1/γ`, `T2 = 1/(γ/2 + γ_φ)` extracted from the dynamics match
  the analytic relations; the steady-state excited population vs `Δ` is a
  power-broadened Lorentzian matching the closed-form optical-Bloch result;
  `T2 ≤ 2 T1` always.

**Phase 4 — Thermal bath / Gibbs relaxation** (`Bloch_and_Thermal.ipynb`)
- Two channels `L₋ = √(γ(n̄+1)) σ⁻`, `L₊ = √(γ n̄) σ⁺` with `n̄` the Bose
  occupation at temperature `T` (detailed balance).
- Validate: `ρ(∞)` equals the Gibbs state `diag(e^{−E_i/kT})/Z` to machine
  precision across a range of `T`; the approach rate is `γ(2n̄+1)`.

**Phase 5 — Decoherence of interference** (`Decoherence.ipynb`)
- The double-slit, reduced to a small model: either a two-path qubit
  (`|L⟩, |R⟩`) with a position-dephasing channel `L ∝ σ_z`, or a coarse
  1D grid Lindblad with `L ∝ x̂` (localization/collision decoherence, the
  Joos–Zeh form).
- Validate: fringe visibility `V = (I_max − I_min)/(I_max + I_min)` decays
  exponentially with the dephasing rate × interaction time; at strong coupling
  the pattern → the incoherent sum of the two single-slit distributions. MP4 of
  the pattern washing out as coupling increases — the headline figure.

**Phase 6 — Cavity QED, Mollow triplet, trajectories** (`CavityQED_and_Mollow.ipynb`)
- Jaynes–Cummings (atom ⊗ Fock, `n_max ≈ 20`) with cavity decay `κ` and atomic
  decay `γ`; and a strongly driven two-level atom for the Mollow spectrum.
- Validate: in the strong-coupling regime the cavity emission spectrum shows
  **vacuum Rabi splitting** at `±g`; the driven-atom resonance-fluorescence
  spectrum is the **Mollow triplet** with sidebands at exactly `±Ω_Rabi` and the
  textbook `1 : 3 : 1`-ish weight/width ratios (via the regression theorem).
- Quantum-trajectory unravelling of the spontaneous-emission case: individual
  trajectories show sharp jumps; the average of ~10³ trajectories reproduces the
  Phase-2 `e^{−γt}` master-equation curve to within Monte Carlo error.

## Progress

- [ ] Phase 1 — density-matrix infra, closed-system Rabi
- [ ] Phase 2 — spontaneous emission, exponential decay
- [ ] Phase 3 — optical Bloch equations, T1/T2
- [ ] Phase 4 — thermal bath, Gibbs steady state
- [ ] Phase 5 — double-slit decoherence
- [ ] Phase 6 — cavity QED, Mollow triplet, quantum trajectories

## Verification

1. Phase 1: exact Rabi oscillations; `ρ` pure, Hermitian, unit-trace.
2. Phase 2: `e^{−γt}` population decay; ground-state steady state.
3. Phase 3: extracted `T1`, `T2` match analytic; power-broadened Lorentzian
   steady state; `T2 ≤ 2 T1`.
4. Phase 4: `ρ(∞)` = Gibbs state to machine precision.
5. Phase 5: exponential visibility decay; incoherent-sum limit at strong
   coupling.
6. Phase 6: vacuum Rabi splitting at `±g`; Mollow sidebands at `±Ω_Rabi`;
   trajectory average reproduces the master equation.
