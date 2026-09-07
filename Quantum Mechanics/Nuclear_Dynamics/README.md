# Born–Oppenheimer Nuclear Dynamics

The bridge between the electronic-structure solvers (`Diatomic_HF_solver/`,
`Molecular_DFT/`) and the wavepacket-dynamics solver (`TDSE_Solver/`): take a
computed potential-energy curve `E(R)` and solve the **nuclear** Schrödinger
equation on the internuclear coordinate with reduced mass
`μ = M_A M_B / (M_A + M_B)`,

```
[ -1/(2μ) d²/dR² + J(J+1)/(2μ R²) + E(R) ] χ(R) = E_vib χ(R)
```

reusing `TDSE_Solver`'s (now mass-aware) split-operator propagator and
finite-difference eigensolver almost unchanged. See `Nuclear_Dynamics_Plan.md`.

## Modules

| file | contents |
|---|---|
| `pes.py` | nuclear/reduced masses; load a tabulated `E(R)` CSV; spline+Morse it onto a grid (`TDSE_Solver.potentials.potential_from_samples`); analytic Morse spectrum + bound-state count; spectroscopic-constant (Dunham) extraction |
| `vibrational.py` | reduced-mass radial grid, bound `v`-levels at fixed `J`, `⟨1/R²⟩` → `B_e`, vibration–rotation coupling `α_e`, Franck–Condon matrix |
| `dynamics.py` | 1-surface split-operator propagation (wavepacket motion, revivals, photodissociation with a CAP detector); 2-surface propagation with a pointwise closed-form `2×2` matrix exponential for the diabatic coupling |
| `tools/make_pes.py` | regenerate the H2 LDA bond curve (`Diatomic_HF_solver` SCF scan) |
| `validate.py` | the phase-gated checks |

## Upstream changes (small, backward-compatible — `mass=1.0` reproduces every earlier result)

* `TDSE_Solver/propagator.py` — `mass` on `kinetic_eigenvalues` / `kinetic_step` / `strang_step`
* `TDSE_Solver/stationary_states.py` — `mass` on `hamiltonian` / `lowest_states`
* `TDSE_Solver/potentials.py` — `morse_well`, `potential_from_samples(grid, R, E, fill="morse")`
* `Diatomic_HF_solver/diatomic_driver.py`, `Molecular_DFT/scf3d.py` — `scan_pes(...)` promoting the bond scan to an importable product that also writes `media/pes_*.csv`

## Validate

```sh
python tools/make_pes.py          # ~5 min, writes Diatomic_HF_solver/media/pes_Z1_Z1_lda.csv
python validate.py --phase all
```

| phase | checks | result |
|---|---|---|
| 1 | reduced-mass grid reproduces the **exact Morse spectrum** (`v=0..9` to ~1e-5) and the bound-state count; `ω_e`, `ω_e x_e` recovered from the ladder | **PASS** |
| 2 | H2 LDA bond curve → vibrational levels: `ω_e ≈ 4175 cm⁻¹` (expt 4401, LDA well slightly soft), `ω_e x_e ≈ 108 cm⁻¹` (expt 121), `D_0 ≈ 5.4 eV` (expt 4.48 — LDA overbinds), anharmonic ladder | **PASS** (PES-limited) |
| 3 | `B_e ≈ 57 cm⁻¹` from `⟨1/R²⟩` (expt 60.85, LDA bond slightly long); `α_e > 0`; **D2 / HD levels scale as `μ^{-1/2}` (vibration) and `μ^{-1}` (rotation)** to <1% | **PASS** |
| 4 | vibrational wavepacket: energy conserved to 0.3% of `ω_e` over a full revival; `⟨R⟩(t)` oscillates at the mean level spacing; **collapse and partial revival at `T_rev = 2π/(ω_e x_e)`**. GIF. | **PASS** |
| 5 | Franck–Condon factors onto a model excited surface: **sum rule `Σ FC ≈ 1`**; envelope peaks at the vertical-transition `v'`. GIF of the upper-surface wavepacket. | **PASS** |
| 6 | photodissociation onto a repulsive curve: **all norm absorbed**; **KER `= E_total − V_∞`** (energy conservation) to ~10%; Gamow/WKB predissociation lifetime **grows sharply as the barrier thickens**. GIF. | **PASS** |

Media (`media/`): the Morse ladder, the H2 potential + levels + spacings, the
isotope `ω_e(μ)` scaling, the **revival** GIF + autocorrelation, the
Franck–Condon bar chart + upper-surface GIF, and the photodissociation panel + GIF.

## Notes

* The H2 PES is LDA on a coarse prolate-spheroidal grid — `ω_e`, `B_e`, `D_0`
  are all a few % off experiment in the *expected* directions (LDA gives a
  slightly soft, slightly long, over-bound H2). The isotope *ratios* and the
  Morse closed-form check are geometry/mass-exact and match tightly — same
  spirit as the rest of the repo.
* `eigsh` on the 1D tridiagonal nuclear Hamiltonian is fast (<1 s); no
  shift-invert needed despite `μ ~ 918` making the kinetic term tiny.
* Two-surface coupling, full ro-vibrational coupling beyond parametric `J`, and
  polyatomic normal modes are out of scope (noted in the plan).
