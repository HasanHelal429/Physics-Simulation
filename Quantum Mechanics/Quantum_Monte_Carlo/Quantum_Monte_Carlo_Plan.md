# Quantum Monte Carlo (VMC → DMC) — Design Plan

## Context

The DFT trilogy (`HF_solver/`, `Diatomic_HF_solver/`, `Molecular_DFT/`) every
single time reports an error against literature Hartree–Fock and then can go no
further — there is **no exact reference in the repo** for the one quantity DFT
approximates: the correlation energy `E_corr = E_exact − E_HF`. Worse, the PZ81
correlation functional those solvers use is itself a fit to **Ceperley–Alder
diffusion Monte Carlo** data for the electron gas. QMC is both the missing
referee and the actual origin of the physics already in use.

This project builds two stochastic, grid-free, dimension-agnostic methods:

- **Variational Monte Carlo (VMC)** — evaluate `⟨H⟩` for a parametrized trial
  wavefunction `Ψ_T(R; c)` by Metropolis-sampling `|Ψ_T|²`, then optimize `c`.
  Gives a rigorous upper bound; accuracy limited by the trial form.
- **Diffusion Monte Carlo (DMC)** — project out the exact ground state by
  evolving the imaginary-time Schrödinger equation as a drift–diffusion–branching
  random walk on a walker population, importance-sampled by `Ψ_T`. The only
  approximation for these systems is the **fixed-node** constraint (nodal surface
  taken from `Ψ_T`); typically recovers 95–99% of the correlation energy.

Scope: **small all-electron systems** — H, He, Li, Be, H2, LiH — plus the
homogeneous electron gas. Exactly the systems where "exact" is achievable and
where the DFT solvers have quantified errors to check.

## What this enables (applications)

- **An accuracy ledger for the whole repo.** "LDA is 11% off for He" becomes a
  row in a table: hydrogenic / Xα / LDA / VMC / DMC / experiment, with the
  correlation energy each method captures made explicit. First unified benchmark
  across all the electronic-structure solvers.
- **Correlation energy itself**, the target of every density functional —
  computed, not approximated, for He/Be/H2.
- **The electron gas.** Reproduce a handful of the Ceperley–Alder points that
  PZ81 was fit to (`r_s = 1, 2, 5, 10`) — closes the loop on the repo's own
  correlation functional.
- **Pair correlation functions `g(r)` and the on-top density `g(0)`** — a direct
  picture of what "correlation" means: electrons avoiding each other *beyond*
  what exchange already enforces. Visual, and impossible to get from a
  mean-field solver.
- **Bond dissociation done right.** Stretched H2, where restricted HF (and LDA)
  fail qualitatively — DMC stays correct. Motivates, and cross-checks, the
  `Tensor_Networks/` project's strong-correlation benchmarks.

## Numerical approach

Everything is a walk in the `3N`-dimensional space of electron coordinates
`R = (r_1, …, r_N)`. No grid, no basis-set truncation for the many-body state —
a genuinely different paradigm from every other solver here.

**Trial wavefunction — Slater–Jastrow**:
```
Ψ_T(R) = D↑(R) · D↓(R) · exp[ Σ_{i<j} u(r_ij) + Σ_{i,I} χ(r_iI) ]
```
- Slater determinants `D` built from analytic hydrogenic orbitals
  (`HF_solver/hydrogenic.py`) or, optionally, the converged numeric HF orbitals
  as a better starting point.
- Jastrow `u(r)`: short-range form obeying the **electron–electron cusp**
  (`u'(0) = ½` antiparallel, `¼` parallel) plus a smooth decaying tail with a
  few variational parameters.
- `χ(r)`: adjusts the **electron–nucleus cusp** when the orbitals do not already
  supply it exactly.

**VMC**:
- Metropolis, electron-by-electron proposals with a drift term
  `v = ∇Ψ_T/Ψ_T` (importance-sampled Langevin proposal) for acceptance
  efficiency near the nodes.
- Local energy `E_L(R) = (Ĥ Ψ_T)(R) / Ψ_T(R)` — needs `∇²Ψ_T/Ψ_T`, assembled from
  the Slater-matrix inverse (Sherman–Morrison on single-electron moves) and the
  Jastrow derivatives.
- Optimization: variance minimization on a fixed set of correlated samples
  (robust, simple), optionally the linear method for the last refinement.
- Error bars via reblocking (the samples are autocorrelated).

**DMC**:
- Short-time importance-sampled Green's function: drift step
  `R → R + τ v(R) + η` (`η` Gaussian, variance `τ`), then branch on the weight
  `exp[-τ (E_L(R) + E_L(R') - 2E_T)/2]`.
- Fixed-node: reject any move that crosses a node of `Ψ_T` (sign change).
- Population control on `E_T`; extrapolate `τ → 0` and population `→ ∞`.
- Mixed estimator for the energy; pure estimators (or forward-walking) for
  `g(r)` and the density if needed.

**Not attempted**: released-node / transient-estimate DMC (fixed-node only);
backflow or multi-determinant trial functions beyond a single CSF; pseudopotentials
(all-electron, light systems only); anything requiring more than ~10 electrons.

## Upstream changes to existing solvers

Minimal — QMC is nearly standalone:

1. **`HF_solver/hydrogenic.py`** — a vectorized
   `orbital_value_grad_lap(n, l, m, xyz) -> (val, grad, lap)` returning the
   value, gradient, and Laplacian of a hydrogenic orbital at arbitrary 3D points
   (current helpers evaluate on a prebuilt grid and return values only). Needed
   for the Slater matrix, the drift velocity, and the kinetic local energy.
2. Optional: `HF_solver/scf.py` and `Diatomic_HF_solver/diatomic_driver.py`
   expose their converged orbitals as callables on arbitrary points (a thin
   spline wrapper) so DMC can optionally use HF orbitals instead of hydrogenic
   ones as the trial. Not required for the first build.

No existing numbers change.

## File layout

```
Quantum Mechanics/Quantum_Monte_Carlo/
    Quantum_Monte_Carlo_Plan.md
    wavefunction.py    # Phase 1-2: Slater matrix + Jastrow, Psi_T, ratio/grad/lap, local energy
    vmc.py             # Phase 1-2: Metropolis + drift sampler, reblocking error bars, parameter optimization
    dmc.py             # Phase 3-4: drift-diffusion-branching, fixed-node, population control, tau extrapolation
    systems.py         # atom/molecule definitions (nuclei, N_up/N_down, trial orbital sets); electron gas box
    estimators.py      # energy, pair-correlation g(r), density
    Validation.ipynb           # Phase 1: H atom zero-variance test
    Atoms_and_Molecules.ipynb  # Phase 2-4: He, Li, Be, H2, LiH
    Accuracy_Ledger.ipynb      # Phase 5: the cross-solver comparison table
    Electron_Gas.ipynb         # Phase 6: HEG vs Ceperley-Alder / PZ81
    media/
```

## Phases

**Phase 1 — VMC infrastructure + hydrogen** (`wavefunction.py`, `vmc.py`)
- Trial `1s` with a variational effective charge `Z_eff`; Metropolis + drift;
  local energy; reblocking.
- Validate: optimal `Z_eff → 1.000`, `E → −0.5000` Ha with the **variance of
  `E_L` going to zero** as the trial approaches exact (the zero-variance
  principle — the single sharpest test in all of QMC).

**Phase 2 — VMC for He** (`wavefunction.py`, `vmc.py`)
- Two `1s` orbitals × a cusp-correct Jastrow; optimize.
- Validate: `E ≈ −2.87` to `−2.90` Ha (VMC with a decent Jastrow recovers
  ~85–90% of the `−0.042` Ha correlation energy) — below `HF_solver`'s LDA
  `−2.834` Ha and approaching HF `−2.862` Ha; variance drops by ~10× vs the
  Jastrow-free determinant.

**Phase 3 — DMC infrastructure + H, He** (`dmc.py`)
- Drift–diffusion–branching, importance sampling, population control,
  `τ`-extrapolation.
- Validate: H → `−0.5000` Ha exactly (nodeless); He →
  `−2.9037 ± 0.0002` Ha after `τ → 0`, matching the exact non-relativistic value
  `−2.90372` Ha — He's ground state is nodeless so fixed-node DMC is *exact*
  here, the cleanest possible DMC validation.

**Phase 4 — DMC for Li, Be, H2, LiH** (`dmc.py`, `Atoms_and_Molecules.ipynb`)
- Now the Slater determinant has real nodes and fixed-node error appears.
- Validate: against experimental / high-level-CI total energies —
  Li `−7.478` Ha, Be `−14.667` Ha, H2 `−1.1745` Ha, LiH `−8.070` Ha — to
  within the fixed-node error (~1–5 mHa for these), i.e. **>99% of correlation
  recovered**. Report the fixed-node error explicitly.

**Phase 5 — The accuracy ledger** (`Accuracy_Ledger.ipynb`)
- One table: `E_total` for H, He, Li, Be, H2 across hydrogenic /
  `HF_solver`-Xα / `HF_solver`-LDA / `Diatomic_HF_solver`-LDA /
  `Molecular_DFT` / VMC / DMC / experiment, with `E_corr` captured per method.
- Deliverable: the repo's first unified, quantitative statement of what each
  electronic-structure method actually costs in accuracy. Figure to `media/`.

**Phase 6 — Homogeneous electron gas** (`Electron_Gas.ipynb`)
- `N = 14` (closed-shell) unpolarized electrons in a periodic cubic box, plane-
  wave Slater determinant × Jastrow, at `r_s = 1, 2, 5, 10`; DMC with
  twist-averaging skipped (single `Γ`-point, note the finite-size caveat).
- Validate: correlation energy per electron vs the Ceperley–Alder values that
  PZ81 parametrizes (`r_s = 1`: `ε_c ≈ −0.060` Ha; `r_s = 5`: `≈ −0.028` Ha) to
  ~10% at single-`k` finite size. Pair-correlation `g(r)` showing the
  exchange–correlation hole; on-top `g(0)` decreasing with `r_s`. Figures to
  `media/`.

## Progress

- [ ] Phase 1 — VMC + H, zero-variance test
- [ ] Phase 2 — VMC He with Jastrow
- [ ] Phase 3 — DMC + H, He (exact, nodeless)
- [ ] Phase 4 — DMC Li, Be, H2, LiH (fixed-node)
- [ ] Phase 5 — cross-solver accuracy ledger
- [ ] Phase 6 — electron gas vs Ceperley–Alder / PZ81

## Verification

1. Phase 1: `Z_eff → 1`, `E → −0.5`, `Var(E_L) → 0` for hydrogen.
2. Phase 2: He VMC energy below LDA, variance reduced ~10× by the Jastrow.
3. Phase 3: He DMC `= −2.9037` Ha (exact, nodeless) after `τ`-extrapolation.
4. Phase 4: Li/Be/H2/LiH DMC within a few mHa of reference; fixed-node error
   reported.
5. Phase 5: the ledger table is internally consistent (DMC ≥ exact ≥ nothing
   violates variational bounds where they apply).
6. Phase 6: HEG correlation energies within ~10% of Ceperley–Alder at single-`k`
   finite size; `g(r)` shows the correct hole.
