# Quantum Monte Carlo — VMC → DMC

The missing *exact* reference for the DFT trilogy (`HF_solver/`,
`Diatomic_HF_solver/`, `Molecular_DFT/`): two stochastic, grid-free,
dimension-agnostic methods that compute the one quantity DFT approximates —
the correlation energy `E_corr = E_exact − E_HF`.

* **VMC** — Metropolis-sample `|Ψ_T|²` for a Slater–Jastrow trial function and
  optimize its parameters (rigorous upper bound).
* **DMC** — project out the fixed-node ground state by a drift–diffusion–
  branching walk on a walker population (exact for a nodeless state).

See `Quantum_Monte_Carlo_Plan.md`.

## Modules

| file | contents |
|---|---|
| `systems.py` | H, He, Li, Be, H2, LiH — nuclei, spin counts, Slater orbital sets (atomic + LCAO); the periodic electron-gas box |
| `wavefunction.py` | `SlaterJastrow`, **batched over walkers** — Slater matrices from analytic hydrogenic orbitals, a cusp-correct Padé Jastrow, and `log_psi` / `grad_ln_psi` (drift) / `lap_over_psi` / `local_energy` |
| `vmc.py` | drift-guided Metropolis (with the Umrigar drift cap), Flyvbjerg–Petersen reblocking, correlated-sample variance-minimization of the Jastrow |
| `dmc.py` | UNR drift–diffusion–branching: Metropolis-accepted moves, fixed-node rejection, weight branching with the effective time step, stochastic-reconfiguration population control, `dτ → 0` extrapolation |
| `estimators.py` | reblocked energy, pair correlation `g(r)`, radial density, the cross-solver accuracy-ledger table |
| `electron_gas.py` | plane-wave Slater determinant (real `{1, cos, sin}` basis) + Ewald interaction + RPA-style Jastrow — a reduced single-k-point VMC estimate of `ε_c` |
| `validate.py` | the phase-gated checks |

## Upstream change

* `HF_solver/hydrogenic.py` — `orbital_value_grad_lap(n, l, m, xyz, Z)`: analytic
  value / gradient / Laplacian of a real hydrogenic orbital at arbitrary 3D
  points (1s, 2s, 2p analytic; FD fallback otherwise). No existing numbers change.

## Validate

```sh
python validate.py --phase all
python validate.py --phase 1 --quick
```

| phase | checks | result |
|---|---|---|
| 1 | VMC + hydrogen: minimum-variance `Z_eff = 1`, `E(Z_eff=1) = −0.5` Ha, and **`var(E_L) → 0` (≈1e-32) at the exact trial** — the zero-variance principle | **PASS** |
| 2 | VMC He with a cusp-correct Jastrow: `E ≈ −2.896` Ha (below LDA `−2.834`, approaching HF `−2.862`), and the **local-energy variance cut ~12×** by the Jastrow | **PASS** |
| 3 | DMC: **H → −0.5 Ha** from an imperfect trial; **He → −2.902 ± 0.002 Ha** after `dτ → 0` (nodeless, so fixed-node is exact — matches `−2.90372`) | **PASS** |
| 4 | DMC for Li, Be, H2, LiH (real nodes): total energies within the **fixed-node error (few mHa)** of reference, reported explicitly | **PASS** |
| 5 | the **accuracy ledger** — `E_total` for H/He/Li/Be/H2 across hydrogenic / HF / LDA / VMC / DMC / experiment, `E_corr` captured per method; internally consistent (DMC ≤ VMC, DMC ≈ exact). Written to `media/accuracy_ledger.txt`. | **PASS** |
| 6 | homogeneous electron gas (reduced, single-k VMC): the **exchange-correlation hole** in `g(r)` and its `r_s` trend; correlation-energy trend vs the Ceperley–Alder points PZ81 was fit to | **PASS** (finite-size caveat) |

Media (`media/`): the zero-variance panel, the He local-energy distribution
(determinant vs Slater–Jastrow), the He DMC time-step extrapolation, the
fixed-node-error bar chart, the correlation-energy-by-method figure, and the
electron-gas `ε_c(r_s)` + `g(r)` panel.

## Notes

* The trial functions use **effective exponents set to `Z`** for the core so the
  electron–nucleus cusp is exact; the Padé Jastrow supplies the
  electron–electron cusp (`u'(0) = ½` antiparallel, `¼` parallel) and its
  denominator + the e–n amplitude are the only variational parameters.
* DMC here is a compact reference implementation — a few-mHa noise floor at these
  walker counts, no released-node / backflow / multi-determinant / pseudopotential
  machinery (all noted out of scope).
* The electron gas is a single Γ-point, `N = 14`, VMC-only estimate — the trend
  in `r_s` and the shape of `g(r)` are robust; the absolute `ε_c` carries a real
  finite-size + trial-function error. A twist-averaged DMC HEG is the natural
  extension.
