# Thermodynamics / Ising_Model — Design Plan

## Context

`Ising Model.ipynb` (now in `legacy/`) sat as a loose top-level notebook with
no `Thermodynamics/` folder yet — the same situation the Quantum Mechanics
notebooks were in before the QM reorg. Reading it end-to-end found it's
mostly *correct* physics but has real methodological gaps that undermine the
one thing it's trying to show — the 2D Ising phase transition:

- **Free (zero-padded) boundary conditions**, not periodic — `conv2d(...,
  padding='same')` pads with zeros. This introduces surface effects that
  round/shift the apparent transition relative to the thermodynamic limit.
- **No equilibration check.** Every `(T, B)` point runs a fixed 1000 sweeps
  and averages the last 400, regardless of temperature. Near `T_c`,
  relaxation time diverges (critical slowing down), so points near the
  transition are likely under-equilibrated while points far from it are fine.
- **No real error bars.** Variance computed directly from 400 highly
  autocorrelated Metropolis samples, no correction for autocorrelation time.
- **No validation against ground truth.** 2D Ising has an exact solution:
  `T_c = 2/ln(1+sqrt(2)) ~= 2.269` (`J/k`, `B=0`) and a closed-form
  magnetization curve below `T_c` (Onsager), plus exact enumeration is
  tractable on small lattices for exact `<E>`, `<M>`, `C`, `chi` at any `T`.
- Minor: the legacy "checkerboard" update is actually a 4-way `i::2, j::2`
  split (valid but ~2x more iterations than needed), and its final
  `fig.savefig(...)` uses a leftover macOS path.

This project rebuilds the Ising model in the house style (`*_Plan.md`, plain
procedural modules, no classes, phased build with a validation gate per
phase, plots to `media/`), fixing the above and adding a cluster algorithm +
finite-size scaling as follow-on phases.

## Numerical approach

**`ising.py`** — core lattice engine, GPU (torch) throughout, periodic
boundary conditions via manual circular padding, standard Hamiltonian
`H = -J*sum_<ij> S_i*S_j - B*sum_i S_i` (J=1) used consistently for both the
Metropolis local-field decision and total-energy measurement (the legacy
per-site-energy-sum convention double-counted bonds and had the field term's
sign flipped relative to this standard form — fixed here so results compare
directly to textbook `T_c`/exponents with no hidden factors). Metropolis
step uses a true 2-color `(i+j)%2` checkerboard (two conflict-free
sublattices), batched over `(T, B)` like the original.

**Discovered during Phase 1/2 validation:** any fully synchronous
parallel-flip update (2-color checkerboard, or the legacy notebook's 4-way
random-offset split) can get permanently trapped in symmetric
configurations where every site's local field is exactly zero — `dE=0`
everywhere means the exact Metropolis rule force-accepts every flip with
probability 1, independent of temperature, so no randomness is ever
invoked and the chain cycles in a temperature-independent absorbing orbit
forever. Confirmed directly (reversing checkerboard order and switching to
the 4-way scheme both stayed trapped). `ising.run_sweeps` adds a targeted
safety net: detect the signature (energy bit-identical across several
sweeps *while the configuration keeps changing*, unlike an ordinary frozen
cold state where the configuration itself also stops changing) and apply
one random single-spin kick to break the symmetry. All later phases should
use `run_sweeps`, not a raw `sweep()` loop. Full writeup in
`Validation.ipynb`.

**`analytic.py`** — exact reference solutions: brute-force exact enumeration
of `<E>`, `<M>`, `C`, `chi` for small periodic lattices (practical up to
`~4x4`, `2^16` states), plus closed-form Onsager `T_c` and exact `B=0`
magnetization curve for the thermodynamic limit.

**`sampling.py`** (Phase 2) — integrated-autocorrelation-time estimator,
running-mean burn-in detection, autocorrelation-corrected error bars.

**`cluster.py`** (Phase 4) — single-cluster Wolff algorithm.

## File layout

```
Thermodynamics/
    Ising_Model/
        Ising_Model_Plan.md
        ising.py                        # Phase 1
        analytic.py                     # Phase 1
        sampling.py                     # Phase 2
        cluster.py                      # Phase 4
        Validation.ipynb                # Phase 1-2
        Phase_Diagram.ipynb             # Phase 3
        Critical_Slowing_Down.ipynb     # Phase 4
        Critical_Exponents.ipynb        # Phase 5
        legacy/
            Ising Model.ipynb
            ising_model.png
        media/
```

## Phases

**Phase 1 — Core engine & exact validation** (`ising.py`, `analytic.py`)
Validate: Metropolis-sampled `<E>`, `<M>`, `C`, `chi` on a `4x4` periodic
lattice match `exact_enumeration` within statistical error at high-T,
near-`T_c`, and low-T.

**Phase 2 — Equilibration & error bars** (`sampling.py`)
Validate: autocorrelation-corrected error bars bracket the exact-enumeration
mean at the expected confidence level; burn-in detection tracks the
running-mean asymptote.

**Phase 3 — Large-lattice phase diagram** (`Phase_Diagram.ipynb`)
Validate: `B=0` transition location and magnetization curve match Onsager.

**Phase 4 — Wolff cluster algorithm** (`cluster.py`,
`Critical_Slowing_Down.ipynb`)
Validate: Wolff reproduces Phase 3 physics with dramatically shorter
autocorrelation time near `T_c` than Metropolis.

**Phase 5 — Finite-size scaling / critical exponents**
(`Critical_Exponents.ipynb`)
Validate: exponents match `beta=1/8`, `gamma=7/4`, `nu=1` within
statistical error.

**Result: partial validation, reported honestly rather than forced.**
`gamma/nu` matches the exact value (within ~2-5%) and every qualitative
finite-size-scaling trend is correct (`chi_max(L)` grows, `M(T_c,L)` shrinks,
the pseudo-critical temperature shifts toward `T_c` from above as `L`
grows). `beta/nu` and `nu` (extracted via the susceptibility peak's shift)
do not match well (`nu~0.7-0.76` vs. exact `1`) — checked across two
independent Monte Carlo runs (`L=8-32` and `L=8-48`), a validated
peak-finder (exact on a synthetic parabola), and a second independent
method (data collapse), all consistently non-convergent to `nu=1`. Likely
cause (standard in the FSS literature, not a bug here): the
susceptibility-peak estimator of the pseudo-critical temperature carries
large sub-leading correction-to-scaling terms and converges slowly; a
Binder-cumulant crossing or much larger `L` would be needed to pin `nu`
down tightly. See `Critical_Exponents.ipynb`'s discussion section for the
full writeup.

Not scoped here: an XY-model sibling project, and the unrelated `ideal gas`
trilogy consolidation noted in the top-level cleanup memory.

## Progress

- [x] Move `Ising Model.ipynb` + `ising_model.png` into `legacy/`
- [x] Phase 1 — `ising.py`, `analytic.py` + exact-enumeration validation
- [x] Phase 2 — `sampling.py` + autocorrelation/error-bar validation
- [x] Phase 3 — `Phase_Diagram.ipynb` vs. Onsager
- [x] Phase 4 — `cluster.py` + `Critical_Slowing_Down.ipynb`
- [x] Phase 5 — `Critical_Exponents.ipynb` (partial: gamma/nu + qualitative
      trends validated; beta/nu and nu not resolved at this system-size
      range — see writeup above)
