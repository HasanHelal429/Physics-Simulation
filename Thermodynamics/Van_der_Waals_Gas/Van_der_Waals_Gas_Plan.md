# Thermodynamics / Van_der_Waals_Gas — Design Plan

## Context

`legacy/Van Der Walls Gas.ipynb` is a genuine molecular-dynamics attempt
-- Lennard-Jones pairwise forces (the microscopic interaction that
produces van der Waals-type non-ideal gas behavior) via the same
spatial-hash neighbor search `Ideal_Gas/ideal_gas.py` already uses,
GPU torch tensors, reflecting walls. It has real bugs though:

- The force calculation loops over close-pair indices one at a time in
  Python (`for idx in range(close_pairs.shape[0])`), which defeats the
  entire point of GPU tensors -- everything else in the step is
  vectorized, this one loop dominates the runtime.
- `motion_adaptive` never actually adapts `dt` -- `dts` is allocated and
  never filled in; the name promises something the code doesn't do.
- The integrator is an ad hoc half-explicit-Euler scheme, not a proper
  symplectic integrator (velocity-Verlet) -- energy isn't guaranteed to
  stay bounded over long runs.
- An unphysical drag force (`drag_coeff * v`) damps the system for
  stability, which isn't representative of an actual (energy
  -conserving) molecular gas.
- The close-range force formula switches to an ad hoc softened formula
  (`7*epsilon*(1 - dist^2/(1.123*sigma)^2)`) with no stated derivation.

This project rebuilds the physics properly: a fully vectorized
Lennard-Jones force, a real velocity-Verlet integrator, and the same
wall-momentum pressure-tracking convention `ideal_gas.py` already uses
-- so the van der Waals gas and the existing ideal (hard-sphere) gas are
directly, apples-to-apples comparable at the same N, box size, and
temperature. House style: own `*_Plan.md`, phased build with a
validation gate per phase, media to `media/`, legacy archived.

## Numerical approach

**`vdw_gas.py`**:
- `lj_force(r, ids_pairs, epsilon, sigma)`: Lennard-Jones pairwise force,
  vectorized over all pairs at once (`torch.combinations` for all N*(N-1)/2
  pairs -- LJ decays fast enough, and N is small enough here, that an
  explicit neighbor cutoff isn't needed for correctness or speed), summed
  per particle via `index_add_` -- no per-pair Python loop.
- `velocity_verlet_step(r, v, a, dt, force_fn, ...)`: proper symplectic
  integration (`r += v*dt + 0.5*a*dt^2`, recompute `a` at the new `r`,
  `v += 0.5*(a_old+a_new)*dt`) -- conserves energy to a bounded,
  non-drifting error, unlike the legacy half-Euler scheme.
- Wall reflection + momentum-transfer pressure tracking: same convention
  as `Ideal_Gas/ideal_gas.py`'s `_reflect_wall`/`track_pressure`, so `P`
  measured here is directly comparable to that project's numbers.
- `second_virial_coefficient(T, epsilon, sigma)`: the 2D Lennard-Jones
  second virial coefficient `B2(T) = -pi * integral_0^inf (exp(-U(r)/kT) - 1) * r dr`,
  computed numerically (`scipy.integrate.quad`) -- the first-principles
  prediction of how much the LJ gas's pressure should deviate from ideal
  at low density, used to validate the simulated equation of state
  without needing an ad hoc Lennard-Jones-to-van-der-Waals-constants
  mapping (that correspondence isn't an exact closed form; the virial
  coefficient is).

## File layout

```
Thermodynamics/
    Van_der_Waals_Gas/
        Van_der_Waals_Gas_Plan.md
        vdw_gas.py
        Validation.ipynb           # Phase 1
        Condensation.ipynb         # Phase 2
        Equation_of_State.ipynb    # Phase 3
        legacy/
        media/
```

## Phases

**Phase 1 — `vdw_gas.py` + `Validation.ipynb`.**
Validate the integrator, not just the force formula: run an isolated
(no walls, no thermostat) N-particle LJ gas and check total energy
(kinetic + potential) stays bounded -- no secular drift -- over a long
run, the standard MD correctness check for a symplectic integrator.
Separately confirm the coded force matches `-dU/dr` for the LJ potential
via a finite-difference/symbolic check.

**Phase 2 — `Condensation.ipynb`.**
A demo `Ideal_Gas` fundamentally can't produce: at low temperature and
moderate density, the LJ attraction should pull particles into a
condensed cluster/droplet instead of an ideal gas's uniform spread.
Quantify with a clustering metric (e.g. mean local density or a
pair-correlation-function peak) rising as the system cools, contrasted
against a high-temperature run that stays gas-like.

**Phase 3 — `Equation_of_State.ipynb`.**
Measure pressure via the same wall-momentum method as `Ideal_Gas`,
across a range of densities at fixed temperature (with a light velocity
-rescaling thermostat to hold T fixed while density varies). Fit the
low-density virial expansion `P/(n*k*T) = 1 + B2*n` to the measured
points and compare the fitted `B2` against `second_virial_coefficient`'s
first-principles prediction -- and contrast both against the ideal gas
law's `B2=0` (measured directly in `Ideal_Gas`, for the same N and box).

## Progress

- [x] Phase 1 — `vdw_gas.py`, `Validation.ipynb`. Coded force matched
  `-dU/dr` to `3.2e-10` relative error across the repulsive core, well,
  and attractive tail. Velocity-Verlet kept total energy bounded within
  `0.45%` of its initial value over 10,000 steps (N=100), including
  through a large KE/PE equilibration transient -- not secularly
  drifting, unlike the legacy prototype's ad hoc half-Euler scheme. Note:
  the LJ repulsive core is stiff -- `dt=1e-3` let energy drift >100%;
  `dt=5e-5` was needed for a bounded result.
- [x] Phase 2 — `Condensation.ipynb`. Equilibrated a gas at T=1.0, then
  branched into a cooled (T=0.03) and a held-hot (T=1.0) run from the
  identical state. Mean-neighbor clustering metric grew monotonically
  from 0.90 to 2.45 under cooling (2.2x the held-hot run's 1.09, which
  stayed flat) -- clustering `Ideal_Gas`'s attraction-free collisions
  cannot produce.
- [x] Phase 3 — `Equation_of_State.ipynb`. Measured Z=P/(nkT) via the
  same wall-momentum method as `Ideal_Gas` across 3 densities at T=0.5,
  4 replicas each. Z decreased monotonically (0.85 -> 0.78 -> 0.66) as
  density increased, unlike `Ideal_Gas`'s Z=1 always. Fitted second
  virial coefficient (`-3.9e-3`) matched the first-principles
  `second_virial_coefficient` prediction (`-5.4e-3`) in sign and order of
  magnitude (0.72x) -- a real quantitative check, not just a plausible
  -looking trend. Getting a clean signal required substantially longer
  steady-state averaging windows than an initial short-run attempt,
  which was noisy enough to scatter Z above and below 1 with no visible
  trend.

All phases complete.
