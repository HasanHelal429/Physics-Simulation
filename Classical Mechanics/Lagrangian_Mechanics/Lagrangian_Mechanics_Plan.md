# Classical Mechanics / Lagrangian_Mechanics — Design Plan

## Context

`Classical Mechanics/` had 6 loose notebooks (`Harmonic Oscillator`,
`Coupled Harmonic Oscillator`, `Double Pendulum`, `Triple Pendulum`,
`Double Spring Pendulum`, plus `Particle Dynamics/Spring Pendulum` — misfiled
there, it's a full Lagrangian-mechanics notebook, not a beginner
particle-dynamics script) that each hand-rolled the same recipe: build a
symbolic Lagrangian, differentiate for the Euler-Lagrange equations,
`sp.solve` for the accelerations, `lambdify`, integrate with `odeint`,
animate. `Harmonic Oscillator.ipynb` was the odd one out — it solved the
damped Newtonian ODE directly instead of going through a Lagrangian at all,
inconsistent with the other five. None of the notebooks validated their
result against anything independent.

This project factors the shared recipe into `lagrangian.py` (house style:
own `*_Plan.md`, plain procedural module, phased build with a validation
gate per phase, media to `media/`), fixing the `Harmonic Oscillator`
inconsistency and adding real validation: closed-form solutions where they
exist, energy conservation and self-consistent small-oscillation checks
where they don't. Legacy notebooks archived to `legacy/`.

## Numerical approach

**`lagrangian.py`**:
- `euler_lagrange(L, coords, t)`: differentiate a symbolic Lagrangian and
  solve for the generalized accelerations.
- `build_ode_system(sols, coords, coords_d, t, params)`: lambdify into an
  `odeint`-ready `dSdt`.
- `integrate(...)`: thin `odeint` wrapper.
- `energy_fn(T_expr, U_expr, coords, coords_d, params)`: lambdify `T+U` for
  conservation checks.
- `linearize(L, coords, t, equilibrium)` / `normal_mode_frequencies(M, K,
  params)`: small-oscillation mass/stiffness matrices and their normal-mode
  frequencies, derived from the same symbolic Lagrangian — no separately
  hand-derived formula needed per system, and it doubles as a
  self-consistency check (a good Lagrangian and its own linearization
  should agree with what the full nonlinear integration does at small
  amplitude).

## File layout

```
Classical Mechanics/
    Lagrangian_Mechanics/
        Lagrangian_Mechanics_Plan.md
        lagrangian.py
        Validation.ipynb              # Phase 1
        Double_Pendulum.ipynb         # Phase 2
        Triple_Pendulum.ipynb         # Phase 3
        Spring_Pendulum.ipynb         # Phase 4
        Double_Spring_Pendulum.ipynb  # Phase 4
        legacy/
        media/
```

## Phases

**Phase 1 — `lagrangian.py` + `Validation.ipynb`** — Done.
Single damped/undamped harmonic oscillator (rebuilt through the Lagrangian
engine instead of the legacy notebook's direct-Newtonian solve; damping via
the `(T-U)*exp(b*t)` Caldirola-Kanai trick) and the 2-mass/3-spring coupled
oscillator. Validated: undamped case matches `x(t)=A*cos(wt)` to `2.35e-7`;
damped case matches the closed-form underdamped solution to `8.1e-8`
(confirming the `exp(b*t)` trick genuinely reproduces linear damping, not
just visually); `linearize()`'s coupled-oscillator normal modes match an
independent hand-derived Hessian exactly and the FFT of the full
integration to `0.004` rad/s.

**Phase 2 — `Double_Pendulum.ipynb`** — Done.
Same `m1=2,m2=1,L1=2,L2=1` chaotic setup as the legacy notebook. Energy
conserved to `1.6e-6` relative drift over the full chaotic run;
small-amplitude FFT frequencies matched `linearize()`'s normal modes to
within `0.6%`.

**Phase 3 — `Triple_Pendulum.ipynb`** — Done.
Same `m1=2,m2=1,m3=1,L1=2,L2=1,L3=1` setup as legacy. Energy conserved to
`1.3e-5` relative drift; all 3 small-amplitude normal modes matched
`linearize()` within `0.9%`. Building this phase also caught a real
performance bug: `euler_lagrange`'s generic `sp.solve` didn't finish in
several minutes for 3 coordinates (fine for 1-2). Since the Euler-Lagrange
equations are always linear in the accelerations, it now solves via
`sp.linear_eq_to_matrix` + `LUsolve` instead — exact, ~0.2s. A follow-on
`sp.simplify()` on the result was tried and made it *slower*; removed
(`lambdify` doesn't need a simplified expression).

**Phase 4 — `Spring_Pendulum.ipynb` + `Double_Spring_Pendulum.ipynb`** — Done.
Single spring pendulum: energy conserved to `1.8e-7`; at the hanging
equilibrium `r1_eq=L1+m1*g/k1` the angular and radial motions decouple at
linear order (the classic elastic-pendulum result, basis of its famous 2:1
resonance), each checked by perturbing its own coordinate — both matched
`linearize()` within ~0.5%.

Double spring pendulum: the legacy notebook ran with `g=0` (just to watch
the springs bounce), but with no gravity the angles have no restoring
force at all — nothing breaks the rotational symmetry, so there's no
hanging equilibrium to linearize around. Ran with `g=9.81` instead, giving
a proper 4-mode equilibrium (`r1_eq, r2_eq` from a simple tension-balance
formula, confirmed against a numerical potential-minimization to machine
precision). Energy conserved to `8.9e-8`. Validating the 4 normal modes
needed a second methodology fix: matching "the loudest FFT peaks" to the
predicted frequencies is fragile once a coordinate couples weakly to a
given mode, since a nonlinear combination tone can then outrank the true
peak. Flipped the check around — for each of the 4 predicted frequencies,
confirm a resolvable peak exists nearby in *some* coordinate's spectrum —
all 4 matched within 0.8%.

## Progress

- [x] Phase 1 — `lagrangian.py`, `Validation.ipynb`
- [x] Phase 2 — `Double_Pendulum.ipynb`
- [x] Phase 3 — `Triple_Pendulum.ipynb`
- [x] Phase 4 — `Spring_Pendulum.ipynb`, `Double_Spring_Pendulum.ipynb`

All phases complete.
