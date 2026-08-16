# Electromagnetism / Wave_Equation — Design Plan

## Context

`legacy/Wave Equation.ipynb` and `legacy/Wave Equation Refraction.ipynb`
(both originally loose top-level notebooks) each hand-roll the same
leapfrog finite-difference solver for the 2D scalar wave equation
(`wave_equation_2d`, copy-pasted between the two, with `c` promoted from
scalar to array in the refraction one for a spatially-varying wave speed).
Neither validates against anything, and the timestep formula
(`dt = courant_number * dx**2 / c`) is a diffusion-equation-style
stability scaling, not the actual wave-equation CFL condition
(`dt <= min(dx,dy)/(c*sqrt(2))` in 2D) -- it happens to produce a stable,
over-conservative `dt` here, but it's the wrong formula to carry into a
validated module.

This project factors the shared recipe into `wave_eq.py` (house style: own
`*_Plan.md`, phased build with a validation gate per phase, media to
`media/`, legacy notebooks archived), fixing the CFL formula and adding
real validation: measured propagation speed and standing-wave mode
frequencies against closed-form predictions, and a quantitative
diffraction check for the refraction demo.

## Numerical approach

**`wave_eq.py`**:
- `cfl_dt(dx, dy, c_max, courant=0.5)`: the correct 2D CFL bound.
- `leapfrog_step(psi_prev, psi_curr, dt, dx, dy, c, mask=None, source=None)`:
  one explicit step (`psi_next = 2*psi_curr - psi_prev + c^2*dt^2*laplacian`),
  `c` scalar or array, Dirichlet boundaries forced inside an optional mask,
  optional additive driving source term.

## File layout

```
Electromagnetism/
    Wave_Equation/
        Wave_Equation_Plan.md
        wave_eq.py
        Validation.ipynb              # Phase 1
        Focusing_Cavity.ipynb         # Phase 2
        Refraction_Diffraction.ipynb  # Phase 3
        legacy/
        media/
```

## Phases

**Phase 1 — `wave_eq.py` + `Validation.ipynb`.** — Done. (Note:
the module is named `wave_eq.py`, not `wave.py` -- the latter collides
with Python's standard-library `wave` audio module and jupyter's import
resolution picked up the stdlib one instead of the local file.)
Plane-wave propagation: rms error vs. the analytic traveling wave shrank
`0.194 -> 0.048 -> 0.012` as `dx` halved (`4.03x`, `4.01x` ratios -- clean
2nd-order convergence). Standing-wave modes: all 9 lowest analytic box
-mode frequencies matched FFT-extracted peaks within 1.2%.

**Phase 2 — `Focusing_Cavity.ipynb`.** — Done.
Rebuild of `legacy/Wave Equation.ipynb`: elliptical cavity, pulse launched
at one focus. The ellipse reflection-property prediction (refocus at
`t=2a/c`) matched the observed peak time (`24.81` vs. predicted `24.00`,
3.4%), with the refocused amplitude `8.8x` larger than the ordinary
direct (unreflected) pass-through amplitude measured at the same point
earlier (`t=2c/c`) -- a real quantitative confirmation, not just a
plausible-looking animation.

**Phase 3 — `Refraction_Diffraction.ipynb`.** — Done.
The legacy notebook combined a lens, a parabolic reflector, and a slit
into one mask via an XOR that isn't obviously physically meaningful, and
validated none of it. Split into two clean, separate setups instead: a
biconvex lens (slower `c` inside) shown as a qualitative demo, and a
single-slit diffraction setup with a real quantitative check. First had
to fix a genuine bug in the initial attempt: the settle time before
recording only accounted for periods, not the actual *transit time* for
the wave to reach the screen (screen 9 units away at `c=1` needs `t=9`,
but 30 periods at `T=0.2` only covers `t=6` -- so the "pattern" recorded
was essentially nothing, before the wave had even arrived). Fixed by
sizing the settle time from the transit distance. Checked the far-field
condition explicitly (`L/(a^2/lambda) = 7.5`) rather than assuming
Fraunhofer applies; the resulting main-lobe intensity matched the
analytic `sinc^2` envelope with `0.992` correlation, `0.044` rms
difference (side lobes excluded from the quantitative claim -- they're
small and noisy relative to finite-domain boundary-reflection artifacts).

## Progress

- [x] Phase 1 — `wave_eq.py`, `Validation.ipynb`
- [x] Phase 2 — `Focusing_Cavity.ipynb`
- [x] Phase 3 — `Refraction_Diffraction.ipynb`

All phases complete.
