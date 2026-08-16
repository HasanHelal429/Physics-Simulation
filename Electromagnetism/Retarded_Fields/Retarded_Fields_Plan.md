# Electromagnetism / Retarded_Fields — Design Plan

## Context

`legacy/Dipoles.py` and `legacy/Public CEM(Not My Code).py` both lean on
the third-party `pycharge` library's retarded-field (Lienard-Wiechert)
solver to animate radiating point charges; the latter's own filename
flags it as not the user's code, and it carries no license/attribution
header. `legacy/s_dipoles.dat` is `Dipoles.py`'s pickled output.

This project builds and validates its own Lienard-Wiechert field solver
for a point charge on an arbitrarily specified path, replacing the
`pycharge` dependency (and its attribution problem) with something built
and checked here. House style: own `*_Plan.md`, phased build with a
validation gate per phase, media to `media/`, legacy scripts archived.

## Numerical approach

**`retarded_fields.py`**: positions/velocities/accelerations are 3
-vectors throughout (even for in-plane charge motion, the radiated
B-field points out of the plane, so 2D vectors aren't sufficient).
- `retarded_time(r_func, x, t, c)`: solves `c*(t-t_ret) = |x-r(t_ret)|`
  for `t_ret <= t` via an auto-expanding bracket + `scipy.optimize.brentq`
  (a subluminal path always has a bracketable root).
- `lw_fields(r_func, v_func, a_func, x, t, q, c, eps0)`: the standard
  Lienard-Wiechert **E** (near/velocity term + far/radiation term) and
  **B** (`n x E / c`) fields at an observation point/time.
- `lw_fields_grid(...)`: convenience wrapper evaluating over a meshgrid
  of observation points (for animation), looping since each point needs
  its own retarded-time solve.

## File layout

```
Electromagnetism/
    Retarded_Fields/
        Retarded_Fields_Plan.md
        retarded_fields.py
        Validation.ipynb          # Phase 1
        Oscillating_Charge.ipynb  # Phase 2
        Dipole_Radiation.ipynb    # Phase 3
        legacy/
        media/
```

## Phases

**Phase 1 — `retarded_fields.py` + `Validation.ipynb`.** — Done.
Static charge: exact Coulomb match (`1.3e-16`), `B=0` exactly.
Uniformly-moving charge (`beta=0.3`): matched the present-position closed
form to `2.7e-16` -- confirms the retarded-time root-find is correct, not
just the zero-velocity limit. Accelerating charge (non-relativistic
circular motion, `v/c=5e-4`): Poynting flux integrated over an 800-point
Fibonacci sphere at `4` wavelengths out matched the Larmor formula to
`6.6e-7`.

**Phase 2 — `Oscillating_Charge.ipynb`.** — Done.
Rebuild of `legacy/Public CEM(Not My Code).py`: charge on a circular path
at `v=0.5c` (mildly relativistic, for a visibly non-trivial pattern),
animated log-scale field-magnitude heatmap + direction quiver, using this
project's own validated solver instead of `pycharge` -- resolving the
missing-attribution issue by replacement rather than adding a license
note to borrowed code. `lw_fields_grid` at a 70x70 grid, 60 frames, ran
in well under a minute (~0.3s/frame).

**Phase 3 — `Dipole_Radiation.ipynb`.** — Done.
`pycharge`'s specific coupling-constant formula wasn't independently
derived (as flagged as acceptable in this plan); a per-charge Larmor
re-check was also judged redundant, since `retarded_fields.py` applies
the same universal LW formula regardless of the path function, already
validated three ways in Phase 1. What's actually new with two charges is
**superposition**: two identical, in-phase point charges oscillating
along y, separated by `d=0.24` wavelengths along x -- summing their
independently-computed fields and reading the far-field angular power
pattern off a large circle matched the standard two-source array-factor
prediction (single-source Larmor pattern x array factor) with `0.999`
correlation.

## Progress

- [x] Phase 1 — `retarded_fields.py`, `Validation.ipynb`
- [x] Phase 2 — `Oscillating_Charge.ipynb`
- [x] Phase 3 — `Dipole_Radiation.ipynb`

All phases complete.
