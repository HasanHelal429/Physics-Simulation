# Electromagnetism / Poisson_Solver — Design Plan

## Context

`legacy/Laplaces Equation for Potential.ipynb` solves Laplace's equation
only (no source/charge term) via Gauss-Seidel relaxation run for a fixed
1000 iterations with no convergence check, then integrates a test charge
through the resulting field with a hand-rolled leapfrog and an ad hoc
"swap positions" boundary hack that isn't real physics. `legacy/
numerical.py` (orphaned -- nothing imported it) has a clean, reusable
finite-difference/sparse-Laplacian builder that the notebook never used.

This project generalizes to real Poisson's equation (an actual charge
density source term), replaces iterative relaxation with a direct sparse
linear solve (exact up to floating point, no iteration-count guesswork),
and reuses `numerical.py`'s Kronecker-sum operator technique (credited)
instead of leaving it orphaned. House style: own `*_Plan.md`, phased
build with a validation gate per phase, media to `media/`, legacy
notebook archived.

## Numerical approach

**`poisson.py`**:
- `laplacian_operator(shape, dx, dy)`: sparse 2D 5-point Laplacian via
  the central-difference-stencil + Kronecker-sum technique from
  `legacy/numerical.py` (credited in the docstring).
- `solve_poisson(rho, dx, dy, fixed_mask, fixed_values, eps0=1.0)`:
  direct sparse linear solve (`scipy.sparse.linalg.spsolve`) of
  `laplacian(V) = -rho/eps0`, with `fixed_mask` cells replaced by trivial
  identity equations enforcing `V=fixed_values` there (conductors /
  Dirichlet boundary).
- `efield(V, dx, dy)`: `E = -grad(V)` via central differences.

Grid convention: `V[i,j]`, axis 0 (`i`) is `x`, axis 1 (`j`) is `y`;
`efield` returns `(Ex, Ey)` matching that axis order.

## File layout

```
Electromagnetism/
    Poisson_Solver/
        Poisson_Solver_Plan.md
        poisson.py
        Validation.ipynb                    # Phase 1
        Charged_Particle_Trajectory.ipynb   # Phase 2
        legacy/
        media/
```

## Phases

**Phase 1 — `poisson.py` + `Validation.ipynb`.** — Done.
Point charge vs. 2D log Green's function: fitted `V(r)=A*ln(r)+B` gave
`A=-0.1609` vs. the analytic `-q/(2*pi*eps0)=-0.1592` (1.1% error).
Manufactured-solution convergence (`V_exact=sin(pi*x/Lx)*sin(pi*y/Ly)`,
avoiding the point charge's singularity): error ratios `3.91, 3.95, 3.98`
on successive grid-halvings -- clean 2nd order. Parallel-plate capacitor:
measured mid-gap field `0.988` vs. the ideal `(V1-V2)/d=1.000` (1.2%,
consistent with finite-plate edge effects at this plate length).

**Phase 2 — `Charged_Particle_Trajectory.ipynb`.** — Done.
A circular charged conducting disk (fixed `V0=5`) in a grounded box, an
opposite-sign test charge integrated through the field via leapfrog +
`scipy.interpolate.RegularGridInterpolator` (for continuous-position
field evaluation) and real elastic-wall reflection (velocity flip, not
the legacy position-swap hack). Since this solver's field falls off as
`E~1/r` (from the 2D log potential) rather than `1/r^2`, orbits generically
precess (Bertrand's theorem: only `1/r^2` and `r` force laws give closed
orbits) -- reported as the correct physics of this force law, not
"fixed." Initial speed calibrated from the locally-measured field for a
bound, moderately eccentric orbit; result: energy conserved to `5.9e-4`
over 40000 steps, orbit radius stayed in `[1.88, 2.51]`, safely outside
the `0.3`-radius charged disk. (First attempt used an uncalibrated
initial speed and grazed the disk at `r=0.096` with `4.7%` energy drift
from the resulting close encounter -- same close-approach lesson as
`Orbital_Dynamics/N_Body_Gravity`, fixed by calibrating the initial
speed to the locally measured field instead of guessing.)

## Progress

- [x] Phase 1 — `poisson.py`, `Validation.ipynb`
- [x] Phase 2 — `Charged_Particle_Trajectory.ipynb`

All phases complete.
