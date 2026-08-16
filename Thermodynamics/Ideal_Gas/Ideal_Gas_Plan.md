# Thermodynamics / Ideal_Gas

## Context

2D hard-sphere gas kinetics, originally three near-duplicate notebooks in
`Classical Mechanics/Ideal Gas/` that had each copy-pasted and locally
tweaked the same O(N^2) all-pairs collision code. Already consolidated
(2026-08) onto one shared module before this project was relocated here
from Classical Mechanics — this file documents the result rather than
proposing new phases; there's no further build planned unless new physics
is added later.

## What's here

**`ideal_gas.py`** — shared 2D hard-sphere kinetics: spatial-hash collision
detection (`find_close_pairs_xy`, replacing the old O(N^2) check), elastic
pairwise collisions (`compute_new_v`), configurable per-wall reflection
with optional pressure tracking (`motion`), and pluggable external
acceleration fields (`uniform_gravity`, `central_gravity`).

- **`Ideal_Gas.ipynb`** — plain reflecting box: two opposing beams
  thermalize into a Maxwell-Boltzmann speed distribution; a pressure-vs.
  -initial-speed sweep confirms pressure grows with temperature as
  expected.
- **`Ideal_Gas_Gravity.ipynb`** — same gas, side walls only (open
  top/bottom), uniform downward gravity — sediments instead of reaching a
  steady pressure.
- **`Ideal_Gas_Orbit.ipynb`** — open domain, `central_gravity` toward a
  point mass, tangential initial velocities — a toy orbiting/proto
  -planetary gas cloud.

Media (animations, histograms) in `media/`.

## Possible future work

Not currently planned, but natural extensions if revisited: measure the
actual Maxwell-Boltzmann fit parameters (temperature) against the
input beam speed analytically; a proper equation-of-state sweep (pressure
vs. density at fixed temperature) to check `PV=NkT` quantitatively, the
same way `Ising_Model`'s phase diagram was checked against Onsager.
