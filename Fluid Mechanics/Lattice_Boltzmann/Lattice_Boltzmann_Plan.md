# Fluid Mechanics / Lattice_Boltzmann — Design Plan

## Context

`Fluid Mechanics/` has two solver projects so far: `Simple_Fluid/` (Stable
Fluids — nice-looking, not quantitatively trustworthy) and
`MAC_Grid_Solver/` (a real staggered-grid Navier–Stokes solver, validated
against the Ghia et al. lid-driven-cavity benchmark; its cylinder
vortex-shedding phase is paused on an unresolved coarse-vs-fine-grid
discrepancy, documented in its own plan/`phase7_investigation/`).

When `MAC_Grid_Solver` was scoped, four solver families were on the table
(Lattice Boltzmann, SPH, a higher-order Eulerian MAC-grid, FLIP/PIC); LBM
lost out to the MAC-grid route at the time. This project builds it as its
own thing: a genuinely different numerical method (mesoscopic kinetic
simulation on a lattice, not a discretized Navier–Stokes PDE) that happens
to recover the same continuum fluid equations — a good contrast to
`MAC_Grid_Solver`, and a same-benchmark cross-check opportunity (both
projects can be validated against Ghia et al.'s cavity data, and both
attempt the cylinder Strouhal-number benchmark that stumped `MAC_Grid_Solver`).

House style, matching `MAC_Grid_Solver`: own project folder,
`*_Plan.md` design doc, phased build with an internal validation gate per
phase, physics/numerics derivations written out inline, plain
NumPy/vectorized array ops (no need for `scipy.sparse` here — LBM's
collision+streaming update is explicit and local, unlike a pressure
Poisson solve), animations to `media/` via matplotlib's `ffmpeg` writer.

## Numerical approach

**D2Q9 lattice BGK method** (Bhatnagar-Gross-Krook), the standard
first-implementation LBM scheme: 9 discrete velocity directions per node
(rest + 4 axis + 4 diagonal), weights `w = [4/9, 1/9x4, 1/36x4]`, lattice
speed of sound `c_s^2 = 1/3` in lattice units (`dx = dt = 1`).

- **Equilibrium distribution**: `f_i^eq = w_i*rho*(1 + 3(e_i.u) +
  4.5(e_i.u)^2 - 1.5|u|^2)`, the 2nd-order-in-Mach expansion of the
  Maxwell-Boltzmann distribution — the small-Mach assumption it relies on
  is enforced by keeping `U_lattice` small (~0.05-0.1) and deriving
  physical Reynolds number from lattice quantities via
  `Re = U_lattice*L_lattice/nu_lattice`.
- **Collision (BGK)**: `f_i <- f_i - (f_i - f_i^eq)/tau`. Macroscopic
  moments: `rho = sum(f_i)`, `rho*u = sum(f_i * e_i)`.
- **Streaming**: `f_i(x + e_i, t+1) = f_i(x, t)` — a pure array shift per
  direction (`np.roll`), no interpolation, exact on the lattice.
- **Chapman-Enskog relation**: `nu = c_s^2 (tau - 1/2) = (tau - 1/2)/3` in
  lattice units — validated numerically in Phase 1 (decaying shear-wave
  test), not just asserted.
- **Known BGK limitations, stated up front**: single relaxation time ties
  numerical stability to tau close to 0.5 (i.e. low viscosity / high Re)
  — an MRT (multi-relaxation-time) collision operator is the standard fix
  but is out of scope; staying at Re <= ~1000 with tau comfortably above
  0.5 (target tau ~0.6-0.9) avoids this without needing MRT.

**Boundary conditions** (`boundary.py`):
- **Bounce-back (no-slip)**, both for domain walls and for an arbitrary
  solid mask (obstacle) — after streaming, reverse the populations that
  streamed into a solid node back along their opposite lattice direction.
  Reuses the same "boolean fluid mask" idea as `Simple_Fluid/fluid_sim.py`'s
  `circle_mask` for the cylinder case.
- **Zou-He velocity BC** for the moving lid (prescribes exact macroscopic
  velocity at a boundary by solving the unknown-population equations
  directly, rather than approximating it) — needed because plain
  bounce-back can't represent a *moving* wall.
- **Body-force driving** (Guo forcing scheme) for periodic channel flow
  (Poiseuille), so Phase 2 doesn't need a Zou-He pressure/inflow BC to get
  a first quantitative flow validation.

**Units / non-dimensionalization** (`solver.py`): helper to go from a
desired `(Re, L_lattice)` to `(U_lattice, tau)` — mirrors `MAC_Grid_Solver`'s
`Re = UL/nu` convention so results are describable/comparable the same way.

## File layout

```
Fluid Mechanics/Lattice_Boltzmann/
    Lattice_Boltzmann_Plan.md
    lattice.py                     # Phase 1: D2Q9 constants, equilibrium, moments, collide+stream
    boundary.py                    # Phase 2: bounce-back (wall + obstacle), Zou-He velocity BC, Guo forcing
    solver.py                      # Phase 2: step() orchestration, Re<->(U,tau) unit conversion
    lattice_numba.py               # Phase 3: numba-jitted fused step, cross-validated vs. the numpy path
    Validation.ipynb               # Phases 1-2
    Lid_Driven_Cavity.ipynb        # Phase 3
    Cylinder_Vortex_Shedding.ipynb # Phase 4
    cavity_results.npz             # Phase 3: precomputed velocity fields the notebook loads
    cylinder_results.npz                  # Phase 4: primary run (blockage 0.125)
    cylinder_results_lowblockage.npz      # Phase 4: confinement-check run (blockage 0.0625)
    media/
```

## Phase 1 — Core lattice (`lattice.py`) — Done

- D2Q9 velocity/weight constants; `equilibrium(rho, ux, uy)`; `moments(f)`
  -> `(rho, ux, uy)`; `collide(f, tau)`; `stream(f)` via `np.roll` per
  direction.
- **Validate**: mass conservation (`sum(rho)` constant to machine
  precision) and momentum conservation across pure collide+stream on a
  periodic domain with no forcing; decaying shear-wave test (sinusoidal
  `u_x(y)`, periodic domain, no walls) — fit the amplitude's exponential
  decay rate and compare to the analytic `nu = (tau-1/2)/3` prediction.
  **Actual**: mass drift `2.2e-10`, momentum drift `<3e-13` after 2000
  steps from a random non-equilibrium field; shear-wave `nu_fit=0.10005`
  vs `nu_analytic=0.1` (`tau=0.8`), `5.05e-4` relative error.

## Phase 2 — Boundary conditions & solver (`boundary.py`, `solver.py`) — Done

- Bounce-back for stationary walls and an arbitrary obstacle mask;
  Zou-He velocity BC for a moving wall; Guo body-force forcing term.
- `step(f, tau, mask=None, forcing=None, moving_walls=None)`: collide ->
  force -> stream -> apply BCs.
- **Validate**: periodic channel (bounce-back walls, uniform body force)
  reaches the analytic parabolic Poiseuille profile
  `u(y) = (F/(2*nu))*y*(H-y)` at steady state; an isolated Zou-He check
  confirms the prescribed lid velocity is recovered exactly at the
  boundary nodes (not just approximately, as plain bounce-back would give).
  **Actual**: bounce-back's no-slip plane sits *halfway* between the
  solid node and its fluid neighbor (verified by tracing a population's
  round trip), so the analytic profile uses effective wall locations at
  `y=0.5`/`y=ny-1.5`, not the raw grid edges -- with that, max relative
  error `4.25e-4`. Zou-He recovers the prescribed lid velocity to
  `2.78e-17` (machine precision) and `u_y=1.39e-17` (exact no-penetration).

## Phase 3 — Validation: lid-driven cavity (`Lid_Driven_Cavity.ipynb`) — Done

- Same setup as `MAC_Grid_Solver`'s Phase 6: unit square, `Re = 100, 400,
  1000` via `Re = U_lattice*L_lattice/nu_lattice`, no-slip on 3 walls,
  Zou-He moving lid on the top.
- Compare steady-state centerline `u(y)` at `x=0.5` and `v(x)` at `y=0.5`
  against Ghia, Ghia & Shin (1982) — the same reference `MAC_Grid_Solver`
  used, so the two projects' results can be compared directly against
  each other as well as against the literature table.
- Visualize streamlines/vorticity; expect the same qualitative
  Re-dependent secondary-corner-vortex behavior `MAC_Grid_Solver` found.

**What actually happened, in order:**

1. **Resolution/stability**: `L=100` and `L=200` (at `U_lid=0.1`,
   `Re=1000`) both diverged to `inf`/`nan` within a few hundred steps —
   plain BGK's stability requires `tau` comfortably above 0.5, and
   `nu = U_lid*L/Re` only rises with `L` at fixed `Re`/`U_lid`. Bisected
   up to `L=255` (odd, for an exact centerline grid point), confirmed
   stable, giving `tau=0.577` at Re=1000 (0.691 at Re=400, 1.265 at
   Re=100).
2. **Convergence rate**: a standalone diagnostic tracked the RMS error
   against Ghia et al. directly (not just kinetic energy, which turned
   out to drift for a very long time as small corner vortices slowly
   develop, long after the dominant centerline profile has settled — a
   strict KE-plateau stopping criterion, which worked fine for the
   Poiseuille/Zou-He checks, proved impractical here). The error fell
   from 0.25 at step 2000 through the 0.08 pass threshold by step
   ~31000-32000, continuing to drop — so `max_steps=40000` (fixed, not
   tolerance-gated) was adopted for all three Re.
3. **Performance**: at `L=255`, 40000 steps in pure-numpy `solver.step`
   is ~1040s of actual compute per Re (~26 ms/step) — and this specific
   environment throttles long-running background processes far below
   that, making a live 3-Re run impractically slow (a `jupyter
   nbconvert`-driven run of this was abandoned after its kernel process
   showed roughly a 10x lower CPU-time/wall-time ratio than the
   equivalent plain `python` process, for reasons unrelated to the
   physics). **Fix**: `lattice_numba.py`'s `step_numba` fuses collision +
   bounce-back + streaming + the Zou-He lid into one `@njit` kernel,
   cross-validated bit-for-bit (`max abs diff = 5.0e-16` after 300 steps)
   against `solver.step` on a small test case before being trusted for
   the real run — ~6x faster (4.3 ms/step vs. 26 ms/step). Ran as a
   standalone plain-`python` script (avoiding the jupyter-kernel
   slowdown found in the same investigation), saving the three converged
   velocity fields to `cavity_results.npz`; `Lid_Driven_Cavity.ipynb`
   loads that file rather than recomputing, since jupyter's own kernel
   is the slow path here, not the numerics.

**Actual results** (RMS error vs. Ghia et al., threshold 0.08): Re=100
`u=0.0098, v=0.0195`; Re=400 `u=0.0226, v=0.0354`; Re=1000 `u=0.0440,
v=0.0385` — all comfortably under threshold, and, unsurprisingly, error
grows with Re (further from the well-resolved low-Re primary vortex).

## Phase 4 — Validation: cylinder vortex shedding (`Cylinder_Vortex_Shedding.ipynb`) — Done

- Channel with a cylindrical bounce-back obstacle (reusing the
  boolean-mask approach), blockage ratio and upstream/downstream fetch
  chosen the same way `MAC_Grid_Solver`'s (paused) Phase 7 reasoned about
  them; cylinder spanning enough lattice nodes across its diameter to
  keep bounce-back's staircase geometry error small.
- Headline case Re ~= 100 (2D laminar periodic-shedding window). Probe
  downstream velocity/lift, run to statistically steady periodic
  shedding, extract frequency via FFT, compute `St = fD/U_inf`.
- Compare against Roshko's correlation (`St ~= 0.198(1-19.7/Re)`, giving
  `St ~= 0.159` at Re=100) and literature point values (`St ~= 0.164-0.167`
  at Re=100).

**New boundary conditions needed** (an open channel, unlike the closed
lid-driven cavity): `boundary.zou_he_inlet_west` (Zou & He velocity
inlet, prescribing the wall-*normal* velocity component — genuinely
harder than the lid's tangential-velocity case, needing Zou & He's
extra "bounce-back of the non-equilibrium normal population" closure;
derived from scratch and verified by substitution into the mass/momentum
identities before use) and `boundary.outflow_zero_gradient` (simple
Neumann outlet). Composed in `solver.step_channel`; `lattice_numba.
step_channel_numba` is the fused/jitted equivalent, cross-validated
bit-for-bit against it (`max abs diff = 5.0e-16` after 500 steps) before
being trusted for the real runs, same pattern as Phase 3.

**Geometry actually used**: `D=25`, `U_in=0.1` (`Re=100` -> `tau=0.575`,
comfortably stable), 6D upstream / 20D downstream fetch, cylinder offset
0.05D off-centerline to seed the shedding asymmetry. Primary run at
blockage `D/H=0.125` (upper end of the plan's recommended range),
45000 steps (~200s numba compute).

**Primary result**: the probe signal saturates into a clean single
-frequency limit cycle by ~15000 steps (RMS amplitude flat at 0.0515
for the rest of the run) — zero-crossing and FFT period estimates agree
to 0.2% (`St=0.2028` vs. `0.2024`), a strong indicator of genuine,
well-resolved periodic shedding. This `St~=0.202` sits well above both
Roshko (`0.159`) and the literature range (`0.164-0.167`).

**Confinement check, not just asserted**: rather than attribute that gap
to blockage confinement without evidence, a second run at half the
blockage (`D/H=0.0625`, everything else identical) was run to test it
directly. First attempt (45000 steps) had a non-saturated amplitude
envelope, so it was extended to 90000 steps — which *still* shows the
envelope fluctuating (RMS bouncing between ~0.02 and ~0.06, no plateau),
unlike the primary run. This is reported as a genuine finding, not
smoothed over: a beating between the primary shedding mode and a weaker
secondary mode is a known phenomenon in less-confined bluff-body wakes,
plausibly what's happening here, though not conclusively diagnosed.
Because of that modulation, the zero-crossing method is unreliable for
this run (it registers spurious crossings from the envelope, giving
wildly inconsistent `St` between different analysis windows), so only
the FFT estimate is trusted — and it *is* stable regardless of which
window is analyzed (`0.1721` over the last 50%, `0.1755` over the last
30%). That FFT-based `St=0.172` sits much closer to Roshko/literature
than the primary run's `0.202`, confirming the confinement-elevates-St
mechanism the plan flagged up front, even though the lower-blockage
wake turned out to have more complex (non-stationary-amplitude)
dynamics than expected.
- Vorticity snapshot showing the von Karman street saved to `media/`
  (a full animation was judged unnecessary given the quantitative
  Strouhal-number check already demonstrates genuine periodic shedding).

## Progress

- [x] Phase 1 — `lattice.py`, `Validation.ipynb`
- [x] Phase 2 — `boundary.py`, `solver.py`
- [x] Phase 3 — `Lid_Driven_Cavity.ipynb`
- [x] Phase 4 — `Cylinder_Vortex_Shedding.ipynb`

All phases complete.

## Verification

1. Phase 1: mass/momentum conservation hold to machine precision; shear
   -wave decay rate matches the Chapman-Enskog viscosity formula.
2. Phase 2: Poiseuille profile matches the analytic parabola; Zou-He
   recovers the exact prescribed boundary velocity.
3. Phase 3: centerline profiles track Ghia et al. at Re=100/400/1000,
   comparably to `MAC_Grid_Solver`'s own Phase 6 result.
4. Phase 4: measured Strouhal number at Re=100 (`St=0.202`, primary
   blockage) is a clean, precisely-determined single-frequency result;
   a lower-blockage control run confirms it's elevated by channel
   confinement as expected (`St` drops to `0.172`, much closer to
   Roshko/literature), with the lower-blockage run's more complex wake
   dynamics reported honestly rather than glossed over.
