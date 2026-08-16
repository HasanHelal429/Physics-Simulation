# MAC Grid Navier–Stokes Solver — Design Plan

## Context

`Fluid Mechanics/Simple_Fluid/` already has a 2D fluid solver, but it's Jos Stam's "Stable Fluids" scheme: a collocated grid, fixed 20-iteration Jacobi relaxation for both diffusion and the pressure Poisson solve (no real convergence check, no guarantee divergence ever reaches zero), and RK1 semi-Lagrangian advection with bilinear interpolation. It's good for a nice-looking animation but isn't quantitatively trustworthy — there's no way to validate it against real physics.

The user wants a second, contrasting fluid project that IS quantitatively trustworthy: real Navier–Stokes numerics, a real convergent pressure solve, and results validated against known CFD benchmarks (not just "looks plausible"). Four solver families were discussed (Lattice Boltzmann, SPH, a higher-order Eulerian MAC-grid solver, FLIP/PIC); FLIP/PIC and SPH lean animation/VFX, LBM and the MAC-grid approach lean physics/engineering. The user chose **the MAC-grid route**: a real staggered Marker-and-Cell grid with a proper projection method, in the spirit of textbook/production CFD rather than computer-graphics fluid tricks.

**House style**, matching the two existing self-contained project precedents (`General_Relativity/Schwarzschild_Raytracer/`, `Quantum Mechanics/HF_solver/`): its own project folder with a `*_Plan.md` design doc, plain procedural functions (no classes), phases marked done as they're completed, physics/numerics derivations written inline rather than assumed, and a final Verification section chaining each phase's internal checks. Animations saved as MP4 via matplotlib's `ffmpeg` writer (the `imageio_ffmpeg`-bundled binary, already installed in `torch.venv`) into a `media/` folder, not just rendered inline — same convention `Simple_Fluid` uses.

**Deliberate departure from Simple_Fluid**: NumPy + SciPy (`scipy.sparse`, `scipy.sparse.linalg`) instead of PyTorch. Simple_Fluid used `torch` for GPU-accelerated animation; this project's goal is numerical correctness, not GPU speed, and SciPy's sparse direct/iterative solvers (`splu`, `cg`) are the mature, standard tool for a real Poisson solve — matching the numpy/scipy/matplotlib stack the GR project also uses.

## Numerical approach

**Staggered MAC grid** (Harlow & Welch 1965): `u` (x-velocity) lives on vertical cell faces, shape `(nx+1, ny)`; `v` (y-velocity) on horizontal cell faces, shape `(nx, ny+1)`; pressure `p` and the obstacle mask at cell centers, shape `(nx, ny)`. This is what actually fixes Simple_Fluid's biggest structural weakness: on a collocated grid, the discrete divergence and pressure-gradient stencils each skip a grid point (checkerboard null mode), which is why Simple_Fluid needed heavy Jacobi smoothing to hide it. On the MAC grid, divergence at `(i,j)` — `(u[i+1,j]-u[i,j])/dx + (v[i,j+1]-v[i,j])/dy` — and the pressure gradient at each face use only immediately-adjacent stored values, with no interpolation and no null mode.

**Face masks derived once**, in `grid.py`, from the cell-center obstacle mask: a `u`-face is solid iff either adjacent cell is solid (`u_mask`, shape `(nx+1,ny)`); same idea for `v_mask`. Every other module (`advection.py`, `diffusion.py`, `pressure.py`, `boundary.py`) consumes these two derived masks rather than re-deriving obstacle logic locally — the fragility of Simple_Fluid's ad hoc per-cell mask corrections came from not doing this.

**Projection method (Chorin 1968)** each step:
1. Advect `u^n` explicitly to get an intermediate velocity.
2. Diffuse implicitly to get `u*`.
3. **Enforce velocity BCs on `u*`** (needed before the divergence/Poisson step below reads them).
4. Solve the pressure Poisson equation `∇²p = (ρ/Δt)∇·u*`.
5. Correct: `u^{n+1} = u* − (Δt/ρ)∇p`.
6. **Re-enforce velocity BCs again** — the correction only touches face-normal velocities from the interior stencil and can leave boundary/ghost values stale.

That BC-touchpoint discipline (steps 3 and 6) is easy to silently break during later refactors, so it gets its own short section in `boundary.py`.

**Advection — conservative finite-volume + RK3**, not semi-Lagrangian. Semi-Lagrangian (what Simple_Fluid uses) is not written in flux-conservative form, and non-conservation error accumulates over the thousands of steps a shedding-frequency measurement needs — exactly the output this project cares about getting right. The CFD literature being validated against (Ghia et al., vortex-shedding papers) is essentially universally finite-volume/finite-difference in conservative form, not semi-Lagrangian (semi-Lagrangian's unconditional-stability selling point is a graphics-motivated tradeoff, precisely the thing being moved away from here). Concretely: face-flux convective terms (`u·u` at `u`-faces directly, `u·v` at cell corners via interpolation) with central differencing (2nd-order, energy-conservative in the inviscid limit), time-stepped with an explicit **3-stage RK3**. RK3 is a correctness requirement, not a style choice: von Neumann analysis shows centered-difference advection combined with forward Euler, RK2, or AB2 has no stability region overlapping the imaginary axis (all three are formally unconditionally unstable for this operator) — RK3 is the minimum explicit RK order that has a genuine CFL-bounded stable region. This construction follows Kim & Moin (1985) for the fractional-step method and Rai & Moin (1991) for the RK3 convective treatment. *Documented fallback*: if this proves like too much for the time available, RK2-midpoint semi-Lagrangian with Catmull-Rom cubic interpolation is a legitimate simpler substitute — track global kinetic-energy/momentum drift as a cheap diagnostic to see empirically what non-conservation costs before deciding.

**Diffusion — implicit, via CG with warm start**, not a direct factorization. The diffusion operator `I − Δt·ν·L_visc` depends on `Δt`; since `Δt` is adaptive (recomputed from CFL each step), a prefactorized solve would need re-factorizing whenever `Δt` changes, defeating the point. `L_visc` is symmetric positive definite, so `scipy.sparse.linalg.cg` warm-started from the previous step's velocity converges in a handful of iterations with no factorization needed at all.

**Pressure — direct sparse solve, prefactorized once**. Unlike diffusion, the pressure Laplacian is `Δt`-independent (only geometry-dependent), and geometry is static within a run — so `scipy.sparse.linalg.splu` factorized once and reused every step is valid and a large performance win. Neumann (zero-normal-flow) at walls and obstacle faces; Dirichlet `p=0` at any outflow boundary. **Null-space case**: the lid-driven-cavity benchmark (Phase 6) has no outflow, so every boundary condition is Neumann — the assembled matrix is exactly singular (adding a constant to `p` doesn't change `∇p`). Fix: pin one reference cell (identity row, `p[0,0]=0`) before factorizing, and check the compatibility condition (`sum(div(u*)) ≈ 0`, which follows from zero normal velocity everywhere) as an internal Phase 4 test. Report `max|∇·u^{n+1}|` after every projection as a concrete diagnostic — something Simple_Fluid's fixed-iteration Jacobi could never actually guarantee reached zero.

**Non-dimensionalization**: `Re = UL/ν`, so every scenario is describable and comparable to literature purely in terms of Reynolds number.

**Adaptive timestep**: `dt = CFL_number * dx / max(|u|, |v|, eps)`, reported as a per-step diagnostic.

**Known limitation, stated up front rather than discovered later**: a boolean cell mask gives a staircase approximation to curved geometry (the cylinder in Phase 7), not a true circular boundary. Keep the cylinder spanning ≥25–30 cells across its diameter so staircase error stays small relative to the scheme's own truncation error; a cut-cell/ghost-cell immersed-boundary treatment is a legitimate future refinement, not required for the first pass.

## File layout

```
Fluid Mechanics/MAC_Grid_Solver/
    MAC_Grid_Solver_Plan.md        # this design document
    grid.py                        # Phase 1: staggered grid, face-mask derivation, cross-interpolation, vorticity
    operators.py                   # Phase 1: divergence/gradient array ops + generic sparse Laplacian assembly
    advection.py                   # Phase 2: conservative FV central-difference convection + RK3 (SL RK2/Catmull-Rom documented fallback)
    diffusion.py                   # Phase 3: implicit (backward-Euler/CN) viscous solve via CG + warm start
    pressure.py                    # Phase 4: Poisson assembly (Neumann walls/obstacles, Dirichlet outflow), null-space handling, splu, projection
    boundary.py                    # Phase 5: velocity BC application — walls, obstacle no-slip, inflow/outflow
    solver.py                      # Phase 5: step() orchestration (advect -> diffuse -> BC -> project -> BC) + adaptive CFL dt
    Validation.ipynb               # Phases 1-5: each phase's checks, run + plotted + saved to media/, one section per phase
    Lid_Driven_Cavity.ipynb        # Phase 6: Ghia et al. (1982) benchmark
    Cylinder_Vortex_Shedding.ipynb # Phase 7: Strouhal-Reynolds / von Kármán vortex street
    media/
```

Phases 1-5 have no standalone demo of their own (they're solver internals), so rather than leaving their validation as throwaway console output, every phase's checks are run and plotted in `Validation.ipynb`, one section per phase, with each figure saved to `media/` — so the evidence a phase actually works is a real artifact, not just a transcript.

## Phase 1 — Grid & operators (`grid.py`, `operators.py`) ✅ done

- Staggered index conventions above; derive `u_mask`/`v_mask` from the cell-center obstacle mask.
- Interpolation helpers for cross-terms: `v` sampled at `u`-locations and vice versa (needed for advection's corner-averaged flux terms and for vorticity).
- `operators.py`: plain-array divergence/gradient (diagnostics, vorticity `ζ = ∂v/∂x − ∂u/∂y` at cell corners), plus a generic sparse Laplacian builder parameterized by per-boundary/per-face BC type (Neumann-drop-neighbor vs. Dirichlet-ghost) — shared by `diffusion.py` and `pressure.py` even though the two resulting matrices differ.
- **Validate**: divergence of a known analytic solenoidal field (e.g. a Taylor–Green vortex sampled onto the staggered grid) is zero to machine precision; the sparse Laplacian applied to a known quadratic field matches its analytic curvature; face masks correctly reproduce a small hand-drawn test obstacle.

## Phase 2 — Advection (`advection.py`) ✅ done

- Derive the discrete convective flux stencils for `u`- and `v`-momentum (face-averaged velocity products: `u·u` directly at `u`-faces, `u·v` at cell corners via interpolation — write out the exact averaging, easy to get an off-by-one here).
- Implement 3-stage explicit RK3 for the advective sub-step.
- **Validate**: a Gaussian blob advected in a uniform flow keeps its shape with only convergence-order-consistent numerical diffusion after one grid-crossing; halving `dx` shows ~2nd-order spatial convergence; a rigid-body-rotation field advects a blob around a closed circular path without excessive spreading.

## Phase 3 — Diffusion (`diffusion.py`) ✅ done

- Backward-Euler/Crank–Nicolson discretization of `∂u/∂t = ν∇²u`; build the (`dt`-dependent) SPD operator via `operators.py`'s builder with Dirichlet ghost treatment; solve via `scipy.sparse.linalg.cg` warm-started each step.
- **Validate**: diffusion of a sinusoidal initial condition decays at the analytic rate `e^{-ν k² t}`; a no-flow, no-forcing steady state stays exactly at rest (regression check that BC/mask handling can't leak spurious velocity).

## Phase 4 — Pressure projection (`pressure.py`) ✅ done

- Derive the Helmholtz–Hodge projection theorem and the discrete Poisson equation; assemble Neumann-at-wall/obstacle, Dirichlet-at-outflow Laplacian; handle the pure-Neumann null space (pin one reference cell) for the enclosed-cavity case; prefactorize with `splu` and reuse for the whole run; implement the correction step and the `max|∇·u^{n+1}|` diagnostic.
- **Validate**: solving Poisson for a prescribed analytic RHS (no obstacle) recovers the known analytic pressure field; projecting a divergent test field drives `max|div|` to near machine precision; for the closed-cavity case, confirm the compatibility condition (`sum(div(u*)) ≈ 0`) and that the pinned-cell solution matches the unpinned one up to a constant offset.

## Phase 5 — Boundary conditions & solver assembly (`boundary.py`, `solver.py`) ✅ done

- Wall BCs: no-slip (ghost values), free-slip, prescribed inflow profile, zero-gradient-velocity + Dirichlet-`p` outflow. Obstacle no-slip applied on staggered faces via `u_mask`/`v_mask`.
- `step(state, params)`: advect → diffuse → enforce BC → project → enforce BC, plus adaptive-CFL `dt`.
- **Validate**: a no-obstacle, zero-forcing box with only wall BCs stays at rest indefinitely (regression test for BC bugs); a simple channel-flow setup relaxes to the analytic parabolic Poiseuille profile at steady state — a genuinely independent full-pipeline check (advection+diffusion+pressure+BCs all participating) before trusting either benchmark notebook.

## Phase 6 — Validation: lid-driven cavity (`Lid_Driven_Cavity.ipynb`) ✅ done

- Unit square, `L=U=ρ=1`, `ν=1/Re`, top lid moving at `U=1`, no-slip on the other three walls. Run to steady state (monitor `max|∇·u|` and per-step kinetic-energy change as the convergence criterion) at Re = 100, 400, 1000.
- Compare steady-state centerline `u(y)` at `x=0.5` and `v(x)` at `y=0.5` against **Ghia, Ghia & Shin (1982)**, *J. Comput. Phys.* 48(3):387–411 — the standard tabulated reference (129×129 grid, vorticity–streamfunction + multigrid; only the converged physical steady state is being compared, not the numerical method).
- Visualize streamlines/vorticity: at Re=100 expect a single dominant primary vortex with a barely-visible bottom-right secondary corner vortex; by Re=1000 both bottom-left and bottom-right secondary corner vortices should be clearly resolved (a good qualitative check alongside the quantitative profile comparison).

## Phase 7 — Validation: cylinder vortex shedding (`Cylinder_Vortex_Shedding.ipynb`) ⏸ paused, unresolved

**Status**: paused mid-investigation, not complete. A full run at cells/D=12 passed its own checks but turned out to be measuring an unsaturated linear instability (~1e-12 amplitude), not real shedding. Re-running properly at the plan's target resolution (cells/D=25, newly affordable after a ~90x `diffusion.py` speedup found along the way — see below) surfaced a real, unresolved puzzle: coarse resolution (cells/D=10-12) robustly produces large saturated shedding (~0.4 amplitude) regardless of perturbation size/blockage/wall type, while fine resolution (cells/D=25) settles into a small stable oscillation (~0.02) regardless of how hard it's perturbed (tried up to 0.2D offsets). Two real solver bugs were found and fixed along the way (`advection.py`'s outflow-boundary wraparound, `operators.gradient()`'s mask-unawareness) and remain fixed regardless of the shedding question. Full writeup, diagnostic scripts, and logs in `phase7_investigation/README.md`.

- Channel domain: cylinder diameter `D`, blockage ratio `D/H ≤ 0.1–0.125` (channel confinement measurably raises shedding frequency relative to the unbounded correlation — keep it small rather than correcting for it), ~5–8`D` upstream and ~20–25`D` downstream of the cylinder so the wake fully develops before the outflow boundary (reflection artifacts there can contaminate the shedding signal). Cylinder spans ≥25–30 cells across its diameter (staircase-geometry accuracy, per the known limitation above).
- Headline case: **Re ≈ 100** — comfortably inside the 2D laminar periodic-shedding window (onset at `Re_c ≈ 47`; real (3D) wakes transition around `Re ≈ 190`, past which a 2D simulation is no longer comparable to real experimental data, so don't push much higher).
- Place a velocity or lift-coefficient probe downstream; run to a statistically steady periodic state; extract the shedding frequency via FFT or zero-crossing timing; compute `St = fD/U∞`.
- Compare against **two** independent references: Roshko's empirical correlation `St ≈ 0.198(1 − 19.7/Re)` (gives `St ≈ 0.159` at Re=100) and literature point values at Re=100 (commonly `St ≈ 0.164–0.167` in 2D CFD benchmark papers) — both should roughly bracket the measured value.
- Animate the vorticity field (von Kármán vortex street) to `media/` as MP4.

## Progress

- [x] Phase 1 — `grid.py`, `operators.py`
- [x] Phase 2 — `advection.py`
- [x] Phase 3 — `diffusion.py`
- [x] Phase 4 — `pressure.py`
- [x] Phase 5 — `boundary.py`, `solver.py`
- [x] Phase 6 — `Lid_Driven_Cavity.ipynb` (64x64, chosen for interactive runtime over Ghia's 129x129)
- [ ] Phase 7 — `Cylinder_Vortex_Shedding.ipynb`

## Verification

Each phase's internal checks build on the last — don't trust a later phase until the earlier ones pass:

1. Phase 1: analytic-divergence and Laplacian-curvature checks pass to machine/truncation precision.
2. Phase 2/3: advection shows the expected ~2nd-order convergence rate under grid refinement; diffusion matches the analytic exponential decay rate.
3. Phase 4: projection drives divergence to near machine precision on a test field; the closed-cavity null-space fix is confirmed consistent (pinned vs. unpinned solutions differ only by a constant).
4. Phase 5: the at-rest regression test and the Poiseuille-flow full-pipeline check both pass — this is the last "boring but load-bearing" gate before the two benchmarks.
5. Phase 6: centerline velocity profiles visually and quantitatively track Ghia et al. at Re=100/400/1000; secondary corner vortices appear at the expected Reynolds numbers.
6. Phase 7: measured Strouhal number at Re=100 falls close to both the Roshko correlation and literature point values; the vorticity animation shows a clean, periodic von Kármán street.
