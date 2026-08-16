# Fluid Mechanics

Fluid simulation projects, built up project by project.

## Projects

- [`Simple_Fluid/`](Simple_Fluid/) — a 2D Stable Fluids solver (Jos Stam,
  1999) on a PyTorch grid, with an open-domain demo and an obstacle-mask
  demo sharing one solver module. See its [README](Simple_Fluid/README.md).
- [`MAC_Grid_Solver/`](MAC_Grid_Solver/) — a real 2D incompressible
  Navier-Stokes solver on a staggered Marker-and-Cell grid (conservative
  finite-volume advection + RK3, implicit diffusion, a proper convergent
  pressure Poisson solve), validated against literature CFD benchmarks
  (lid-driven cavity, cylinder vortex shedding / Strouhal number) rather
  than just rendered for looks. See its
  [design plan](MAC_Grid_Solver/MAC_Grid_Solver_Plan.md).

Future additions will live as sibling project folders here, each with its
own README (and, for anything nontrivial, a design plan).
