# Simple_Fluid

A 2D Stable Fluids solver (Jos Stam, 1999) implemented with PyTorch tensors,
with two notebooks demonstrating it.

## Structure

- [`fluid_sim.py`](fluid_sim.py) — the solver itself: boundary conditions,
  diffusion, semi-Lagrangian advection, pressure projection, and the
  velocity/density step functions. Every step function takes an optional
  boolean `mask` (True = fluid, False = solid obstacle); pass `mask=None`
  for an open domain.
- [`Simple Fluid Sim.ipynb`](Simple%20Fluid%20Sim.ipynb) — density/velocity
  advecting through an open domain.
- [`Fluid Sim Obstacle.ipynb`](Fluid%20Sim%20Obstacle.ipynb) — the same
  solver with flow routed around a solid obstacle mask.
- `media/` — PNGs and MP4s produced by running the notebooks (initial
  conditions, before/after density comparisons, and the animations
  themselves). Notebooks write here rather than only rendering inline.

## Notes

- Animations are saved as MP4 via the `ffmpeg` writer (using the
  `imageio_ffmpeg`-bundled binary, so no separate ffmpeg install is needed)
  rather than GIF — much smaller files for the same frame count.
- Both notebooks import `fluid_sim.py`, so changes to the solver only need
  to be made in one place.
