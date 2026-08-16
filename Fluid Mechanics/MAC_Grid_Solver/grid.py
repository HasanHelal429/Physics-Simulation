"""
Staggered Marker-and-Cell (MAC) grid for the 2D incompressible Navier-Stokes
solver (Harlow & Welch 1965). See MAC_Grid_Solver_Plan.md for the numerical
approach and derivations.

Layout on an nx x ny grid of cells spanning [0, lx] x [0, ly]:
  u (x-velocity): vertical cell faces,   x = i*dx (i=0..nx),     y = (j+0.5)*dy (j=0..ny-1) -> shape (nx+1, ny)
  v (y-velocity): horizontal cell faces, x = (i+0.5)*dx (i=0..nx-1), y = j*dy (j=0..ny)      -> shape (nx, ny+1)
  p (pressure), obstacle mask: cell centers, x=(i+0.5)*dx, y=(j+0.5)*dy                       -> shape (nx, ny)
"""

from collections import namedtuple

import numpy as np

Grid = namedtuple('Grid', ['nx', 'ny', 'dx', 'dy', 'cell_mask', 'u_mask', 'v_mask'])


def face_masks(cell_mask):
    """Derive u/v-face fluid masks from a cell-center obstacle mask (True = fluid).

    A face is fluid iff every cell it touches is fluid: both neighbors for
    an interior face, the single neighbor for a domain-boundary face (a
    face is never blocked just for being on the domain edge -- that's
    governed separately by boundary conditions, not by the obstacle mask).
    """
    nx, ny = cell_mask.shape
    u_mask = np.empty((nx + 1, ny), dtype=bool)
    u_mask[1:-1, :] = cell_mask[:-1, :] & cell_mask[1:, :]
    u_mask[0, :] = cell_mask[0, :]
    u_mask[-1, :] = cell_mask[-1, :]

    v_mask = np.empty((nx, ny + 1), dtype=bool)
    v_mask[:, 1:-1] = cell_mask[:, :-1] & cell_mask[:, 1:]
    v_mask[:, 0] = cell_mask[:, 0]
    v_mask[:, -1] = cell_mask[:, -1]
    return u_mask, v_mask


def make_grid(nx, ny, lx, ly, cell_mask=None):
    dx, dy = lx / nx, ly / ny
    if cell_mask is None:
        cell_mask = np.ones((nx, ny), dtype=bool)
    u_mask, v_mask = face_masks(cell_mask)
    return Grid(nx, ny, dx, dy, cell_mask, u_mask, v_mask)


def circle_obstacle(nx, ny, lx, ly, center, radius):
    """Cell-center fluid mask (True = fluid) with a circular obstacle cut out."""
    x = (np.arange(nx) + 0.5) * (lx / nx)
    y = (np.arange(ny) + 0.5) * (ly / ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return (X - center[0]) ** 2 + (Y - center[1]) ** 2 > radius ** 2


def velocity_at_centers(u, v):
    """Interpolate (u, v) onto cell centers, shape (nx, ny) each -- for
    visualization and diagnostics (not used in the solver's inner loop)."""
    u_c = 0.5 * (u[:-1, :] + u[1:, :])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    return u_c, v_c


def interp_v_to_u(v):
    """Interpolate v (shape (nx, ny+1)) onto u's staggered locations (shape (nx+1, ny)).

    Two-stage bilinear average: first average adjacent v columns onto the
    interior u x-locations (one-sided copy at the two domain-edge
    x-locations, where only one v column exists), then average adjacent
    y-samples onto u's y-locations (always interior to v's y-range, no
    edge case needed there).
    """
    x_interior = 0.5 * (v[:-1, :] + v[1:, :])
    v_x = np.concatenate([v[:1, :], x_interior, v[-1:, :]], axis=0)
    return 0.5 * (v_x[:, :-1] + v_x[:, 1:])


def interp_u_to_v(u):
    """Interpolate u (shape (nx+1, ny)) onto v's staggered locations (shape (nx, ny+1)).
    Mirror of `interp_v_to_u`, averaging over y first then x."""
    y_interior = 0.5 * (u[:, :-1] + u[:, 1:])
    u_y = np.concatenate([u[:, :1], y_interior, u[:, -1:]], axis=1)
    return 0.5 * (u_y[:-1, :] + u_y[1:, :])
