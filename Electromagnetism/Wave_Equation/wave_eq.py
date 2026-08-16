"""Shared 2D scalar wave equation solver: explicit leapfrog finite
differences for d^2(psi)/dt^2 = c^2 * laplacian(psi), with an optional
spatially-varying wave speed, a driving source term, and Dirichlet
boundaries applied inside an arbitrary mask.
"""
import numpy as np


def cfl_dt(dx, dy, c_max, courant=0.5):
    """The correct 2D wave-equation CFL stability bound is
    dt <= min(dx,dy)/(c*sqrt(2)) -- not the diffusion-style dt ~ dx^2/c
    the legacy notebooks used (over-conservative for this equation, and
    the wrong physical scaling to carry into a validated module)."""
    return courant * min(dx, dy) / (c_max * np.sqrt(2))


def leapfrog_step(psi_prev, psi_curr, dt, dx, dy, c, mask=None, source=None):
    """One explicit leapfrog step. `c` may be a scalar or an array shaped
    like psi (spatially varying wave speed). `mask`, if given, is a
    boolean array (True = inside the domain); psi is forced to zero
    outside the mask and on its boundary (Dirichlet). `source`, if given,
    is an array added to the result (a driving term)."""
    c2 = c**2 if np.isscalar(c) else c[1:-1, 1:-1]**2
    lap = ((psi_curr[2:, 1:-1] - 2 * psi_curr[1:-1, 1:-1] + psi_curr[:-2, 1:-1]) / dx**2
           + (psi_curr[1:-1, 2:] - 2 * psi_curr[1:-1, 1:-1] + psi_curr[1:-1, :-2]) / dy**2)

    psi_next = np.zeros_like(psi_curr)
    psi_next[1:-1, 1:-1] = 2 * psi_curr[1:-1, 1:-1] - psi_prev[1:-1, 1:-1] + c2 * dt**2 * lap

    if mask is not None:
        psi_next[~mask] = 0.0
        psi_next[0, mask[0, :]] = 0.0
        psi_next[-1, mask[-1, :]] = 0.0
        psi_next[mask[:, 0], 0] = 0.0
        psi_next[mask[:, -1], -1] = 0.0
    else:
        psi_next[0, :] = 0.0
        psi_next[-1, :] = 0.0
        psi_next[:, 0] = 0.0
        psi_next[:, -1] = 0.0

    if source is not None:
        psi_next = psi_next + source
    return psi_next
