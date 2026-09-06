"""Smooth absorbing boundary mask for the periodic supercell grid.

The ground-state SCF (scf3d.py) does not need this -- its density decays to
negligible amplitude well inside the box. It is here for rt-TDDFT
(Quantum Mechanics/TDDFT/) strong-field runs, where an ionizing pulse pushes
probability out toward the box edge: multiplying every orbital by this mask
once per timestep soaks up the outgoing flux (a soft complex-absorbing-
potential equivalent) instead of letting it wrap around the periodic
boundary and re-enter as spurious interference. The norm lost each step is
then a physical diagnostic -- the ionized fraction -- not an error.
"""

import numpy as np


def boundary_mask(grid, width, order=2):
    """1 in the interior, cosine taper to 0 over the outer `width` (Bohr) of
    the box on every face.

    grid: (x, X, Y, Z, dx) or (x, X, Y, Z, dx, G2) from grid3d.
    width: taper thickness in Bohr (e.g. 4.0).
    order: exponent on the cosine (2 = the usual cos^2 window).
    """
    x = grid[0]
    dx = x[1] - x[0]
    half_L = (x[-1] - x[0] + dx) / 2.0     # box half-length (periodic: L = N*dx)
    edge = half_L - width
    if edge <= 0:
        raise ValueError(f"boundary_mask: width {width} too large for box half-length {half_L}")

    def taper(coord):
        t = np.clip((np.abs(coord) - edge) / width, 0.0, 1.0)
        return np.cos(0.5 * np.pi * t) ** order

    X, Y, Z = grid[1], grid[2], grid[3]
    return taper(X) * taper(Y) * taper(Z)
