"""D2Q9 lattice Boltzmann core: equilibrium distribution, macroscopic
moments, BGK collision, and streaming, in lattice units (dx = dt = 1).

Distribution functions f are stored as an array of shape (9, nx, ny);
axis 0 = "x", axis 1 = "y" (matching this repo's other grid solvers).

Lattice speed of sound c_s^2 = 1/3. The equilibrium below is the
standard 2nd-order-in-Mach expansion of the Maxwell-Boltzmann
distribution onto the D2Q9 velocity set, valid for |u| << c_s -- keep
lattice velocities small (~0.05-0.1) so the expansion stays accurate.
"""
import numpy as np

# Directions: rest, 4 axis-aligned, 4 diagonal.
E = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=np.int64)

W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

# OPPOSITE[i] is the direction index of -E[i] (used for bounce-back).
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

CS2 = 1.0 / 3.0


def equilibrium(rho, ux, uy):
    """f_i^eq = w_i*rho*(1 + 3(e_i.u) + 4.5(e_i.u)^2 - 1.5|u|^2)."""
    eu = E[:, 0, None, None] * ux[None, :, :] + E[:, 1, None, None] * uy[None, :, :]
    u2 = ux ** 2 + uy ** 2
    return W[:, None, None] * rho[None, :, :] * (
        1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * u2[None, :, :]
    )


def moments(f):
    """rho = sum_i f_i, rho*u = sum_i f_i*e_i."""
    rho = f.sum(axis=0)
    ux = np.tensordot(E[:, 0], f, axes=(0, 0)) / rho
    uy = np.tensordot(E[:, 1], f, axes=(0, 0)) / rho
    return rho, ux, uy


def collide(f, tau):
    """BGK relaxation toward local equilibrium: f <- f - (f - f^eq)/tau."""
    rho, ux, uy = moments(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def stream(f):
    """f_i(x + e_i, t+1) = f_i(x, t): an exact per-direction array roll."""
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(f[i], shift=(E[i, 0], E[i, 1]), axis=(0, 1))
    return out
