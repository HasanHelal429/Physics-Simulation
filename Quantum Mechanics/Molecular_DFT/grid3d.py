"""Uniform 3D Cartesian grid + FFT G-vector (spatial frequency) utilities
for the periodic/supercell approach (see Molecular_DFT_Plan.md): a
molecule sits in a cubic box of side length L with periodic boundary
conditions, vacuum-padded so its density is negligible at the boundary.
"""

import numpy as np


def make_grid(L, N):
    """Uniform grid on a periodic cubic box of side length L (Bohr), N
    points per axis, centered on the box (so a molecule can be placed
    near the origin without straddling the periodic boundary). Returns
    (x, X, Y, Z, dx): x is the 1D coordinate array, X/Y/Z the 3D
    meshgrid, dx the grid spacing.
    """
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return x, X, Y, Z, dx


def g_vectors(L, N):
    """Angular spatial-frequency (G) 3D meshgrid for FFT-based
    differentiation on a periodic box of side length L, N points per
    axis: G = 2*pi*m/L for integer m (the box's reciprocal lattice
    vectors), via np.fft.fftfreq's cycles-per-length convention times
    2*pi. Returns (Gx, Gy, Gz, G2) with G2 = Gx**2+Gy**2+Gz**2.
    """
    dx = L / N
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    Gx, Gy, Gz = np.meshgrid(k, k, k, indexing="ij")
    G2 = Gx**2 + Gy**2 + Gz**2
    return Gx, Gy, Gz, G2
