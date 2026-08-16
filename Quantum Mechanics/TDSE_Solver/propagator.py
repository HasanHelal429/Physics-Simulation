"""
Split-operator (Strang-splitting) time propagation for the time-dependent
Schrodinger equation, i*psi_t = H*psi = (-1/2 laplacian + V) psi (atomic
units, hbar=m=1).

    e^{-i H dt} ~= e^{-i V dt/2} . e^{-i T dt} . e^{-i V dt/2}      (*)

is accurate to O(dt^3) per step (2nd order globally) by the symmetric
(Strang) splitting identity -- exact if V and T=-1/2 laplacian commuted,
which they don't in general (V is diagonal in position space, T is
diagonal in the spectral/momentum basis, and multiplication doesn't
commute with the Laplacian). T's propagator is applied as transform ->
multiply by a fixed phase -> inverse transform, using grid.k_axes; V's
propagator is a plain elementwise multiply in real space. That split --
spectral for T, real-space for V -- is what makes one implementation work
unchanged in 1D/2D/3D and under either boundary convention in grid.py.

Both the FFT (periodic) and DST-I (box) transform pairs used below are
exact matched forward/inverse transforms (idstn undoes dstn, ifftn undoes
fftn) and both families are unitary up to a fixed real overall scale
regardless of normalization convention -- so inserting a unit-modulus
phase array between the forward and inverse transform conserves the L2
norm (Parseval) exactly to floating-point precision. That's what makes
kinetic_step norm-conserving on its own, independent of the potential step.
"""

import numpy as np
from scipy.fft import dstn, idstn


def kinetic_eigenvalues(grid):
    """k^2 = kx^2 + ky^2 + ... at every grid point, shape == grid.shape.
    This is the eigenvalue of T=-1/2 laplacian (times 2) in whichever
    spectral basis grid.boundary selects -- pass it into kinetic_step to
    avoid recomputing it every call when dt is fixed across many steps."""
    k_components = np.meshgrid(*grid.k_axes, indexing='ij')
    return sum(k ** 2 for k in k_components)


def _forward_transform(psi, grid):
    if grid.boundary == 'periodic':
        return np.fft.fftn(psi)
    return dstn(psi, type=1)


def _inverse_transform(psi_hat, grid):
    if grid.boundary == 'periodic':
        return np.fft.ifftn(psi_hat)
    return idstn(psi_hat, type=1)


def kinetic_step(psi, grid, dt, k2=None):
    """Apply e^{-i T dt} exactly (T = -1/2 laplacian), one full step."""
    if k2 is None:
        k2 = kinetic_eigenvalues(grid)
    psi_hat = _forward_transform(psi, grid)
    psi_hat = psi_hat * np.exp(-1j * k2 * dt / 2)
    return _inverse_transform(psi_hat, grid)


def potential_step(psi, V, dt):
    """Apply e^{-i V dt} -- V is diagonal in real space, so this is just
    an elementwise multiply. Named/used as a half-step (dt/2) inside
    strang_step, but takes whatever dt it's given so it can also serve a
    full step on its own (e.g. a pure V=0 sanity check)."""
    return psi * np.exp(-1j * V * dt)


def strang_step(psi, grid, V, dt, k2=None):
    """One full step of the symmetric split-operator method:

        e^{-i H dt} ~= e^{-i V dt/2} . e^{-i T dt} . e^{-i V dt/2}

    2nd-order accurate in dt (the splitting error is O(dt^3) per step,
    O(dt^2) globally) since V and T don't commute in general -- exact
    only in the special case V=0, where the middle kinetic_step is itself
    an exact (not approximate) application of the free-particle
    propagator and the potential half-steps are identity."""
    psi = potential_step(psi, V, dt / 2)
    psi = kinetic_step(psi, grid, dt, k2=k2)
    psi = potential_step(psi, V, dt / 2)
    return psi
