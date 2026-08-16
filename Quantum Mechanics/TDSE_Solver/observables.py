"""
Diagnostics computed from a wavefunction on a grid.Grid: probability
density, total norm, position/momentum expectation values, and
region-integrated probability (e.g. what fraction of |psi|^2 has crossed
past a barrier -- a transmission probability).
"""

import numpy as np
from scipy.fft import dstn

import propagator as prop


def probability_density(psi):
    """|psi|^2, same shape as psi."""
    return np.abs(psi) ** 2


def norm(psi, grid):
    """Continuum normalization: integral |psi|^2 dV ~ sum(|psi|^2) * cell_volume.
    A freshly-built grid.gaussian_wavepacket starts at ~1; under
    propagator.strang_step alone this stays ~1 to machine precision
    (Phase 1/2); it only drops once an absorbing boundary (potentials.
    absorbing_boundary) is added to V, by design -- the lost norm *is*
    the probability flux that left through the CAP."""
    cell_volume = np.prod(grid.dx)
    return np.sum(probability_density(psi)) * cell_volume


def expectation_position(psi, grid):
    """<x_i> per axis, as a length-ndim array. The cell volume that would
    normally scale both the numerator and denominator integrals cancels
    in this ratio, so it isn't needed explicitly."""
    density = probability_density(psi)
    total = np.sum(density)
    return np.array([np.sum(x * density) / total for x in grid.coords])


def expectation_momentum(psi, grid):
    """<p_i> per axis. On a periodic grid, p is diagonal in the same FFT
    basis kinetic_step already uses (p <-> k), so this reuses that basis
    directly. On a box (hard-wall) grid the DST-I basis diagonalizes
    -d^2/dx^2, not d/dx, so <p> is instead evaluated directly from a
    centered finite difference of psi (2nd order accurate) with psi
    assumed exactly 0 just outside the array, matching the box's own
    Dirichlet convention."""
    density = probability_density(psi)
    total = np.sum(density)
    if grid.boundary == 'periodic':
        psi_hat = np.fft.fftn(psi)
        density_k = np.abs(psi_hat) ** 2
        total_k = np.sum(density_k)
        k_components = np.meshgrid(*grid.k_axes, indexing='ij')
        return np.array([np.sum(k * density_k) / total_k for k in k_components])
    p = []
    for axis in range(grid.ndim):
        psi_padded = np.pad(psi, [(1, 1) if a == axis else (0, 0) for a in range(grid.ndim)])
        slicer_hi = tuple(slice(2, None) if a == axis else slice(None) for a in range(grid.ndim))
        slicer_lo = tuple(slice(0, -2) if a == axis else slice(None) for a in range(grid.ndim))
        dpsi_dx = (psi_padded[slicer_hi] - psi_padded[slicer_lo]) / (2 * grid.dx[axis])
        integrand = np.conj(psi) * (-1j) * dpsi_dx
        p.append(np.real(np.sum(integrand)) / total)
    return np.array(p)


def expectation_energy(psi, grid, V):
    """<H> = <T> + <V> for H = -1/2*laplacian + V, with <T> evaluated in
    the same spectral basis propagator.kinetic_step actually uses (FFT
    for periodic, DST-I for box) -- so this measures exactly the
    quantity strang_step conserves (H has no explicit time dependence),
    rather than an independently-discretized approximation to it the way
    stationary_states.hamiltonian's finite-difference operator is. Used
    in Validation.ipynb (Phase 7) to confirm energy conservation over a
    long run in a potential with no simple closed-form trajectory to
    check against instead."""
    density = probability_density(psi)
    total = np.sum(density)
    V_expect = np.sum(V * density) / total

    if grid.boundary == 'periodic':
        psi_hat = np.fft.fftn(psi)
    else:
        psi_hat = dstn(psi, type=1)
    density_k = np.abs(psi_hat) ** 2
    total_k = np.sum(density_k)
    k2 = prop.kinetic_eigenvalues(grid)
    T_expect = 0.5 * np.sum(k2 * density_k) / total_k

    return T_expect + V_expect


def region_probability(psi, grid, axis, x_min=None, x_max=None):
    """integral of |psi|^2 dV restricted to x_min <= x_axis <= x_max along
    `axis` (integrated over every other axis), e.g. the probability found
    past a barrier or beyond a detection screen. Returns an absolute
    value (same continuum normalization as `norm`); divide by the
    wavefunction's initial norm for a transmission/reflection
    probability if the run also has lossy absorbing boundaries active."""
    density = probability_density(psi)
    x = grid.axes[axis]
    mask_1d = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask_1d &= x >= x_min
    if x_max is not None:
        mask_1d &= x <= x_max
    shape = [1] * grid.ndim
    shape[axis] = len(x)
    mask = mask_1d.reshape(shape)
    cell_volume = np.prod(grid.dx)
    return np.sum(density * mask) * cell_volume
