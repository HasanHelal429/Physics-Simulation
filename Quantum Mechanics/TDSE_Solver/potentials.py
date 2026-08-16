"""
Potential-energy landscapes V(r) for the split-operator propagator
(propagator.py), plus a couple of closed-form reference solutions used
only to validate the propagator against known physics, not part of the
propagation pipeline itself.
"""

import math

import numpy as np
from scipy import special


def harmonic_well(grid, omega, center=None):
    """Isotropic quantum harmonic oscillator, V(r) = 1/2 * omega^2 * |r-center|^2
    (atomic units). Its exact eigenstates (harmonic_eigenstate below) are
    what Validation.ipynb uses to check the propagator against a case
    where V and T=-1/2 laplacian genuinely don't commute -- unlike a free
    particle (V=0), so there's a real O(dt^2) splitting error to measure."""
    if center is None:
        center = np.zeros(grid.ndim)
    center = np.broadcast_to(center, (grid.ndim,))
    r2 = sum((x - c) ** 2 for x, c in zip(grid.coords, center))
    return 0.5 * omega ** 2 * r2


def harmonic_eigenstate(grid, omega, n=0, center=None):
    """Exact eigenstate of harmonic_well(grid, omega, center): mode `n`
    (Hermite polynomial H_n) along axis 0, ground state along every other
    axis. Returns (psi, energy) with energy = omega*(n+1/2) + omega/2*(ndim-1).

    This is a genuine closed-form solution of the full (kinetic +
    potential) time-independent Schrodinger equation, so propagating it
    and checking psi(t) == psi(0)*exp(-i*energy*t) is a strong,
    independent test of strang_step -- it doesn't rely on any numerical
    diagonalization routine (stationary_states.py, Phase 4) being correct
    first.
    """
    if center is None:
        center = np.zeros(grid.ndim)
    center = np.broadcast_to(center, (grid.ndim,))
    psi = np.ones(grid.shape, dtype=complex)
    energy = 0.0
    for axis, (x, c) in enumerate(zip(grid.coords, center)):
        this_n = n if axis == 0 else 0
        xi = np.sqrt(omega) * (x - c)
        herm = special.hermite(this_n, monic=False)
        norm = (omega / np.pi) ** 0.25 / np.sqrt(float(2 ** this_n * math.factorial(this_n)))
        psi = psi * norm * herm(xi) * np.exp(-xi ** 2 / 2)
        energy += omega * (this_n + 0.5)
    return psi, energy


def rectangular_barrier(grid, V0, x_min, x_max, axis=0):
    """V0 for x_min <= x_axis <= x_max along `axis`, else 0 -- a rectangular
    potential barrier, uniform across every other axis (so it's usable as
    a 1D barrier or as a wall segment in a higher-dimensional slit
    geometry). See rectangular_barrier_transmission for the exact 1D
    plane-wave transmission probability through this shape."""
    x = grid.axes[axis]
    barrier_1d = np.where((x >= x_min) & (x <= x_max), V0, 0.0)
    shape = [1] * grid.ndim
    shape[axis] = len(x)
    return np.broadcast_to(barrier_1d.reshape(shape), grid.shape).astype(float).copy()


def rectangular_barrier_transmission(E, V0, a):
    """Exact transmission probability of a monochromatic plane wave of
    energy E through a 1D rectangular barrier of height V0, width a
    (atomic units) -- the standard textbook result (e.g. Griffiths,
    Introduction to Quantum Mechanics), covering both the tunneling
    (E<V0, using sinh) and over-barrier (E>V0, using sin -- and showing
    resonant perfect transmission whenever sqrt(2(E-V0))*a is a multiple
    of pi) regimes, plus the E=V0 limit.

    A real (non-monochromatic) Gaussian wavepacket has a nonzero energy
    spread (~1/(4*sigma0) in this convention), so a wavepacket-integrated
    transmission probability will differ from this pointwise formula by
    an amount that grows wherever T(E) is highly curved -- most visibly in
    the deep-tunneling regime, where T is exponentially sensitive to E.
    That's an expected, physical systematic (see Validation.ipynb, Phase
    5), not a sign either number is wrong.
    """
    scalar_input = np.ndim(E) == 0
    E = np.atleast_1d(np.asarray(E, dtype=float))
    T = np.empty_like(E)

    tunneling = E < V0
    over = E > V0
    at_top = ~(tunneling | over)

    kappa = np.sqrt(2 * (V0 - E[tunneling]))
    T[tunneling] = 1.0 / (1.0 + (V0 ** 2 * np.sinh(kappa * a) ** 2) / (4 * E[tunneling] * (V0 - E[tunneling])))

    k2 = np.sqrt(2 * (E[over] - V0))
    T[over] = 1.0 / (1.0 + (V0 ** 2 * np.sin(k2 * a) ** 2) / (4 * E[over] * (E[over] - V0)))

    T[at_top] = 1.0 / (1.0 + V0 * a ** 2 / 2)

    return float(T[0]) if scalar_input else T


def soft_coulomb_well(grid, strength, softening, center=None):
    """V(r) = -strength / sqrt(r^2 + softening^2) -- an attractive
    Coulomb-like well, softened near r=0 so the singularity (which no
    finite grid could resolve anyway) doesn't appear; reduces to the
    ordinary Coulomb potential -strength/r for r >> softening."""
    if center is None:
        center = np.zeros(grid.ndim)
    center = np.broadcast_to(center, (grid.ndim,))
    r2 = sum((x - c) ** 2 for x, c in zip(grid.coords, center))
    return -strength / np.sqrt(r2 + softening ** 2)


def double_slit_barrier(grid, V0, x_min, x_max, slit_width, slit_separation, axis=0, slit_axis=1):
    """A wall of height V0 spanning x_min<=x_axis<=x_max along `axis`,
    pierced by two slits of width `slit_width` centered at
    +-slit_separation/2 along `slit_axis` -- 0 everywhere else (including
    outside the wall's thickness). Analytic reference: for an incident
    wave of de Broglie wavelength lambda=2*pi/k0, the interference fringe
    spacing on a screen a distance L past the wall is `lambda*L/
    slit_separation` in the small-angle (paraxial) limit -- valid near
    the pattern's center, where the fringe position is close to linear
    in the diffraction angle; it grows away from center as the exact
    path-difference condition (m*lambda = d*sin(theta)) departs from
    that linear approximation. See Validation.ipynb (Phase 6)."""
    x = grid.coords[axis]
    y = grid.coords[slit_axis]
    in_wall_x = (x >= x_min) & (x <= x_max)
    in_slit = (np.abs(y - slit_separation / 2) <= slit_width / 2) | (np.abs(y + slit_separation / 2) <= slit_width / 2)
    return np.where(in_wall_x & ~in_slit, V0, 0.0)


def absorbing_boundary(grid, width, eta, order=3, axes=None):
    """Complex absorbing potential (CAP): a real, non-negative ramp W(r)
    that grows over a layer of thickness `width` at the domain edges,
    meant to be subtracted in as an imaginary term when building the
    effective potential passed to propagator.strang_step:

        V_eff = V_real - 1j * W

    propagator.potential_step's existing exp(-i*V_eff*dt) then factors as
    exp(-i*V_real*dt) * exp(-W*dt) -- an ordinary phase times a real decay
    factor -- with no changes needed to propagator.py itself, since it
    never assumed V was real. This is the standard companion to periodic
    (FFT) propagation for open/scattering problems: instead of outgoing
    probability density either reflecting off a hard wall or wrapping
    back around through the FFT's implicit periodicity, it's absorbed
    before it reaches the edge. That absorption necessarily costs norm
    (see observables.norm) -- the lost norm *is* the flux that left.

    `eta` sets the ramp's strength and `order` (2-4 are typical) its
    steepness; both interact with `width` and the wavepacket's momentum
    to determine how much of the incident wave reflects off the CAP
    itself rather than being absorbed -- tuned by direct simulation in
    Validation.ipynb (Phase 3) rather than derived analytically here.
    width=15, eta=15, order=3 (this function's defaults) were found there
    to give reflection/leakage on the order of 1e-9 for a representative
    wavepacket, and are a reasonable starting point elsewhere in this
    project.

    axes: which axes get a layer (default: all of them) -- e.g. leave a
    transverse axis alone in a 2D scattering setup where only the
    propagation axis should absorb.
    """
    if axes is None:
        axes = range(grid.ndim)
    W = np.zeros(grid.shape)
    for axis in axes:
        x = grid.axes[axis]
        L = grid.lengths[axis]
        if grid.boundary == 'periodic':
            edge_lo, edge_hi = -L / 2, L / 2
        else:
            edge_lo, edge_hi = 0.0, L
        dist_lo = np.clip((width - (x - edge_lo)) / width, 0.0, None)
        dist_hi = np.clip((width - (edge_hi - x)) / width, 0.0, None)
        ramp = eta * (dist_lo ** order + dist_hi ** order)
        shape = [1] * grid.ndim
        shape[axis] = len(x)
        W = W + ramp.reshape(shape)
    return W
