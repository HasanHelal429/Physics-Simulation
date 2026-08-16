"""
Position and reciprocal-space grids for the split-operator TDSE solver
(propagator.py). Two boundary conventions, chosen once per Grid and shared
by every downstream module:

  'periodic' -- domain wraps around; the kinetic propagator's spectral
    transform is the ordinary FFT, and the free-particle plane-wave modes
    e^{ikx} are exact eigenfunctions of -d^2/dx^2 with eigenvalue k^2.
  'box'      -- hard walls (psi=0 exactly at the domain edges, an infinite
    square well); the spectral transform is the type-I discrete sine
    transform (DST-I), whose basis functions sin(n*pi*x/L) are exactly the
    Dirichlet eigenfunctions of -d^2/dx^2 with eigenvalue (n*pi/L)^2.

Both cases reduce to the same downstream code in propagator.py: transform
-> multiply by exp(-i*k^2*dt/2) -> inverse transform. `k_axes` is what
makes that code boundary-agnostic -- it always means "the sqrt of the
-d^2/dx^2 eigenvalue per axis", whether that comes from an FFT frequency
or a sine-series mode index.
"""

from collections import namedtuple

import numpy as np

Grid = namedtuple('Grid', ['ndim', 'shape', 'boundary', 'lengths', 'dx', 'axes', 'coords', 'k_axes'])


def make_grid(lengths, ns, boundary='periodic'):
    """Build an ndim-generic grid. `lengths`/`ns` are per-axis physical
    length and point count (one entry per dimension, same order).

    'periodic': axis samples span (-L/2, L/2) with spacing L/n -- the
      standard FFT convention (no duplicated point at both ends).
    'box': axis samples are the N *interior* points of [0, L]; the walls
      at x=0 and x=L are themselves excluded (psi is exactly 0 there, not
      represented in the array) -- the standard DST-I convention, spacing
      L/(n+1).
    """
    if boundary not in ('periodic', 'box'):
        raise ValueError(f"unknown boundary {boundary!r}")
    lengths = tuple(lengths)
    ns = tuple(ns)
    ndim = len(lengths)
    if len(ns) != ndim:
        raise ValueError("lengths and ns must have the same number of axes")

    axes, dx, k_axes = [], [], []
    for L, n in zip(lengths, ns):
        if boundary == 'periodic':
            x, h = np.linspace(-L / 2, L / 2, n, endpoint=False, retstep=True)
            k = 2 * np.pi * np.fft.fftfreq(n, d=h)
        else:
            h = L / (n + 1)
            x = np.arange(1, n + 1) * h
            k = np.arange(1, n + 1) * np.pi / L
        axes.append(x)
        dx.append(h)
        k_axes.append(k)

    coords = np.meshgrid(*axes, indexing='ij')
    shape = coords[0].shape

    return Grid(ndim=ndim, shape=shape, boundary=boundary, lengths=lengths,
                dx=tuple(dx), axes=tuple(axes), coords=tuple(coords), k_axes=tuple(k_axes))


def gaussian_wavepacket(grid, center, sigma, k0=None):
    """Minimum-uncertainty Gaussian wavepacket on `grid`:

        psi(r) = prod_i (2*pi*sigma_i^2)^{-1/4} exp(i*k0_i*(x_i-center_i))
                        * exp(-(x_i-center_i)^2 / (4*sigma_i^2))

    so |psi(r)|^2 is a (continuum-)normalized Gaussian with standard
    deviation sigma_i along axis i -- the standard convention (e.g.
    Griffiths) whose free-particle time evolution has the closed form
    sigma(t) = sigma_i * sqrt(1 + (t/(2*sigma_i^2))^2) used to validate
    the propagator in Validation.ipynb.

    center, sigma, k0: scalars, or length-ndim sequences (one value per
    axis). k0 defaults to 0 (packet at rest).
    """
    center = np.broadcast_to(center, (grid.ndim,))
    sigma = np.broadcast_to(sigma, (grid.ndim,))
    k0 = np.zeros(grid.ndim) if k0 is None else np.broadcast_to(k0, (grid.ndim,))
    psi = np.ones(grid.shape, dtype=complex)
    for x, x0, s, k in zip(grid.coords, center, sigma, k0):
        psi = psi * (2 * np.pi * s ** 2) ** (-0.25) * np.exp(1j * k * (x - x0)) * np.exp(-(x - x0) ** 2 / (4 * s ** 2))
    return psi
