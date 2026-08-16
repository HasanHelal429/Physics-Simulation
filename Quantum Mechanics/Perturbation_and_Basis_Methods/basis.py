"""
Analytic eigenbases (infinite square well, harmonic oscillator) for
representing and evolving a 1D wavefunction as a superposition of
stationary states -- the natural complement to `TDSE_Solver`'s
grid/spectral propagator: instead of stepping psi forward on a grid, a
wavefunction is expanded once in a known eigenbasis and then evolved by
just multiplying each coefficient by its own phase.

Both eigenbases here are sampled on a shared position grid and packaged
into one `BasisSet`, so `project`/`reconstruct`/`evolve` are written once
and work for either basis (or any future one built the same way).
"""

import math
from collections import namedtuple

import numpy as np
from scipy import special
from scipy.integrate import simpson

BasisSet = namedtuple('BasisSet', ['x', 'eigenfunctions', 'energies'])


def box_basis(x, L, num_modes):
    """Infinite square well on [0, L] (atomic units, hbar=m=1):
    phi_n(x) = sqrt(2/L)*sin(n*pi*x/L), E_n = (n*pi/L)^2/2, n=1..num_modes.
    Same energy formula independently validated in
    `TDSE_Solver/stationary_states.py` (Phase 4)."""
    n = np.arange(1, num_modes + 1)
    eigenfunctions = np.sqrt(2 / L) * np.sin(np.outer(n, np.pi * x / L))
    energies = (n * np.pi / L) ** 2 / 2
    return BasisSet(x=x, eigenfunctions=eigenfunctions, energies=energies)


def harmonic_basis(x, omega, num_modes):
    """Quantum harmonic oscillator (atomic units): phi_n(x) built from
    Hermite polynomials, E_n = omega*(n+1/2), n=0..num_modes-1. Same
    construction (and normalization convention) as
    `TDSE_Solver/potentials.harmonic_eigenstate`, just sampled on a plain
    1D array here instead of an ndim grid."""
    n = np.arange(num_modes)
    xi = np.sqrt(omega) * x
    eigenfunctions = np.empty((num_modes, len(x)))
    for k in n:
        k_int = int(k)  # plain Python int: 2**k_int * factorial(k_int) must
                         # use arbitrary-precision int arithmetic -- numpy's
                         # fixed-width int64 silently overflows around k=20
        herm = special.hermite(k_int, monic=False)(xi)
        norm = (omega / np.pi) ** 0.25 / np.sqrt(float(2 ** k_int * math.factorial(k_int)))
        eigenfunctions[k] = norm * herm * np.exp(-xi ** 2 / 2)
    energies = omega * (n + 0.5)
    return BasisSet(x=x, eigenfunctions=eigenfunctions, energies=energies)


def project(basis, psi_samples):
    """Expansion coefficients c_n = <phi_n|psi> = integral(phi_n* psi dx),
    for every mode at once via Simpson's rule (vectorized over modes --
    one call, not a per-mode loop)."""
    integrand = basis.eigenfunctions * psi_samples[np.newaxis, :]
    return simpson(integrand, x=basis.x, axis=1)


def reconstruct(basis, c):
    """psi(x) = sum_n c_n * phi_n(x) -- a single matrix-vector product."""
    return c @ basis.eigenfunctions


def evolve(basis, c0, t):
    """Coefficients at time t: c_n(t) = c_n(0)*exp(-i*E_n*t) -- the
    correct sign (the legacy notebooks this project replaces used
    exp(+i*E*t), which runs the dynamics backwards). `t` may be a scalar
    or an array; if an array, returns shape (len(t), num_modes)."""
    phase = np.exp(-1j * np.outer(np.atleast_1d(t), basis.energies))
    c_t = c0[np.newaxis, :] * phase
    return c_t[0] if np.isscalar(t) else c_t
