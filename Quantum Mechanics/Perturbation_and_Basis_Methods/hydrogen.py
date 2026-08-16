"""
Hydrogen atom bound-state wavefunctions psi_nlm(r,theta,phi) = R_nl(r) *
Y_l^m(theta,phi), atomic units (hbar=m=e=1, a0=1), Z=1 unless stated.

Uses `scipy.special.sph_harm_y(l, m, theta, phi)` -- the *current* scipy
API (the older `scipy.special.sph_harm` has been removed entirely; this
was confirmed directly rather than assumed while building this module).
Its docstring is explicit about convention: theta is the polar
(colatitude) angle in [0, pi], phi is the azimuthal angle in [0, 2*pi] --
the opposite labeling from some textbooks, so it's worth restating here
since getting it backwards silently gives wrong answers for any m != 0.
"""

import math

import numpy as np
from scipy import special


def energy(n, Z=1.0):
    """E_n = -Z^2/(2n^2), atomic units."""
    return -Z ** 2 / (2 * n ** 2)


def radial_wavefunction(n, l, r, Z=1.0):
    """R_nl(r), normalized so that integral R_nl(r)^2 r^2 dr = 1."""
    a = 1.0 / Z
    rho = 2 * r / (n * a)
    norm = np.sqrt((2 / (n * a)) ** 3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    laguerre = special.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return norm * np.exp(-rho / 2) * rho ** l * laguerre


def angular_wavefunction(l, m, theta, phi):
    """Y_l^m(theta, phi), complex, normalized so integral |Y|^2 dOmega = 1."""
    return special.sph_harm_y(l, m, theta, phi)


def wavefunction(n, l, m, r, theta, phi, Z=1.0):
    """Full psi_nlm(r, theta, phi) = R_nl(r) * Y_l^m(theta, phi)."""
    return radial_wavefunction(n, l, r, Z) * angular_wavefunction(l, m, theta, phi)


def real_p_orbitals(n, r, theta, phi, Z=1.0):
    """Real combinations of the l=1 (n>=2) orbitals: (p_x, p_y, p_z), the
    standard real spherical harmonic combinations
        p_z = Y_1^0
        p_x = (Y_1^-1 - Y_1^1)/sqrt(2)
        p_y = i*(Y_1^-1 + Y_1^1)/sqrt(2)
    each still carrying R_n1(r). These come out real up to floating-point
    residue (checked in Validation.ipynb) since they're eigenstates of a
    real Hamiltonian built from a real linear combination of complex
    conjugate-paired spherical harmonics."""
    R = radial_wavefunction(n, 1, r, Z)
    y_m1 = angular_wavefunction(1, -1, theta, phi)
    y_0 = angular_wavefunction(1, 0, theta, phi)
    y_p1 = angular_wavefunction(1, 1, theta, phi)
    p_z = R * y_0
    p_x = R * (y_m1 - y_p1) / np.sqrt(2)
    p_y = R * 1j * (y_m1 + y_p1) / np.sqrt(2)
    return p_x, p_y, p_z
