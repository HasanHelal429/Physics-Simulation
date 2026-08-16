"""Analytic hydrogen-like orbitals: radial wavefunctions, angular shapes, visualization."""

from math import factorial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_genlaguerre, sph_harm_y


def radial_wavefunction(n, l, r, Z=1):
    """Normalized hydrogen-like R_nl(r) in atomic units (a0=1)."""
    if not (0 <= l < n):
        raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
    r = np.asarray(r, dtype=float)
    rho = 2 * Z * r / n
    norm = np.sqrt((2 * Z / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def radial_density(n, l, r, Z=1):
    """Radial distribution function r^2 R_nl(r)^2."""
    r = np.asarray(r, dtype=float)
    return r**2 * radial_wavefunction(n, l, r, Z) ** 2


def check_normalization(n, l, Z=1, r_max=None, n_points=20000):
    """Numerically integrate r^2 R_nl(r)^2 dr over [0, r_max]; should return ~1."""
    if r_max is None:
        r_max = 3 * n**2 / Z + 20
    r = np.linspace(1e-8, r_max, n_points)
    return np.trapezoid(radial_density(n, l, r, Z), r)


def psi_nlm(n, l, m, r, theta, phi, Z=1):
    """Complex L_z-eigenstate R_nl(r)*Y_l^m; theta=polar in [0,pi], phi=azimuthal in [0,2pi]."""
    return radial_wavefunction(n, l, r, Z) * sph_harm_y(l, m, theta, phi)


def real_spherical_harmonic(l, m, theta, phi):
    """Real orbital-shape harmonic: cosine-lobe for m>0, sine-lobe for m<0, Y_l^0 for m=0."""
    if m == 0:
        return np.real(sph_harm_y(l, 0, theta, phi))
    Y = sph_harm_y(l, abs(m), theta, phi)
    return np.sqrt(2) * (-1) ** m * (np.real(Y) if m > 0 else np.imag(Y))


def real_orbital(n, l, m, r, theta, phi, Z=1, R_func=None):
    """Real hydrogen-like orbital (classic s/p/d lobe shapes), used for visualization.

    R_func, if given, overrides the analytic radial_wavefunction(n, l, r, Z)
    with any callable r -> R(r) -- e.g. a numeric R_nl(r) from an SCF run,
    swapped in so the same angular-shape/cross-section plotting code works
    for both analytic hydrogenic and self-consistent orbitals.
    """
    R = R_func(r) if R_func is not None else radial_wavefunction(n, l, r, Z)
    return R * real_spherical_harmonic(l, m, theta, phi)


def probability_density(psi):
    return np.abs(psi) ** 2


def plot_angular_shape(l, m, ax=None, n_theta=120, n_phi=120, cmap="RdYlBu"):
    """3D surface of a real orbital's angular lobe shape, colored by sign of Y."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)

    Y = real_spherical_harmonic(l, m, THETA, PHI)
    R = np.abs(Y)
    X = R * np.sin(THETA) * np.cos(PHI)
    Ycoord = R * np.sin(THETA) * np.sin(PHI)
    Zcoord = R * np.cos(THETA)

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    norm = plt.Normalize(Y.min(), Y.max())
    ax.plot_surface(X, Ycoord, Zcoord, facecolors=mpl.colormaps[cmap](norm(Y)), rstride=1, cstride=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Angular shape: l={l}, m={m}")
    return ax


def plot_radial_density(n, l, Z=1, r_max=None, n_points=2000, ax=None, R_func=None, label=None):
    """Line plot of the radial distribution function r^2 R_nl(r)^2.

    R_func, if given, overrides the analytic radial_wavefunction (see real_orbital).
    """
    if r_max is None:
        r_max = 3 * n**2 / Z + 20
    r = np.linspace(1e-8, r_max, n_points)

    if ax is None:
        _, ax = plt.subplots()
    R = R_func(r) if R_func is not None else radial_wavefunction(n, l, r, Z)
    ax.plot(r, r**2 * R**2, label=label)
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel(r"$r^2 R_{nl}(r)^2$")
    ax.set_title(f"Radial distribution: n={n}, l={l}, Z={Z}")
    return ax


def plot_orbital_density_slice(n, l, m, Z=1, extent=None, n_points=300, ax=None, cmap="inferno", R_func=None):
    """imshow heatmap of |real_orbital|^2 on the y=0 (x-z) plane.

    R_func, if given, overrides the analytic radial_wavefunction (see real_orbital).
    """
    if extent is None:
        extent = 3 * n**2 / Z + 20
    x = np.linspace(-extent, extent, n_points)
    z = np.linspace(-extent, extent, n_points)
    X, Zc = np.meshgrid(x, z)

    r = np.sqrt(X**2 + Zc**2)
    theta = np.arccos(np.divide(Zc, r, out=np.zeros_like(Zc), where=r > 0))
    phi = np.arctan2(np.zeros_like(X), X)  # y=0 plane -> azimuthal angle is 0 or pi

    density = real_orbital(n, l, m, r, theta, phi, Z, R_func=R_func) ** 2

    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(density, extent=[-extent, extent, -extent, extent], origin="lower", cmap=cmap)
    ax.set_xlabel("x (Bohr)")
    ax.set_ylabel("z (Bohr)")
    ax.set_title(f"|psi|^2 slice (y=0): n={n}, l={l}, m={m}, Z={Z}")
    plt.colorbar(im, ax=ax)
    return ax
