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


def orbital_value_grad_lap(n, l, m, xyz, Z=1):
    """Value, gradient and Laplacian of a *real* hydrogen-like orbital at
    arbitrary 3D points ``xyz`` (shape ``(..., 3)``, relative to the
    nucleus). Returns ``(val, grad, lap)`` with shapes ``(...)``,
    ``(..., 3)``, ``(...)``.

    Unlike the grid helpers above (which evaluate a prebuilt meshgrid and
    return values only), this is what a real-space many-body method needs:
    the Slater matrix wants values, the drift velocity wants
    ``grad psi / psi``, and the kinetic local energy wants
    ``lap psi / psi``. Analytic for the s and p orbitals that the small
    all-electron QMC systems (H, He, Li, Be, H2, LiH) actually use; a
    high-order finite-difference fallback covers any other ``(n, l, m)``.
    """
    xyz = np.asarray(xyz, dtype=float)
    r = np.sqrt(np.sum(xyz ** 2, axis=-1))
    r_safe = np.where(r > 1e-12, r, 1e-12)
    rhat = xyz / r_safe[..., None]

    def from_radial(f, df, d2f):
        """phi = f(r): grad = f' rhat, lap = f'' + 2 f'/r."""
        val = f(r)
        grad = df(r)[..., None] * rhat
        lap = d2f(r) + 2.0 * df(r) / r_safe
        return val, grad, lap

    def from_p(g, dg, d2g, axis):
        """phi = x_axis * g(r): grad_i = delta_{i,axis} g + x_axis g' rhat_i,
        lap = x_axis (g'' + 4 g'/r)."""
        xa = xyz[..., axis]
        val = xa * g(r)
        grad = xa[..., None] * (dg(r)[..., None] * rhat)
        e = np.zeros_like(xyz)
        e[..., axis] = 1.0
        grad = grad + e * g(r)[..., None]
        lap = xa * (d2g(r) + 4.0 * dg(r) / r_safe)
        return val, grad, lap

    a = Z / n  # effective decay scale of the exponential
    if (n, l) == (1, 0):
        norm = np.sqrt(Z ** 3 / np.pi)
        f = lambda rr: norm * np.exp(-a * rr)
        df = lambda rr: -a * norm * np.exp(-a * rr)
        d2f = lambda rr: a * a * norm * np.exp(-a * rr)
        return from_radial(f, df, d2f)

    if (n, l) == (2, 0):
        norm = np.sqrt(Z ** 3 / (8 * np.pi)) / np.sqrt(2)
        # R_20 ~ (1 - Z r / 2) exp(-Z r / 2)
        b = Z / 2.0
        f = lambda rr: norm * (1.0 - b * rr) * np.exp(-b * rr)
        df = lambda rr: norm * (-b * np.exp(-b * rr) + (1.0 - b * rr) * (-b) * np.exp(-b * rr))
        d2f = lambda rr: norm * np.exp(-b * rr) * (b * b * (1.0 - b * rr) + 2.0 * b * b)
        return from_radial(f, df, d2f)

    if (n, l) == (2, 1):
        axis = {1: 0, -1: 1, 0: 2}[m]           # 2px, 2py, 2pz
        norm = np.sqrt(Z ** 5 / (32 * np.pi))
        b = Z / 2.0
        g = lambda rr: norm * np.exp(-b * rr)
        dg = lambda rr: -b * norm * np.exp(-b * rr)
        d2g = lambda rr: b * b * norm * np.exp(-b * rr)
        return from_p(g, dg, d2g, axis)

    # ---- finite-difference fallback for anything else ----
    h = 1e-4
    base = psi_nlm_real_value(n, l, m, xyz, Z)
    grad = np.zeros_like(xyz)
    lap = np.zeros(r.shape)
    for ax in range(3):
        step = np.zeros(3)
        step[ax] = h
        fp = psi_nlm_real_value(n, l, m, xyz + step, Z)
        fm = psi_nlm_real_value(n, l, m, xyz - step, Z)
        grad[..., ax] = (fp - fm) / (2 * h)
        lap += (fp - 2 * base + fm) / h ** 2
    return base, grad, lap


def psi_nlm_real_value(n, l, m, xyz, Z=1):
    """Real hydrogen-like orbital value at Cartesian points ``xyz`` (relative
    to the nucleus). Helper for orbital_value_grad_lap's fallback."""
    xyz = np.asarray(xyz, float)
    r = np.sqrt(np.sum(xyz ** 2, axis=-1))
    r_safe = np.where(r > 1e-12, r, 1e-12)
    theta = np.arccos(np.clip(xyz[..., 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    R = radial_wavefunction(n, l, r, Z)
    return R * real_spherical_harmonic(l, m, theta, phi)


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
