"""Prolate spheroidal coordinates for the two-center (diatomic) problem (Phase 1).

    mu = (r_A + r_B) / R   in [1, infinity)
    nu = (r_A - r_B) / R   in [-1, 1]
    phi = azimuthal angle about the internuclear axis, in [0, 2*pi)

Nucleus A sits at z=-R/2, nucleus B at z=+R/2 on the shared axis. This is
the exact coordinate system in which the bare two-center Coulomb potential
-Z_A/r_A - Z_B/r_B separates cleanly (see diatomic_scf.py's Hamiltonian
derivation) -- the diatomic analog of the atomic solver's radial log-grid
in r.

mu=1 is the line segment between the two nuclei; nu=+-1 is the axis outside
that segment. The two nuclear positions themselves are the corners
(mu=1, nu=+1) [nucleus B] and (mu=1, nu=-1) [nucleus A].
"""

import numpy as np


def to_cartesian(mu, nu, phi, R):
    """(mu, nu, phi) -> (x, y, z), nucleus A at z=-R/2, nucleus B at z=+R/2."""
    rho = (R / 2) * np.sqrt(np.clip((mu**2 - 1) * (1 - nu**2), 0, None))
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = (R / 2) * mu * nu
    return x, y, z


def r_A_r_B(mu, nu, R):
    """Electron-nucleus distances from (mu, nu): r_A = R(mu+nu)/2, r_B = R(mu-nu)/2."""
    return R * (mu + nu) / 2, R * (mu - nu) / 2


def mu_grid(mu_max, N, s_min=1e-3):
    """Log-spaced grid in s = mu - 1, from s_min to mu_max - 1 -- clusters
    points near mu=1 (the nuclear-cusp line) and spreads them out over the
    smooth long-range tail, the diatomic analog of the atomic solver's
    log-radial grid (built for the same reason: the wavefunction has a Kato
    cusp at each nucleus, and a uniform grid needs impractically many points
    to resolve it -- confirmed empirically: a uniform grid's H2+ energy was
    still off by several percent even at ~300k grid points). Requires
    diatomic_scf's finite-volume discretization (not a plain uniform
    finite-difference stencil), which handles nonuniform spacing correctly."""
    s = np.exp(np.linspace(np.log(s_min), np.log(mu_max - 1), N))
    return 1 + s


def nu_grid(N, eps=1e-6):
    """Grid on [-1+eps, 1-eps], clustered near both endpoints via
    nu = sin(pi/2 * xi) on a uniform xi -- the Kato cusp at each nucleus
    sits at the corners (mu=1, nu=+-1), so, like mu_grid, nu needs extra
    resolution near +-1 (the region near nu=0, far from both nuclei, is
    smooth and doesn't). This is a standard Chebyshev-extrema-style
    clustering (sin has vanishing slope at xi=+-1, packing points there for
    uniformly-spaced xi), not a log-grid, since nu -- unlike mu -- has no
    long tail to also resolve, just two symmetric endpoint singularities."""
    xi = np.linspace(-1 + eps, 1 - eps, N)
    return np.sin(np.pi / 2 * xi)
