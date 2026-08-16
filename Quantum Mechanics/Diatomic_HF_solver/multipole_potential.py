"""Two-center Hartree (electron-electron Coulomb) potential (Phase 2), via
the Neumann expansion -- the classic two-center addition theorem for
1/|r-r'| in prolate spheroidal coordinates:

    1/|r-r'| = (2/R) * sum_{l,m} (-1)^m (2l+1) (l-|m|)!/(l+|m|)!
               * P_l^m(mu_<) * Q_l^m(mu_>) * P_l^m(nu) * P_l^m(nu')
               * exp(im(phi-phi'))

Every density this project ever builds is cylindrically symmetric (no
phi-dependence -- true for any diatomic ground-state electron density, by
the symmetry of the two-center problem about its own axis), so integrating
the source density over phi' kills every m!=0 term (they integrate to
zero over a full period) and only the m=0 (ordinary Legendre P_l, Q_l)
series survives. P_l(mu_<)*Q_l(mu_>) plays the role that 1/r_> played in
the atomic solver's shell-theorem Hartree potential (itself just this
expansion's l=0 term, generalized to a series in l).

Q_l here is the branch that decays as mu^-(l+1) as mu -> infinity (the
physically correct "outside" Green's-function solution) -- built from its
own P_l/Q_l share the same three-term recursion (it comes from the
Legendre ODE, not the specific solution), so only the seed values differ:
P_0=1, P_1=mu vs. Q_0=arctanh(1/mu), Q_1=mu*Q_0-1. Implemented directly
here (not via scipy.special) to sidestep any branch/normalization
ambiguity for the mu>1 domain.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


def legendre_P(l_max, x):
    """P_0(x)..P_{l_max}(x), vectorized over x. Shape (l_max+1, len(x))."""
    x = np.atleast_1d(x).astype(float)
    P = np.empty((l_max + 1, len(x)))
    P[0] = 1.0
    if l_max >= 1:
        P[1] = x
    for l in range(1, l_max):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
    return P


def legendre_Q(l_max, mu):
    """Q_0(mu)..Q_{l_max}(mu) for mu>1. Shape (l_max+1, len(mu))."""
    mu = np.atleast_1d(mu).astype(float)
    Q = np.empty((l_max + 1, len(mu)))
    Q[0] = np.arctanh(1 / mu)
    if l_max >= 1:
        Q[1] = mu * Q[0] - 1
    for l in range(1, l_max):
        Q[l + 1] = ((2 * l + 1) * mu * Q[l] - l * Q[l - 1]) / (l + 1)
    return Q


def _trapz_weights(x):
    """Vector w such that sum(f*w) ~= integral(f, x) (trapezoidal rule),
    for arbitrary (possibly nonuniform) x -- lets an l-indexed family of
    nu-integrals be done as one matrix multiply instead of a Python loop."""
    w = np.zeros(len(x))
    d = np.diff(x)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def hartree_potential_multipole(mu, nu, rho, R, l_max=8):
    """Two-center Hartree potential on the (mu, nu) grid, truncated at
    order l_max. rho is a (len(mu), len(nu)) cylindrically-symmetric
    density array, normalized so that
    integral(rho * (R/2)^3*(mu^2-nu^2), dmu dnu dphi) = N (total electron
    count) -- the two-center analog of the atomic solver's
    integral(rho*4*pi*r^2, dr) = N convention.

    Derivation: integrating rho(r')/|r-r'| over the source coordinates,
    using the Neumann expansion above and the full volume element
    (R/2)^3*(mu'^2-nu'^2) dmu' dnu' dphi', and defining the l-th "moment"

        rho_l(mu') = integral_{-1}^{1} rho(mu',nu') (mu'^2-nu'^2) P_l(nu') dnu'

    gives

        V_H(mu,nu) = (pi*R^2/2) * sum_l (2l+1) P_l(nu)
                     * [ Q_l(mu) * integral_1^mu P_l(mu')rho_l(mu') dmu'
                       + P_l(mu) * integral_mu^mu_max Q_l(mu')rho_l(mu') dmu' ]

    -- the two-center analog of the atomic Hartree potential's
    Q(r)/r + (S(r_max)-S(r)) enclosed/outside split, now one such split
    per multipole order l.
    """
    N_mu, N_nu = len(mu), len(nu)
    MU, NU = np.meshgrid(mu, nu, indexing="ij")

    P_nu = legendre_P(l_max, nu)  # (l_max+1, N_nu)
    P_mu = legendre_P(l_max, mu)  # (l_max+1, N_mu)
    Q_mu = legendre_Q(l_max, mu)  # (l_max+1, N_mu)

    w_nu = _trapz_weights(nu)
    weighted = rho * (MU**2 - NU**2)  # (N_mu, N_nu)
    rho_l = weighted @ (P_nu * w_nu[None, :]).T  # (N_mu, l_max+1)

    V_H = np.zeros((N_mu, N_nu))
    for l in range(l_max + 1):
        f_A = P_mu[l] * rho_l[:, l]
        f_B = Q_mu[l] * rho_l[:, l]
        A_l = cumulative_trapezoid(f_A, mu, initial=0.0)  # integral_1^mu
        total_B = trapezoid(f_B, mu)
        B_l = total_B - cumulative_trapezoid(f_B, mu, initial=0.0)  # integral_mu^mu_max
        term = Q_mu[l][:, None] * A_l[:, None] + P_mu[l][:, None] * B_l[:, None]
        V_H += (2 * l + 1) * term * P_nu[l][None, :]
    return (np.pi * R**2 / 2) * V_H
