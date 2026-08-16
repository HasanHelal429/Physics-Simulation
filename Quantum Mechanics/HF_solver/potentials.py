"""Effective potential: Hartree (electron-electron) term + Slater/X-alpha exchange (Phase 3)."""

import numpy as np
from scipy.integrate import cumulative_trapezoid


def hartree_potential(r, rho):
    """Electron-electron Coulomb potential via the shell theorem (O(N), not O(N^2)).

    rho(r) is a number density normalized so that integral(rho*4*pi*r**2, dr) = N
    (total electron count). Splits each charge shell's contribution into the
    "enclosed" part (acts like a point charge at the origin) and the "outside"
    part (contributes its own 1/r', constant inside that shell):

        Q(r) = integral_0^r rho(r') 4*pi*r'**2 dr'      (charge enclosed within r)
        S(r) = integral_0^r rho(r') 4*pi*r'  dr'
        V_H(r) = Q(r)/r + (S(r_max) - S(r))

    Q(r_max) should equal N -- a useful sanity check for callers.
    """
    Q = cumulative_trapezoid(4 * np.pi * r**2 * rho, r, initial=0.0)
    S = cumulative_trapezoid(4 * np.pi * r * rho, r, initial=0.0)
    return Q / r + (S[-1] - S)


# Common alpha presets for Slater/X-alpha exchange.
ALPHA_LDA = 2 / 3  # theoretically exact local-density-approximation value
ALPHA_SLATER = 1.0  # classic Slater exchange
ALPHA_SCHWARZ = 0.7  # Schwarz-optimized empirical compromise (default)


def slater_exchange_potential(r, rho, alpha=ALPHA_SCHWARZ):
    """Local Slater/X-alpha exchange potential: V_x(r) = -3*alpha*(3*rho(r)/(8*pi))**(1/3).

    alpha is a tunable approximation to true non-local HF exchange, not a fixed
    constant -- see ALPHA_LDA/ALPHA_SLATER/ALPHA_SCHWARZ presets above.
    """
    return -3 * alpha * (3 * rho / (8 * np.pi)) ** (1 / 3)


def effective_potential(r, rho, Z, alpha=ALPHA_SCHWARZ):
    """Total central-field potential: -Z/r + Hartree + Slater/X-alpha exchange."""
    return -Z / r + hartree_potential(r, rho) + slater_exchange_potential(r, rho, alpha)
