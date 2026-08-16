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


def pz81_correlation(rho):
    """Perdew-Zunger (1981) parametrization of the LDA correlation energy per
    electron eps_c(rs) (a fit to Ceperley-Alder quantum Monte Carlo data for
    the unpolarized/paramagnetic electron gas) and its corresponding potential
    V_c(rho) = eps_c + rho*d(eps_c)/d(rho) = eps_c - (rs/3)*d(eps_c)/d(rs),
    via the usual Wigner-Seitz radius rs = (3/(4*pi*rho))**(1/3).

    This is the piece Slater/X-alpha (see slater_exchange_potential) never
    had: Xalpha is a *local exchange* approximation only, with no
    correlation term at all, and its alpha is an empirically-tuned knob
    rather than derived from the density functional itself. Pairing
    ALPHA_LDA exchange with this correlation functional gives genuine
    Kohn-Sham LDA -- both pieces now fixed, parameter-free functionals of
    rho, not a tunable local-exchange model.

    Two-branch fit, continuous (not necessarily smooth) across rs=1:
      rs < 1  (high density): eps_c = A*ln(rs) + B + C*rs*ln(rs) + D*rs
      rs >= 1 (low density):  eps_c = gamma / (1 + beta1*sqrt(rs) + beta2*rs)
    """
    # Far out on the radial grid the density underflows to exact float64 0.0
    # (found via a real crash: rs = (.../rho)**(1/3) -> inf there, then the
    # low-density branch's V_c formula hits a literal 0*inf -> NaN, which
    # poisons the effective-potential matrix and makes the sparse LU
    # factorization inside eigsh fail with "Factor is exactly singular").
    # Physically rho=0 just means eps_c, V_c -> 0 (both branches' true
    # mathematical limit as rs -> infinity) -- clipping to a floor far below
    # any physically meaningful density avoids the literal 0*inf without
    # changing the physics anywhere it matters.
    rho = np.maximum(rho, 1e-300)
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    eps_c = np.empty_like(rho, dtype=float)
    V_c = np.empty_like(rho, dtype=float)

    high = rs < 1
    low = ~high

    A, B, C, D = 0.0311, -0.0480, 0.0020, -0.0116
    rs_h = rs[high]
    ln_rs_h = np.log(rs_h)
    eps_c[high] = A * ln_rs_h + B + C * rs_h * ln_rs_h + D * rs_h
    V_c[high] = A * ln_rs_h + (B - A / 3) + (2 / 3) * C * rs_h * ln_rs_h + ((2 * D - C) / 3) * rs_h

    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    rs_l = rs[low]
    sqrt_rs_l = np.sqrt(rs_l)
    denom = 1 + beta1 * sqrt_rs_l + beta2 * rs_l
    eps_c[low] = gamma / denom
    V_c[low] = eps_c[low] * (1 + (7 / 6) * beta1 * sqrt_rs_l + (4 / 3) * beta2 * rs_l) / denom

    return eps_c, V_c


def lda_xc_potential(r, rho):
    """Genuine Kohn-Sham LDA exchange-correlation potential: exact-LDA Slater
    exchange (alpha=ALPHA_LDA, not the empirical Xalpha alpha) plus PZ81
    correlation. Returns (V_xc, eps_c) -- eps_c is also needed by the total
    -energy correlation term, so it's returned alongside rather than
    recomputed."""
    V_x = slater_exchange_potential(r, rho, ALPHA_LDA)
    eps_c, V_c = pz81_correlation(rho)
    return V_x + V_c, eps_c


def effective_potential_lda(r, rho, Z):
    """Total central-field potential under genuine Kohn-Sham LDA: -Z/r +
    Hartree + (exact-LDA exchange + PZ81 correlation). Kept as a separate
    function from effective_potential (Xalpha) rather than a mode flag on
    it, so existing Xalpha-based code/notebooks are untouched."""
    V_xc, _ = lda_xc_potential(r, rho)
    return -Z / r + hartree_potential(r, rho) + V_xc
