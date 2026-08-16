"""Blackbody radiation: Planck's law derived from cavity mode density x
Bose-Einstein photon occupation (not just typed in from memory), plus a
Monte Carlo photon-frequency sampler and a planetary radiative-
equilibrium solver built on top of it.
"""
import numpy as np
from scipy import constants as const

h = const.h
c = const.c
k = const.k
sigma_sb = const.sigma  # Stefan-Boltzmann constant -- used only to validate against, never assumed


def spectral_radiance(nu, T):
    """Planck's law: spectral radiance B(nu, T) [W / (m^2 sr Hz)].

    Derived as (cavity mode density) x (mean photon energy per mode):

        rho(nu)   = 8*pi*nu^2 / c^3                    -- E&M standing-wave modes per unit volume per unit
                                                            frequency in a cavity, both polarizations
        <E>(nu,T) = h*nu / (exp(h*nu/(k*T)) - 1)        -- Bose-Einstein mode occupation x quantum energy

    Converting the resulting cavity energy density per unit frequency
    into the radiance escaping through a small hole in the cavity wall
    picks up a factor c/(4*pi) (the standard blackbody-radiance
    relation), giving B(nu,T) = c/(4*pi) * rho(nu) * <E>(nu,T), which
    simplifies to the familiar 2*h*nu^3/c^2 / (exp(h*nu/kT)-1) form.
    """
    x = h * nu / (k * T)
    with np.errstate(over='ignore'):
        denom = np.expm1(x)
    return (2 * h * nu**3 / c**2) / denom


def spectral_radiance_wavelength(wavelength, T):
    """Planck's law by wavelength, B_lambda(lambda,T) [W/(m^2 sr m)] --
    NOT simply spectral_radiance with nu replaced by c/lambda.
    Converting a per-unit-frequency density to a per-unit-wavelength one
    needs the |dnu/dlambda| = c/lambda^2 Jacobian, since B(nu)dnu must
    equal B_lambda(lambda)dlambda with dnu = -c/lambda^2 dlambda.
    """
    x = h * c / (wavelength * k * T)
    with np.errstate(over='ignore'):
        denom = np.expm1(x)
    return (2 * h * c**2 / wavelength**5) / denom


def rayleigh_jeans(nu, T):
    """The classical (pre-quantum) prediction: each of the same
    rho(nu) = 8*pi*nu^2/c^3 cavity modes carries k*T of energy
    (equipartition) instead of the quantum h*nu/(exp(h*nu/kT)-1) mean.
    This is exactly spectral_radiance's low-frequency (h*nu << k*T)
    limit, and the formula whose unbounded nu^2 growth is the
    ultraviolet catastrophe that Planck's law fixes.
    """
    return (2 * nu**2 * k * T) / c**2


def planck_cdf_sample(T, n, nu_max_factor=30, n_grid=20000, rng=None):
    """Monte Carlo sample `n` photon frequencies from the (normalized)
    Planck spectral-radiance distribution at temperature T, by inverse-
    CDF sampling: the Planck distribution's CDF has no closed form, so
    this builds a fine numerical CDF once via cumulative trapezoidal
    integration, then inverts it by interpolation for each uniform-
    random draw -- cheap after the one-time O(n_grid) setup.
    `nu_max_factor` sets the frequency cutoff as a multiple of the
    thermal frequency scale k*T/h, wide enough to capture the tail.
    """
    if rng is None:
        rng = np.random.default_rng()
    nu_thermal = k * T / h
    nu_grid = np.linspace(1e-6 * nu_thermal, nu_max_factor * nu_thermal, n_grid)
    pdf = spectral_radiance(nu_grid, T)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(nu_grid))])
    cdf /= cdf[-1]
    u = rng.random(n)
    return np.interp(u, cdf, nu_grid)


def equilibrium_temperature(L_star, distance, albedo=0.0, greenhouse=0.0):
    """Planetary equilibrium temperature from radiative flux balance:
    absorbed stellar power = emitted blackbody power.

    Incident flux at the planet: F = L_star / (4*pi*distance^2).
    Absorbed power = F*(1-albedo) * (pi*R^2 cross-section); emitted
    power = sigma*T^4 * (4*pi*R^2 full sphere) -- R cancels, giving the
    standard T_bare^4 = F*(1-albedo) / (4*sigma).

    `greenhouse` (0 <= greenhouse < 1) is a toy single-parameter
    correction for partial atmospheric opacity to outgoing infrared:
    T^4 = T_bare^4 / (1 - greenhouse) (greenhouse=0 reduces exactly to
    the bare-rock result).
    """
    flux = L_star / (4 * np.pi * distance**2)
    T_bare4 = flux * (1 - albedo) / (4 * sigma_sb)
    return (T_bare4 / (1 - greenhouse))**0.25
