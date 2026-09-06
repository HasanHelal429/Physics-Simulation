"""Perturbations / drives for rt-TDDFT.

Phase 3: the instantaneous dipole "delta-kick" -- multiply every orbital by
exp(i k * x_a) at t=0. This boosts the whole electron cloud along axis `a`
with momentum k; the induced dipole response, Fourier-transformed, gives the
full linear absorption spectrum in one propagation (see response.py).

Phase 5: `dipole_field` builds a time-dependent V_ext(r,t) = E(t) * x_a for a
laser pulse (length gauge).
"""

import numpy as np


def dipole_kick(psi, coords, k, axis=0):
    """psi_j(r) -> exp(i k * coord_axis) psi_j(r), in place-safe (returns a
    new array). `coords` = (X, Y, Z); `axis` in {0, 1, 2}."""
    phase = np.exp(1j * k * coords[axis])
    return psi * phase


def sin2_pulse(E0, omega, n_cycles, t0=0.0):
    """A carrier-envelope few-cycle pulse:
        E(t) = E0 * sin^2(pi (t - t0) / T_pulse) * sin(omega (t - t0))
    for t0 <= t <= t0 + T_pulse, else 0, with T_pulse = n_cycles * 2 pi / omega.
    Returns a callable E(t)."""
    T_pulse = n_cycles * 2 * np.pi / omega

    def E(t):
        tau = t - t0
        if tau < 0 or tau > T_pulse:
            return 0.0
        return E0 * np.sin(np.pi * tau / T_pulse) ** 2 * np.sin(omega * tau)

    E.T_pulse = T_pulse
    return E


def dipole_field(E_of_t, coords, axis=0):
    """Wrap a scalar field E(t) into v_ext(t) -> 3D array  E(t) * coord_axis
    (length gauge). Returns a callable suitable for propagate(v_ext_fn=...)."""
    c = coords[axis]

    def v_ext(t):
        e = E_of_t(t)
        return None if e == 0.0 else e * c

    return v_ext
