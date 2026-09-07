"""
Time-dependent nuclear wavepacket dynamics on one or two Born-Oppenheimer
surfaces, built on `TDSE_Solver.propagator` (mass-aware split-operator).

One surface   -- vibrational wavepacket motion, dephasing/revivals,
                 photodissociation with a complex absorbing potential detector.
Two surfaces  -- diabatic V_1(R), V_2(R) and coupling V_12(R); the potential
                 half-step becomes a pointwise 2x2 matrix exponential (closed
                 form, no scipy needed), the kinetic step acts channel-wise.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "TDSE_Solver"))

import propagator as prop         # noqa: E402


# ----------------------------------------------------------------- one surface
def propagate_1surface(grid, V, psi0, mu, dt, n_steps, W=None,
                       observables=None, record_every=1):
    """Split-operator propagation of a single nuclear wavepacket.

    ``W``  optional real non-negative absorbing ramp; ``V_eff = V - i W`` so
           outgoing flux is absorbed (see TDSE_Solver.potentials.absorbing_boundary).
    ``observables``  dict name -> callable(psi) -> value, sampled every
           ``record_every`` steps.

    Returns ``(times, records, psi_final, absorbed_norm)`` where
    ``absorbed_norm`` is 1 - <psi|psi> at the end (the dissociated fraction).
    """
    k2 = prop.kinetic_eigenvalues(grid, mass=mu)
    V_eff = np.asarray(V, complex)
    if W is not None:
        V_eff = V_eff - 1j * np.asarray(W, float)
    psi = np.asarray(psi0, complex).copy()
    dx = grid.dx[0]

    times, records = [], {k: [] for k in (observables or {})}
    for step in range(n_steps + 1):
        if step:
            psi = prop.strang_step(psi, grid, V_eff, dt, k2=k2, mass=mu)
        if step % record_every == 0:
            times.append(step * dt)
            for name, fn in (observables or {}).items():
                records[name].append(fn(psi))
    norm = np.sum(np.abs(psi) ** 2) * dx
    return (np.array(times), {k: np.array(v) for k, v in records.items()},
            psi, 1.0 - norm)


def coherent_vibrational_packet(grid, states, weights, phases=None):
    """Normalized superposition sum_v c_v chi_v of vibrational eigenstates."""
    weights = np.asarray(weights, float)
    if phases is None:
        phases = np.zeros_like(weights)
    psi = np.zeros(grid.shape, complex)
    for c, ph, chi in zip(weights, phases, states):
        psi = psi + c * np.exp(1j * ph) * chi
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dx[0])
    return psi


def expectation_R(grid, psi):
    R = grid.axes[0]
    d = np.abs(psi) ** 2
    return float(np.sum(R * d) / np.sum(d))


def revival_time(omega_e_xe):
    """T_rev = 2 pi / (omega_e x_e) -- the anharmonic full-revival time
    (atomic units)."""
    return 2.0 * np.pi / omega_e_xe


def absorbed_flux_spectrum(grid, V, psi0, mu, dt, n_steps, W, R_detect):
    """Photodissociation analysis. Propagate with the CAP `W` active; once the
    wavepacket has left the potential slope (region ``R > R_detect``, where
    ``V`` is flat) its kinetic-energy content *is* the kinetic-energy release.

    Energy conservation makes this exact: at large R, ``<T> = E_total - V_inf``.
    The KER distribution is read from the momentum distribution of the
    outgoing part of the wavepacket, snapshotted just before it is absorbed.

    Returns ``(ker_grid, P_of_ker, total_absorbed)``.
    """
    k2 = prop.kinetic_eigenvalues(grid, mass=mu)
    W = np.asarray(W, float)
    V_eff = np.asarray(V, complex) - 1j * W
    psi = np.asarray(psi0, complex).copy()
    dx = grid.dx[0]
    R = grid.axes[0]
    k_axis = grid.k_axes[0]
    outgoing = k_axis > 0
    norm0 = np.sum(np.abs(psi) ** 2) * dx

    beyond_mask = R > R_detect
    absorber_on = W > 1e-8
    keep = beyond_mask & (~absorber_on)             # flat V, not yet in the CAP

    ker_accum = np.zeros(np.count_nonzero(outgoing))
    prev_captured = 0.0
    for step in range(1, n_steps + 1):
        psi = prop.strang_step(psi, grid, V_eff, dt, k2=k2, mass=mu)
        seg = np.where(keep, psi, 0.0)
        w_seg = np.sum(np.abs(seg) ** 2) * dx
        if w_seg > 1e-6:
            spec = np.abs(np.fft.fft(seg))[:len(k_axis)][outgoing] ** 2
            s = spec.sum()
            if s > 0:
                # weight this snapshot by how much new norm entered the window
                dw = max(w_seg - prev_captured * 0.5, w_seg * 0.02)
                ker_accum += dw * spec / s
        prev_captured = w_seg
    absorbed = norm0 - np.sum(np.abs(psi) ** 2) * dx
    ker_grid = 0.5 * k_axis[outgoing] ** 2 / mu
    order = np.argsort(ker_grid)
    return ker_grid[order], ker_accum[order], absorbed


# ---------------------------------------------------------------- two surfaces
def _expm_2x2_hermitian(a, b, c_re, c_im, t):
    """exp(-i t H) for H = [[a, c],[c*, b]] pointwise (arrays a,b,c over the
    grid). Closed form via the 2x2 spectral decomposition."""
    avg = 0.5 * (a + b)
    diff = 0.5 * (a - b)
    cabs2 = c_re ** 2 + c_im ** 2
    omega = np.sqrt(diff ** 2 + cabs2)
    omega_safe = np.where(omega > 1e-300, omega, 1.0)
    phase = np.exp(-1j * avg * t)
    cos = np.cos(omega * t)
    sinc = np.where(omega > 1e-300, np.sin(omega * t) / omega_safe, t)
    U00 = phase * (cos - 1j * sinc * diff)
    U11 = phase * (cos + 1j * sinc * diff)
    U01 = phase * (-1j * sinc * (c_re - 1j * c_im))
    U10 = phase * (-1j * sinc * (c_re + 1j * c_im))
    return U00, U01, U10, U11


def propagate_2surface(grid, V1, V2, V12, psi1_0, psi2_0, mu, dt, n_steps,
                       W=None, observables=None, record_every=1):
    """Two-channel split-operator propagation. Potentials real; ``V12`` the
    (real) diabatic coupling. Returns ``(times, records, (psi1, psi2))``."""
    k2 = prop.kinetic_eigenvalues(grid, mass=mu)
    a = np.asarray(V1, float)
    b = np.asarray(V2, float)
    c_re = np.asarray(V12, float) * np.ones_like(a)
    c_im = np.zeros_like(a)
    if W is not None:
        Wc = np.asarray(W, float)
    psi1 = np.asarray(psi1_0, complex).copy()
    psi2 = np.asarray(psi2_0, complex).copy()

    U00h, U01h, U10h, U11h = _expm_2x2_hermitian(a, b, c_re, c_im, dt / 2)

    def vstep(p1, p2):
        q1 = U00h * p1 + U01h * p2
        q2 = U10h * p1 + U11h * p2
        if W is not None:
            decay = np.exp(-Wc * dt / 2)
            q1 *= decay
            q2 *= decay
        return q1, q2

    times, records = [], {k: [] for k in (observables or {})}
    for step in range(n_steps + 1):
        if step:
            psi1, psi2 = vstep(psi1, psi2)
            psi1 = prop.kinetic_step(psi1, grid, dt, k2=k2, mass=mu)
            psi2 = prop.kinetic_step(psi2, grid, dt, k2=k2, mass=mu)
            psi1, psi2 = vstep(psi1, psi2)
        if step % record_every == 0:
            times.append(step * dt)
            for name, fn in (observables or {}).items():
                records[name].append(fn(psi1, psi2))
    return (np.array(times), {k: np.array(v) for k, v in records.items()},
            (psi1, psi2))
