"""
Bound vibrational (and rotational) states of a diatomic on a Born-Oppenheimer
potential curve E(R): the radial nuclear Schrodinger equation

    [ -1/(2 mu) d^2/dR^2 + hbar^2 J(J+1)/(2 mu R^2) + E(R) ] chi(R) = E_vib chi(R)

solved on a uniform R-grid with `TDSE_Solver.stationary_states` (the
finite-difference eigensolver, now mass-aware) -- the electronic-structure
half of the repo joined to the wavepacket-dynamics half by a thin layer.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "TDSE_Solver"))

import grid as tdse_grid          # noqa: E402
import stationary_states as ss    # noqa: E402
import observables as obs         # noqa: E402


def radial_grid(R_min, R_max, n):
    """1D box grid on [R_min, R_max] (Dirichlet walls -- chi -> 0 at both
    ends, appropriate for bound states well inside the range). Returns a
    TDSE_Solver Grid whose single axis is the internuclear coordinate."""
    length = R_max - R_min
    g = tdse_grid.make_grid([length], [n], boundary="box")
    # shift the axis so it starts at R_min (make_grid's box axis starts at L/(n+1))
    shifted_axes = (g.axes[0] + R_min,)
    shifted_coords = (g.coords[0] + R_min,)
    return g._replace(axes=shifted_axes, coords=shifted_coords)


def vibrational_levels(grid, V, mu, J=0, k=12):
    """Lowest `k` eigenpairs of the reduced-mass radial Hamiltonian at fixed
    rotational quantum number `J`. Returns (energies, states) with `states`
    a list of 1D arrays, each continuum-normalized so
    integral |chi(R)|^2 dR = 1.

    The centrifugal term hbar^2 J(J+1)/(2 mu R^2) is added straight onto the
    potential -- the parametric-J ("rigid-rotor-per-vibrational-state")
    treatment, matching the plan's scope.
    """
    R = grid.axes[0]
    V_eff = np.asarray(V, float).copy()
    if J:
        V_eff = V_eff + J * (J + 1) / (2.0 * mu * R ** 2)
    energies, states = ss.lowest_states(grid, V_eff, k=k, mass=mu)
    norm_states = []
    for s in states:
        s = np.asarray(s, float)
        s = s / np.sqrt(np.sum(s ** 2) * grid.dx[0])
        norm_states.append(s)
    return energies, norm_states


def expectation(grid, chi, f_of_R):
    """<chi| f(R) |chi> for a normalized radial state."""
    R = grid.axes[0]
    return float(np.sum(np.abs(chi) ** 2 * f_of_R(R)) * grid.dx[0])


def rotational_constant(grid, chi_v0, mu):
    """B_v = <1/R^2>_v / (2 mu) for the vibrational state `chi_v0`."""
    return expectation(grid, chi_v0, lambda R: 1.0 / R ** 2) / (2.0 * mu)


def vibration_rotation_coupling(grid, V, mu, v_max=3):
    """alpha_e from a linear fit of B_v vs (v + 1/2):  B_v = B_e - alpha_e (v+1/2).
    Returns (B_e, alpha_e, B_v_array)."""
    energies, states = vibrational_levels(grid, V, mu, J=0, k=v_max + 2)
    Bv = np.array([rotational_constant(grid, states[v], mu) for v in range(v_max + 1)])
    x = np.arange(v_max + 1) + 0.5
    slope, inter = np.polyfit(x, Bv, 1)
    return inter, -slope, Bv


def dunham_fit(energies, n_levels=6):
    """Fit E_v = T + we (v+1/2) - wexe (v+1/2)^2 + weye (v+1/2)^3 to the
    lowest `n_levels`. Returns (T, we, wexe, weye)."""
    energies = np.asarray(energies, float)
    n = min(n_levels, len(energies))
    x = np.arange(n) + 0.5
    A = np.vstack([np.ones_like(x), x, -x ** 2, x ** 3]).T
    coef, *_ = np.linalg.lstsq(A, energies[:n], rcond=None)
    return tuple(coef)


def franck_condon_matrix(grid, states_lower, states_upper):
    """FC factors ``|<chi^upper_{v'} | chi^lower_v>|^2`` as a
    ``(len(upper), len(lower))`` array. Both state lists must live on the
    same `grid`."""
    dx = grid.dx[0]
    U = np.array(states_upper)
    Lm = np.array(states_lower)
    overlap = (U @ Lm.T) * dx
    return np.abs(overlap) ** 2
