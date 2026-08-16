"""Self-consistent field driver (Phase 5) plus the total-energy expression (Phase 6)
that its convergence check needs. Ties together the radial eigensolver (Phase 2),
effective potential (Phase 3), and fixed-configuration shell filling (Phase 4).
"""

import warnings

import numpy as np
from scipy.integrate import trapezoid

import atomic_scf
import potentials as pot
import shells


def _initial_density(r, Z):
    """Crude seed density: an exponential profile normalized to N=Z electrons."""
    rho = np.exp(-2 * Z * r)
    N = trapezoid(4 * np.pi * r**2 * rho, r)
    return rho * (Z / N)


def _density_from_occupied(r, occupied):
    """rho(r) = (1/(4*pi*r**2)) * sum_nl N_nl*u_nl(r)**2, from (n, l, N_nl, eps_nl, u_nl) tuples."""
    total = np.zeros_like(r)
    for _, _, N_nl, _, u_nl in occupied:
        total += N_nl * u_nl**2
    return total / (4 * np.pi * r**2)


def compute_total_energy(r, rho, V_H, V_x, occupied):
    """E_total = sum_nl(N_nl*eps_nl) - E_H - (1/3)*E_x.

    Orbital eigenvalues double-count electron-electron interaction: summing
    N_nl*eps_nl over occupied shells counts the Hartree term twice and the
    (nonlinear) exchange term 4/3 times, so both must be corrected out. See
    HF_Solver_Plan.md Phase 6 for the derivation (Hartree: simple 1/2 factor;
    exchange: 1/3 factor from Euler's theorem on the rho**(4/3) functional).
    """
    sum_eps = sum(N_nl * eps_nl for _, _, N_nl, eps_nl, _ in occupied)
    E_H = 0.5 * trapezoid(V_H * rho * 4 * np.pi * r**2, r)
    E_x = 0.75 * trapezoid(V_x * rho * 4 * np.pi * r**2, r)
    return sum_eps - E_H - E_x / 3


def run_scf(
    Z,
    alpha=pot.ALPHA_SCHWARZ,
    max_iter=200,
    mix_beta=0.3,
    tol_E=1e-6,
    tol_n=1e-5,
    grid=None,
    verbose=False,
    record_history=False,
):
    """Run the Xa self-consistent field loop for atomic number Z to convergence.

    Occupations come from shells.ground_state_configuration(Z) and are held
    fixed for the whole run. Converges when both |dE_total| < tol_E and the
    integrated density change < tol_n hold for 2 consecutive iterations.

    If record_history=True, a per-iteration snapshot (rho, V, orbital_energies,
    E_total, dE, dn) is appended to the returned "snapshots" list -- purely for
    visualizing/animating the SCF loop itself (Phase 7); the physics is
    unaffected either way.
    """
    r, h = grid if grid is not None else atomic_scf.default_grid(Z)
    config = shells.ground_state_configuration(Z)

    by_l = {}
    for (n, l), N_nl in config.items():
        by_l.setdefault(l, {})[n] = N_nl
    n_states_per_l = {l: max(ns) - l for l, ns in by_l.items()}

    rho = _initial_density(r, Z)
    E_prev = None
    converged_streak = 0
    history = []
    snapshots = []
    dE = dn = np.inf

    for it in range(1, max_iter + 1):
        V = pot.effective_potential(r, rho, Z, alpha)
        V_H = pot.hartree_potential(r, rho)
        V_x = pot.slater_exchange_potential(r, rho, alpha)

        occupied = []
        for l, ns in by_l.items():
            energies, u = atomic_scf.solve_radial_channel(r, l, V, h, n_states=n_states_per_l[l])
            for n, N_nl in ns.items():
                idx = n - l - 1
                occupied.append((n, l, N_nl, energies[idx], u[:, idx]))

        rho_new = _density_from_occupied(r, occupied)
        E_total = compute_total_energy(r, rho, V_H, V_x, occupied)

        dn = trapezoid(np.abs(rho_new - rho) * 4 * np.pi * r**2, r)
        dE = abs(E_total - E_prev) if E_prev is not None else np.inf
        history.append(E_total)
        if record_history:
            snapshots.append(
                {
                    "iteration": it,
                    "rho": rho.copy(),
                    "V": V.copy(),
                    "orbital_energies": {(n, l): eps for n, l, _, eps, _ in occupied},
                    "E_total": E_total,
                    "dE": dE,
                    "dn": dn,
                }
            )
        if verbose:
            print(f"iter {it:3d}: E_total = {E_total: .8f}  dE={dE:.2e}  dn={dn:.2e}")

        if dE < tol_E and dn < tol_n:
            converged_streak += 1
            if converged_streak >= 2:
                rho = rho_new
                break
        else:
            converged_streak = 0

        rho = (1 - mix_beta) * rho + mix_beta * rho_new
        E_prev = E_total
    else:
        warnings.warn(f"run_scf: Z={Z} did not converge within {max_iter} iterations (dE={dE:.2e}, dn={dn:.2e})")

    orbitals = {(n, l): u / r for n, l, _, _, u in occupied}
    orbital_energies = {(n, l): eps for n, l, _, eps, _ in occupied}

    return {
        "Z": Z,
        "r": r,
        "h": h,
        "rho": rho,
        "V": V,
        "V_H": V_H,
        "V_x": V_x,
        "config": config,
        "orbitals": orbitals,
        "orbital_energies": orbital_energies,
        "E_total": E_total,
        "iterations": it,
        "history": history,
        "snapshots": snapshots,
    }
