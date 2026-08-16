"""Full 3D SCF loop + total energy (Phase 3+). Same mixing/convergence
shape as HF_solver/scf.py and Diatomic_HF_solver/diatomic_driver.py:
build V_eff from the current density -> diagonalize (via an iterative
eigensolver over an implicit, FFT-applied Hamiltonian -- a 3D grid is far
too large for the earlier solvers' direct/shift-invert sparse approach,
see Molecular_DFT_Plan.md) -> aufbau-fill (from the first iteration only,
then held fixed, same stability rationale as before) -> rebuild density ->
mix -> check convergence.
"""

import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

import fft_ops
import potentials3d as pot


def hamiltonian_operator(V, G2, shape):
    """H = -0.5*nabla^2 + V as a matrix-free scipy LinearOperator (kinetic
    via FFT, potential via pointwise multiply) -- no explicit matrix is
    ever formed, since a 3D grid's N^3 points make even a sparse matrix
    too large to factorize the way the earlier solvers' shift-invert
    eigensolvers did."""
    N_total = int(np.prod(shape))

    def matvec(psi_flat):
        psi = psi_flat.reshape(shape)
        Hpsi = fft_ops.apply_kinetic(psi, G2) + V * psi
        return Hpsi.ravel()

    return LinearOperator((N_total, N_total), matvec=matvec, dtype=np.float64)


def solve_lowest_states(V, G2, shape, n_states):
    """Lowest n_states eigenpairs of H = -0.5*nabla^2 + V via Lanczos
    (`which="SA"`, smallest algebraic) directly on the implicit
    LinearOperator -- no shift-invert (that would need factorizing
    H-sigma*M, impossible without an explicit matrix at this size)."""
    H_op = hamiltonian_operator(V, G2, shape)
    N_total = int(np.prod(shape))
    ncv = min(N_total, max(6 * n_states + 1, 40))
    energies, vecs = eigsh(H_op, k=n_states, which="SA", ncv=ncv)
    order = np.argsort(energies)
    return energies[order], vecs[:, order]


def aufbau_fill(energies, N_electrons):
    """Simple aufbau: 2 electrons per orbital (spin up+down), ascending
    energy, no orbital-angular-momentum bookkeeping needed here (unlike
    the atomic/diatomic solvers -- a general 3D grid has no exploitable
    exact symmetry left to label orbitals by)."""
    occ = np.zeros(len(energies))
    remaining = N_electrons
    for i in range(len(energies)):
        if remaining <= 0:
            break
        occ[i] = min(2, remaining)
        remaining -= occ[i]
    if remaining > 0:
        raise ValueError("aufbau_fill: ran out of states -- request more n_states")
    return occ


def _integrate(f, dx):
    return np.sum(f) * dx**3


def run_scf(
    nuclei,
    N_electrons,
    grid,
    method="lda",
    alpha=pot.ALPHA_SCHWARZ,
    n_states=4,
    max_iter=60,
    mix_beta=0.3,
    tol_E=1e-5,
    tol_n=1e-3,
    verbose=False,
):
    """Run the 3D SCF loop to convergence. grid = (x, X, Y, Z, dx, G2)
    from grid3d.make_grid/g_vectors. method is "xalpha" or "lda", same
    two options as the earlier solvers.
    """
    if method not in ("xalpha", "lda"):
        raise ValueError(f"method must be 'xalpha' or 'lda', got {method!r}")

    x, X, Y, Z, dx, G2 = grid
    shape = X.shape
    V_nuc = pot.nuclear_potential(X, Y, Z, nuclei)

    # crude seed density: a normalized Gaussian blob at the origin
    R2 = X**2 + Y**2 + Z**2
    rho = np.exp(-R2)
    rho *= N_electrons / _integrate(rho, dx)

    occ = None
    E_prev = None
    converged_streak = 0
    history = []

    for it in range(1, max_iter + 1):
        V_H = fft_ops.solve_poisson(rho, G2)
        if method == "lda":
            V_x = pot.slater_exchange_potential(rho, pot.ALPHA_LDA)
            eps_c, V_c = pot.pz81_correlation(rho)
            V_xc = V_x + V_c
        else:
            V_x = pot.slater_exchange_potential(rho, alpha)
            V_xc, eps_c, V_c = V_x, None, None
        V_eff = V_nuc + V_H + V_xc

        energies, vecs = solve_lowest_states(V_eff, G2, shape, n_states)

        if occ is None:
            occ = aufbau_fill(energies, N_electrons)

        rho_new = np.zeros(shape)
        for i, occ_i in enumerate(occ):
            if occ_i == 0:
                continue
            psi = vecs[:, i].reshape(shape)
            psi = psi / np.sqrt(_integrate(psi**2, dx))  # normalize: integral(|psi|^2 d^3r) = 1
            rho_new += occ_i * psi**2

        sum_eps = float(np.sum(occ * energies))
        E_H = 0.5 * _integrate(V_H * rho, dx)
        E_x = 0.75 * _integrate(V_x * rho, dx)
        E_total = sum_eps - E_H - E_x / 3
        if eps_c is not None:
            E_c = _integrate(eps_c * rho, dx)
            E_total += E_c - _integrate(V_c * rho, dx)

        N_check = _integrate(rho, dx)
        dn = _integrate(np.abs(rho_new - rho), dx)
        dE = abs(E_total - E_prev) if E_prev is not None else np.inf
        history.append(E_total)
        if verbose:
            print(f"iter {it:3d}: E_total={E_total: .6f}  dE={dE:.2e}  dn={dn:.2e}  N_check={N_check:.6f}")

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
        warnings.warn(f"run_scf: did not converge within {max_iter} iterations (dE={dE:.2e}, dn={dn:.2e})")

    return {
        "method": method,
        "grid": grid,
        "rho": rho,
        "V_eff": V_eff,
        "energies": energies,
        "occ": occ,
        "E_total": E_total,
        "iterations": it,
        "history": history,
        "N_check": N_check,
    }
