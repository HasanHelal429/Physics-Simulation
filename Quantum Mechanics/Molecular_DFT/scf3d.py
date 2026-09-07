"""Full 3D SCF loop + total energy (Phase 3+). Same mixing/convergence
shape as HF_solver/scf.py and Diatomic_HF_solver/diatomic_driver.py:
build V_eff from the current density -> diagonalize (via an iterative
eigensolver over an implicit, FFT-applied Hamiltonian -- a 3D grid is far
too large for the earlier solvers' direct/shift-invert sparse approach,
see Molecular_DFT_Plan.md) -> aufbau-fill (from the first iteration only,
then held fixed, same stability rationale as before) -> rebuild density ->
mix -> check convergence.
"""

import os
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
    return_orbitals=False,
):
    """Run the 3D SCF loop to convergence. grid = (x, X, Y, Z, dx, G2)
    from grid3d.make_grid/g_vectors. method is "xalpha" or "lda", same
    two options as the earlier solvers.

    return_orbitals: also return the converged occupied Kohn-Sham orbitals
    (normalized, one 3D array per occupied level) as result["orbitals"]
    (shape (n_occ, *grid)) with occupations result["occ_orbitals"] --
    rt-TDDFT (Quantum Mechanics/TDDFT/) needs them as its initial state.
    """
    if method not in ("xalpha", "lda"):
        raise ValueError(f"method must be 'xalpha' or 'lda', got {method!r}")

    x, X, Y, Z, dx, G2 = grid
    shape = X.shape
    V_nuc = pot.nuclear_potential(X, Y, Z, nuclei)

    # nuclear-nuclear repulsion: zero for a single atom (Phases 3-4, where
    # this was never needed), but essential for anything with more than one
    # nucleus -- found missing via a real bug: H2's electronic-only energy
    # (-1.963 Ha) looked wildly overbound until adding Z_A*Z_B/R (0.714 Ha
    # at R=1.4) brought it in line with Diatomic_HF_solver's cross-check.
    E_nuc_nuc = 0.0
    for i in range(len(nuclei)):
        Z_i, xi, yi, zi, _ = nuclei[i]
        for j in range(i + 1, len(nuclei)):
            Z_j, xj, yj, zj, _ = nuclei[j]
            E_nuc_nuc += Z_i * Z_j / np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)

    # crude seed density: a normalized Gaussian blob at the origin
    R2 = X**2 + Y**2 + Z**2
    rho = np.exp(-R2)
    rho *= N_electrons / _integrate(rho, dx)

    occ = None
    E_prev = None
    converged_streak = 0
    history = []

    for it in range(1, max_iter + 1):
        V_eff, parts = pot.ks_potential(rho, V_nuc, G2, method, alpha)
        V_H, V_x, eps_c, V_c = parts["V_H"], parts["V_x"], parts["eps_c"], parts["V_c"]

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
        E_total = sum_eps - E_H - E_x / 3 + E_nuc_nuc
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

    result = {
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

    if return_orbitals:
        orbs, occ_orbs = [], []
        for i, occ_i in enumerate(occ):
            if occ_i == 0:
                continue
            psi = vecs[:, i].reshape(shape)
            psi = psi / np.sqrt(_integrate(psi**2, dx))
            orbs.append(psi)
            occ_orbs.append(occ_i)
        result["orbitals"] = np.array(orbs)
        result["occ_orbitals"] = np.array(occ_orbs)

    return result


def scan_pes(Z_A, Z_B, R_values, grid, N_electrons=None, method="lda",
             softening=0.3, csv_path=None, verbose=False, **scf_kw):
    """Born-Oppenheimer potential energy curve on the 3D Cartesian grid:
    place the two nuclei symmetrically about the origin along z, separated by
    R, run the SCF, collect the total energy for each R in `R_values`.

    Returns (R_array, E_array), and (if `csv_path`) writes a two-column CSV --
    the analogue of Diatomic_HF_solver.diatomic_driver.scan_pes, making the
    PES a clean importable product for Nuclear_Dynamics/. `csv_path=True`
    writes media/pes3d_Z<A>_Z<B>_<method>.csv next to this module.
    """
    if N_electrons is None:
        N_electrons = Z_A + Z_B
    R_values = np.asarray(R_values, dtype=float)
    E = np.empty_like(R_values)
    for i, R in enumerate(R_values):
        nuclei = [(Z_A, 0.0, 0.0, -R / 2, softening), (Z_B, 0.0, 0.0, +R / 2, softening)]
        res = run_scf(nuclei, N_electrons, grid, method=method, verbose=False, **scf_kw)
        E[i] = res["E_total"]
        if verbose:
            print(f"  R = {R:6.3f} Bohr   E = {E[i]: .8f} Ha   ({res['iterations']} iters)")
    if csv_path:
        if csv_path is True:
            media = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
            os.makedirs(media, exist_ok=True)
            csv_path = os.path.join(media, f"pes3d_Z{Z_A}_Z{Z_B}_{method}.csv")
        np.savetxt(csv_path, np.column_stack([R_values, E]), delimiter=",",
                   header=f"R_bohr,E_total_ha   (Z_A={Z_A}, Z_B={Z_B}, method={method}, 3D grid)",
                   comments="# ")
    return R_values, E
