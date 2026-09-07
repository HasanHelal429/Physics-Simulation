"""Full diatomic SCF loop + total energy (Phase 4). Same mixing/convergence
shape as HF_solver/scf.py's run_scf, generalized to the two-center
(mu, nu) problem: build V_eff from the current density -> diagonalize each
lambda-channel -> aufbau-fill (held fixed after the first iteration, same
stability rationale as the atomic solver -- re-determining occupations
every iteration risked oscillating between configurations near a level
crossing) -> rebuild density -> mix -> check convergence.

Slater/Xalpha exchange and PZ81 LDA correlation are copied from
HF_solver/potentials.py rather than imported cross-folder, matching this
repo's convention of self-contained project folders (see
Diatomic_HF_Solver_Plan.md's file-layout note).
"""

import os
import warnings

import numpy as np
from scipy.integrate import trapezoid

import diatomic_scf as dsc
import molecular_shells as ms
import multipole_potential as mp

ALPHA_LDA = 2 / 3
ALPHA_SCHWARZ = 0.7


def slater_exchange_potential(rho, alpha=ALPHA_SCHWARZ):
    """V_x(rho) = -3*alpha*(3*rho/(8*pi))**(1/3) -- purely local, so this
    ports over unchanged from the atomic solver regardless of geometry."""
    return -3 * alpha * (3 * rho / (8 * np.pi)) ** (1 / 3)


def pz81_correlation(rho):
    """Perdew-Zunger (1981) LDA correlation energy density and potential
    (copied from HF_solver/potentials.py -- see there for the derivation
    and the rs=1 branch-continuity/zero-density-clipping notes)."""
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


def _integrate_3d(f, mu, nu, R):
    """integral(f * (R/2)^3*(mu^2-nu^2) dmu dnu dphi) for a cylindrically
    symmetric f(mu,nu) -- the phi-integral is trivial (f doesn't depend on
    it) and just contributes a factor of 2*pi."""
    MU, NU = np.meshgrid(mu, nu, indexing="ij")
    vol = (R / 2) ** 3 * (MU**2 - NU**2)
    return 2 * np.pi * trapezoid(trapezoid(f * vol, nu, axis=1), mu)


def _normalize_eigvec(w, M, R):
    """Rescale a raw eigsh eigenvector so integral(|psi|^2 d^3r) = 1, using
    the identity integral(f^2*(R/2)^3(mu^2-nu^2) dmu dnu)*2*pi
    = (pi*R/2) * (w^T M w) (M is the mass matrix already built by
    diatomic_scf.build_two_center_matrices, so this needs no separate
    quadrature-weight computation)."""
    norm2 = (np.pi * R / 2) * (w @ (M @ w))
    return w / np.sqrt(norm2)


def _density_from_occupied(mu, nu, R, occupied):
    """rho(mu,nu) = sum_occupied occ_i * f_i(mu,nu)^2, with each f_i
    normalized per _normalize_eigvec -- matches multipole_potential's
    integral(rho*(R/2)^3(mu^2-nu^2), dmu dnu dphi) = N convention (derived
    in this module's docstring / Diatomic_HF_Solver_Plan.md)."""
    N_mu, N_nu = len(mu), len(nu)
    rho = np.zeros((N_mu, N_nu))
    for lam, idx, energy, occ, f in occupied:
        rho += occ * f.reshape(N_mu, N_nu) ** 2
    return rho


def _initial_density(mu, nu, R, Z_A, Z_B):
    """Crude seed density: a normalized isotropic-ish blob near the bond
    midpoint (mu small), scaled to N=Z_A+Z_B electrons -- doesn't need to
    be good, just physically reasonable enough for the first V_eff."""
    MU, NU = np.meshgrid(mu, nu, indexing="ij")
    rho = np.exp(-1.5 * (MU - 1))
    N = _integrate_3d(rho, mu, nu, R)
    return rho * ((Z_A + Z_B) / N)


def run_scf(
    Z_A,
    Z_B,
    R,
    N_electrons=None,
    method="xalpha",
    alpha=ALPHA_SCHWARZ,
    lambda_max=2,
    n_states_per_lambda=4,
    hartree_l_max=4,
    mu=None,
    nu=None,
    max_iter=100,
    mix_beta=0.3,
    tol_E=1e-6,
    tol_n=1e-4,
    verbose=False,
):
    """Run the diatomic self-consistent field loop to convergence.

    N_electrons defaults to Z_A+Z_B (the neutral molecule). method is
    "xalpha" (tunable Slater local exchange only, alpha applies) or "lda"
    (exact-LDA exchange + PZ81 correlation, parameter-free, alpha ignored)
    -- same two options as HF_solver/scf.py's run_scf.

    hartree_l_max defaults to 4, deliberately more conservative than
    multipole_potential's own l_max=8 validation default. Found via a real
    SCF divergence (H2, method="xalpha"): the first ~8 iterations converged
    smoothly, tracking toward the right ballpark -- then V_H (which must be
    positive everywhere for a physical density) went negative and blew up
    exponentially over the next few iterations (-0.06 -> -1.2 -> -177 ->
    -2.6e6), taking the whole SCF run with it. Root cause: as the density
    evolves away from the crude initial guess toward the true, more
    sharply-peaked-near-the-nuclei bonding orbital, its higher-l Legendre
    moments grow, and representing those accurately needs more mu/nu
    resolution than this run's grid had -- l_max=8 pushed past what the
    grid could stably support; l_max=4 (confirmed by direct comparison,
    same grid) kept V_H positive and stable throughout, with the excited
    -state channel energies no longer drifting either. l_max and grid
    resolution need to be refined together for a given molecule/grid, same
    conclusion as Diatomic_SCF_Validation.ipynb's Phase 2 check.

    Occupations (which (lambda, level-index) pairs get how many electrons)
    are aufbau-filled from the *first* iteration's orbital energies and
    then held fixed for the rest of the run -- same stability rationale as
    the atomic solver (re-filling every iteration risks oscillating
    between near-degenerate configurations rather than converging).
    """
    if method not in ("xalpha", "lda"):
        raise ValueError(f"method must be 'xalpha' or 'lda', got {method!r}")
    if N_electrons is None:
        N_electrons = Z_A + Z_B

    import prolate_coords as pc

    if mu is None:
        mu = pc.mu_grid(mu_max=30, N=250, s_min=1e-3)
    if nu is None:
        nu = pc.nu_grid(N=140)

    rho = _initial_density(mu, nu, R, Z_A, Z_B)
    occupied_spec = None  # list of (lambda, idx) once aufbau-determined
    E_prev = None
    converged_streak = 0
    history = []
    # per-lambda-channel shift-invert sigma, re-derived from the PREVIOUS
    # iteration's own lowest energy in that channel each time (mirrors
    # HF_solver/atomic_scf.py's solve_radial_channel, which re-estimates
    # sigma from the current V at every call rather than a fixed guess) --
    # a fixed sigma (based only on Z_A,Z_B, ignoring how V_eff evolves as
    # Hartree+exchange grow in) was found to make eigsh's shift-invert
    # silently converge to spurious eigenvalues clustered near that stale
    # sigma once V_eff moved far enough from the bare two-center problem
    # (multiple returned "energies" collapsing toward the same wrong
    # value, then the whole SCF run diverging catastrophically) --
    # confirmed by tracing the raw per-channel energies iteration by
    # iteration, not an isolated ARPACK exception the existing retry loop
    # would have caught.
    sigma_by_lambda = {}

    for it in range(1, max_iter + 1):
        V_H = mp.hartree_potential_multipole(mu, nu, rho, R, l_max=hartree_l_max)
        if method == "lda":
            V_x = slater_exchange_potential(rho, ALPHA_LDA)
            eps_c, V_c = pz81_correlation(rho)
            V_xc = V_x + V_c
        else:
            V_x = slater_exchange_potential(rho, alpha)
            V_xc, eps_c, V_c = V_x, None, None
        V_eff = V_H + V_xc

        # solve every lambda-channel, collect all candidate (lambda, idx, energy)
        # plus the M (mass) matrix each channel needs for eigenvector normalization
        channel_results = {}
        candidates = []
        for lam in range(lambda_max + 1):
            sigma = sigma_by_lambda.get(lam)
            energies, W = dsc.solve_two_center_channel(
                mu, nu, lam, Z_A, Z_B, R, V_extra=V_eff, n_states=n_states_per_lambda, sigma=sigma
            )
            sigma_by_lambda[lam] = energies[0] - 0.1  # bias slightly below, to stay a lower bound
            _, M_lam = dsc.build_two_center_matrices(mu, nu, lam, Z_A, Z_B, R, V_eff)
            channel_results[lam] = (energies, W, M_lam)
            for idx, e in enumerate(energies):
                candidates.append((lam, idx, e))

        if occupied_spec is None:
            filled = ms.aufbau_fill(candidates, N_electrons)
            occupied_spec = [(lam, idx) for lam, idx, _, _ in filled]

        # rebuild the occupied-orbital list at THIS iteration's energies/eigenvectors
        # (occupations/spec fixed, but energies and wavefunctions still update every
        # iteration as V_eff evolves -- same as the atomic solver's fixed-config SCF)
        occupied = []
        cand_lookup = {(lam, idx): e for lam, idx, e in candidates}
        occ_lookup = {(lam, idx): occ for lam, idx, _, occ in filled}
        for lam, idx in occupied_spec:
            energies, W, M_lam = channel_results[lam]
            w = _normalize_eigvec(W[:, idx], M_lam, R)
            occupied.append((lam, idx, cand_lookup[(lam, idx)], occ_lookup[(lam, idx)], w))

        rho_new = _density_from_occupied(mu, nu, R, occupied)
        sum_eps = sum(occ * e for _, _, e, occ, _ in occupied)
        E_H = 0.5 * _integrate_3d(V_H * rho, mu, nu, R)
        E_x = 0.75 * _integrate_3d(V_x * rho, mu, nu, R)
        E_elec = sum_eps - E_H - E_x / 3
        if eps_c is not None:
            E_c = _integrate_3d(eps_c * rho, mu, nu, R)
            E_elec += E_c - _integrate_3d(V_c * rho, mu, nu, R)
        E_total = E_elec + Z_A * Z_B / R

        N_check = _integrate_3d(rho, mu, nu, R)
        dn = _integrate_3d(np.abs(rho_new - rho), mu, nu, R)
        dE = abs(E_total - E_prev) if E_prev is not None else np.inf
        history.append(E_total)
        if verbose:
            print(f"iter {it:3d}: E_total={E_total: .8f}  dE={dE:.2e}  dn={dn:.2e}  N_check={N_check:.6f}")

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
        "Z_A": Z_A,
        "Z_B": Z_B,
        "R": R,
        "method": method,
        "mu": mu,
        "nu": nu,
        "rho": rho,
        "V_H": V_H,
        "V_x": V_x,
        "occupied": occupied,
        "filled": filled,
        "E_total": E_total,
        "E_elec": E_elec,
        "iterations": it,
        "history": history,
        "N_check": N_check,
    }


def scan_pes(Z_A, Z_B, R_values, method="lda", csv_path=None, verbose=False, **scf_kw):
    """Born-Oppenheimer potential energy curve: run the diatomic SCF at each
    fixed nuclear separation in `R_values` and collect the total energy.

    Returns (R_array, E_array). If `csv_path` is given (or set to True, which
    picks media/pes_Z<A>_Z<B>_<method>.csv next to this module) the curve is
    also written as a two-column CSV -- so a PES becomes a clean importable
    product for Nuclear_Dynamics/ rather than something buried in a notebook.
    No physics change: this is exactly the bond-scan loop the validation
    notebooks already run inline.
    """
    R_values = np.asarray(R_values, dtype=float)
    E = np.empty_like(R_values)
    for i, R in enumerate(R_values):
        res = run_scf(Z_A, Z_B, float(R), method=method, verbose=False, **scf_kw)
        E[i] = res["E_total"]
        if verbose:
            print(f"  R = {R:6.3f} Bohr   E = {E[i]: .8f} Ha   ({res['iterations']} iters)")

    if csv_path:
        if csv_path is True:
            media = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
            os.makedirs(media, exist_ok=True)
            csv_path = os.path.join(media, f"pes_Z{Z_A}_Z{Z_B}_{method}.csv")
        header = f"R_bohr,E_total_ha   (Z_A={Z_A}, Z_B={Z_B}, method={method})"
        np.savetxt(csv_path, np.column_stack([R_values, E]), delimiter=",",
                   header=header, comments="# ")
    return R_values, E
