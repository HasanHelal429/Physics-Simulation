"""Radial finite-difference eigensolver on a logarithmic grid (Phase 2)."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def log_grid(r_min, r_max, n_points):
    """Logarithmic radial grid r_i = r_min * exp(i*h); returns (r, h)."""
    h = np.log(r_max / r_min) / (n_points - 1)
    r = r_min * np.exp(np.arange(n_points) * h)
    return r, h


def default_grid(Z, r_max=150, n_points=4000):
    """Sane log-grid defaults for atomic number Z (core orbitals scale as 1/Z).

    r_min = 1e-5/Z keeps the finite-difference matrices well away from the
    float64 dynamic-range/conditioning issues that showed up when r_min was
    chosen independent of Z (worst-case eigenvalue error stays ~1e-5 to
    1e-4 across Z=1..100, see Radial SCF Validation.ipynb).
    """
    return log_grid(1e-5 / Z, r_max, n_points)


def build_radial_matrices(r, l, V, h):
    """Generalized eigenproblem H w = E*M w for the radial equation on a log grid.

    Substitution u(r)=r*R(r), then u(r)=sqrt(r)*w(x) with x=ln(r) removes the
    first-derivative term and turns the centrifugal term into (l+1/2)^2 (a
    correct artifact of the transform, not l(l+1)). Left as a *generalized*
    eigenproblem (H, M both natural-scale) rather than rescaled to a standard
    one by dividing through by r: that rescaling makes matrix entries span
    ~1/r_min^2 in dynamic range, which overflows float64 precision for the
    small r_min heavy atoms need to resolve their core orbitals.
    """
    main_diag_H = 2 / h**2 + (l + 0.5) ** 2 + 2 * r**2 * V
    off_diag_H = -np.ones(len(r) - 1) / h**2
    M_diag = 2 * r**2
    return main_diag_H, off_diag_H, M_diag


def solve_radial_channel(r, l, V, h, n_states=8, sigma=None):
    """Lowest n_states eigenpairs (energies, u_nl(r)) for angular momentum l.

    Solved as a sparse generalized eigenproblem via ARPACK shift-invert,
    which factorizes the natural-scale tridiagonal (H - sigma*M) directly
    instead of forming an ill-conditioned 1/r-rescaled matrix.
    u_nl is normalized so that trapezoid(u_nl**2, r) == 1 for each state.

    ARPACK's Lanczos iteration occasionally fails to converge for small
    n_states against the crude, far-from-self-consistent V_eff seen on early
    SCF iterations (seen in practice for a p-channel during Cu's SCF run).
    Retried with a progressively larger Krylov subspace (ncv) and a slightly
    perturbed shift before giving up.
    """
    main_diag_H, off_diag_H, M_diag = build_radial_matrices(r, l, V, h)
    H = diags([off_diag_H, main_diag_H, off_diag_H], [-1, 0, 1], format="csc")
    M = diags(M_diag, format="csc")
    if sigma is None:
        z_est = -r[0] * V[0]  # near r_min, V(r) ~ -Z/r for any physical effective potential
        sigma = -3 * max(z_est, 1.0) ** 2

    last_error = None
    for attempt in range(4):
        ncv = min(len(r) - 1, max(4 * n_states + 1, 40) * 2**attempt)
        try:
            energies, w = eigsh(H, k=n_states, M=M, sigma=sigma * 1.2**attempt, which="LM", ncv=ncv, maxiter=20000)
            break
        except ArpackNoConvergence as exc:
            last_error = exc
    else:
        raise RuntimeError(f"eigsh failed to converge for l={l}, n_states={n_states} after retries") from last_error

    order = np.argsort(energies)
    energies, w = energies[order], w[:, order]
    u = w * np.sqrt(r)[:, None]
    u = u / np.sqrt(np.trapezoid(u**2, r, axis=0))
    return energies, u


def radial_function_from_u(u, r):
    """R_nl(r) = u_nl(r)/r, given u from solve_radial_channel."""
    return u / r[:, None]
