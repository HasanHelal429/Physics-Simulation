"""2D (mu, nu) finite-difference eigensolver for the two-center problem (Phase 1).

Derivation (atomic units, hbar=m=e=1). The electronic Schrodinger equation
-0.5*nabla^2 psi + V*psi = E*psi, with psi = f(mu)*g(nu)*exp(i*lambda*phi),
and the prolate-spheroidal Laplacian

    nabla^2 psi = [4/(R^2*(mu^2-nu^2))] * { L_mu[f]*g + f*L_nu[g]
                    - lambda^2*C(mu,nu)*f*g }

    L_mu[f] = d/dmu[(mu^2-1) df/dmu],   L_nu[g] = d/dnu[(1-nu^2) dg/dnu]
    C(mu,nu) = (mu^2-nu^2) / ((mu^2-1)*(1-nu^2))

becomes, after multiplying through by Q = R^2*(mu^2-nu^2)/2 (which cancels
the coordinate-singular 1/(mu^2-nu^2) prefactor and turns the bare two
-center nuclear potential -Z_A/r_A - Z_B/r_B into the simple linear-in
-(mu,nu) form below, since it is exactly separable in these coordinates):

    H*psi := -L_mu[psi] - L_nu[psi] + lambda^2*C*psi
             - R*[(Z_A+Z_B)*mu - (Z_A-Z_B)*nu]*psi + Q*V_extra*psi
           = Q*E*psi =: M*E*psi

a generalized eigenproblem H w = E M w, same shape as the atomic solver's
(H, M) pencil (see HF_solver/atomic_scf.py) -- just a 2D operator instead
of a 1D radial one. V_extra is an optional additional potential (Hartree +
exchange-correlation, added from Phase 2 onward); None for this phase's
bare two-center problem.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _sl_operator(x, p_func):
    """Symmetric finite-volume discretization of the self-adjoint operator
    -d/dx[p(x) df/dx] on a grid x that need not be uniform (required for
    mu_grid/nu_grid's cusp-clustered spacing -- a plain uniform stencil
    turned out to need impractically many points to resolve the
    wavefunction's Kato cusp at each nucleus; confirmed empirically: still
    several percent off at ~300k uniform grid points).

    Standard control-volume construction: integrate -d/dx[p df/dx] over
    each point's cell (bounded by the midpoints to its neighbors), giving

        (stiffness f)_i = coupling_{i-1}*(f_i-f_{i-1}) + coupling_i*(f_i-f_{i+1})
        coupling_i = p(midpoint_i) / (x_{i+1}-x_i)

    -- exactly symmetric (coupling_i is shared identically by rows i and
    i+1) on any grid, uniform or not. Also returns each point's
    control-volume width (half the distance to each neighbor, summed) --
    needed so that any diagonal (mass/potential) term assembled alongside
    this operator is normalized on the same footing (see
    build_two_center_matrices), which a naive nonuniform extension of the
    old fixed-h^2-denominator stencil got wrong.

    Returns (main_diag, off_diag, weights); main_diag/off_diag represent
    the operator directly as -L[f] (positive-semi-definite stiffness, not
    L[f] itself).
    """
    x = np.asarray(x, dtype=float)
    h = np.diff(x)
    x_mid = (x[:-1] + x[1:]) / 2
    coupling = p_func(x_mid) / h

    main_diag = np.zeros(len(x))
    main_diag[:-1] += coupling
    main_diag[1:] += coupling
    off_diag = -coupling

    weights = np.zeros(len(x))
    weights[:-1] += h / 2
    weights[1:] += h / 2
    return main_diag, off_diag, weights


def build_two_center_matrices(mu, nu, lam, Z_A, Z_B, R, V_extra=None):
    """Build the (H, M) generalized-eigenproblem pencil for one lambda
    -channel two-center problem on the (mu, nu) grid.

    V_extra, if given, is a (len(mu), len(nu)) array of an additional
    potential (e.g. Hartree+XC from a many-electron density) added to the
    bare two-center nuclear attraction -- None reproduces the pure two
    -center Coulomb problem this phase validates against exact H2+-family
    limits.

    Flattening convention: a (len(mu), len(nu)) array is rows-of-nu-per-mu
    (numpy C-order .ravel()), so kron(T_mu, W_nu) applies T_mu (weighted by
    nu's local cell width, since it's being integrated over that
    direction's control volume) along the mu axis, and kron(W_mu, T_nu)
    applies T_nu along the nu axis -- confirmed via the standard identity
    kron(A, B) @ vec(X) = vec(A @ X @ B^T) for row-major-flattened X (B
    diagonal here, so @ B^T = @ B is just a per-column scaling).
    """
    N_mu, N_nu = len(mu), len(nu)
    Tm_diag, Tm_off, w_mu = _sl_operator(mu, lambda m: m**2 - 1)
    Tn_diag, Tn_off, w_nu = _sl_operator(nu, lambda n: 1 - n**2)

    T_mu = sp.diags([Tm_off, Tm_diag, Tm_off], [-1, 0, 1])
    T_nu = sp.diags([Tn_off, Tn_diag, Tn_off], [-1, 0, 1])
    W_mu = sp.diags(w_mu)
    W_nu = sp.diags(w_nu)

    # T_mu/T_nu already represent -L_mu/-L_nu (see _sl_operator), so this
    # sum is directly "-L_mu[f]*g - f*L_nu[g]", each weighted by the other
    # coordinate's local cell width to normalize as a 2D control volume.
    kinetic2D = sp.kron(T_mu, W_nu) + sp.kron(W_mu, T_nu)

    MU, NU = np.meshgrid(mu, nu, indexing="ij")
    centrifugal = (MU**2 - NU**2) / ((MU**2 - 1) * (1 - NU**2))
    V_nuc_term = -R * ((Z_A + Z_B) * MU - (Z_A - Z_B) * NU)
    Q = R**2 * (MU**2 - NU**2) / 2
    W2D = np.outer(w_mu, w_nu)

    diag_extra = (lam**2 * centrifugal + V_nuc_term) * W2D
    if V_extra is not None:
        diag_extra = diag_extra + Q * V_extra * W2D

    H = kinetic2D + sp.diags(diag_extra.ravel())
    M = sp.diags((Q * W2D).ravel())
    return H.tocsc(), M.tocsc()


def solve_two_center_channel(mu, nu, lam, Z_A, Z_B, R, V_extra=None, n_states=4, sigma=None):
    """Solve one lambda-channel via shift-invert eigsh (mirrors
    atomic_scf.solve_radial_channel's approach: factorize H - sigma*M once,
    which stays well-conditioned at natural scale). sigma defaults to a
    rough bound-state energy estimate, -0.5*(Z_A+Z_B)**2 (the united-atom
    1s-like scale), generous enough to bracket the low-lying spectrum for
    any R.

    Returns (energies, eigvecs) with eigvecs of shape (N_mu*N_nu, n_states),
    sorted ascending; eigvecs are un-normalized solutions of the raw
    generalized eigenproblem (not yet normalized against the true 3D volume
    element -- that Jacobian factor is applied by callers when needed).
    """
    H, M = build_two_center_matrices(mu, nu, lam, Z_A, Z_B, R, V_extra)
    if sigma is None:
        sigma = -0.5 * (Z_A + Z_B) ** 2

    ncv = max(2 * n_states + 1, 20)
    for attempt in range(4):
        try:
            energies, w = eigsh(H, k=n_states, M=M, sigma=sigma * 1.2**attempt, which="LM", ncv=ncv)
            break
        except Exception:
            ncv = int(ncv * 1.5) + 5
    else:
        raise RuntimeError(f"solve_two_center_channel: eigsh failed to converge for lambda={lam}, R={R}")

    order = np.argsort(energies)
    return energies[order], w[:, order]
