"""Nuclear (softened Coulomb) potential + Slater/PZ81 XC on the 3D grid
(Phase 3). Slater exchange and PZ81 correlation are copied from
HF_solver/potentials.py (not imported cross-folder, per this repo's
self-contained-project convention, already followed by
Diatomic_HF_solver) -- the formulas are purely local functions of rho, so
they port over completely unchanged from a 1D radial grid to a 3D one.
"""

import numpy as np

import fft_ops

ALPHA_LDA = 2 / 3
ALPHA_SCHWARZ = 0.7


def nuclear_potential(X, Y, Z, nuclei):
    """nuclei: list of (Z_i, x_i, y_i, z_i, softening_i). A bare 1/r
    singularity isn't resolvable on a finite Cartesian grid the way the
    radial/prolate-spheroidal solvers' analytic cusp handling managed it
    -- softened to -Z/sqrt(r^2+softening^2), the same regularization used
    throughout this session's N-body work for the identical reason (a
    finite grid can't resolve a true point singularity). softening's
    effect on accuracy is characterized empirically, not assumed
    negligible (see Atom_SCF_Validation.ipynb)."""
    V = np.zeros_like(X)
    for Z_i, x_i, y_i, z_i, soft_i in nuclei:
        r2 = (X - x_i) ** 2 + (Y - y_i) ** 2 + (Z - z_i) ** 2
        V += -Z_i / np.sqrt(r2 + soft_i**2)
    return V


def slater_exchange_potential(rho, alpha=ALPHA_SCHWARZ):
    """V_x(r) = -3*alpha*(3*rho(r)/(8*pi))**(1/3) -- purely local, so this
    is identical code to the atomic/diatomic solvers, just evaluated on a
    3D array instead of 1D/2D."""
    return -3 * alpha * (3 * rho / (8 * np.pi)) ** (1 / 3)


def pz81_correlation(rho):
    """Perdew-Zunger (1981) LDA correlation energy density and potential
    (copied from HF_solver/potentials.py -- see there for the derivation
    and the rs=1 branch-continuity/zero-density-clipping notes). Boolean
    masking works identically regardless of rho's array shape, so this
    needs no changes at all for the 3D case."""
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


def ks_potential(rho, V_nuc, G2, method="lda", alpha=ALPHA_SCHWARZ):
    """The Kohn-Sham effective potential V_eff = V_nuc + V_H[rho] + V_xc[rho]
    for a given density rho, on the FFT G-vector grid G2.

    Factored out of scf3d.run_scf's iteration so the SCF loop and the
    rt-TDDFT propagator (Quantum Mechanics/TDDFT/) build V_eff from the
    *identical* code -- that identity is what makes "propagating the
    converged ground state produces only phase evolution" an exact
    consistency test rather than an approximate one.

    method: "lda" (Slater exchange at ALPHA_LDA + PZ81 correlation) or
    "xalpha" (Slater exchange at `alpha`).

    Returns (V_eff, parts) with parts = {"V_H", "V_x", "eps_c", "V_c"} for
    the total-energy bookkeeping; eps_c/V_c are None for method="xalpha".
    """
    if method not in ("xalpha", "lda"):
        raise ValueError(f"method must be 'xalpha' or 'lda', got {method!r}")
    V_H = fft_ops.solve_poisson(rho, G2)
    if method == "lda":
        V_x = slater_exchange_potential(rho, ALPHA_LDA)
        eps_c, V_c = pz81_correlation(rho)
        V_xc = V_x + V_c
    else:
        V_x = slater_exchange_potential(rho, alpha)
        V_xc, eps_c, V_c = V_x, None, None
    V_eff = V_nuc + V_H + V_xc
    return V_eff, {"V_H": V_H, "V_x": V_x, "eps_c": eps_c, "V_c": V_c}
