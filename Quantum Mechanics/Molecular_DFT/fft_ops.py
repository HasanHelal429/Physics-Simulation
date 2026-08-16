"""FFT-based kinetic energy operator and Poisson solver (Phases 1-2),
sharing the same G-vector grid (grid3d.g_vectors) -- periodic boundary
conditions make both operators exactly diagonal in Fourier space, giving
an O(N log N), floating-point-exact (not finite-difference-approximate)
implementation of each. See Molecular_DFT_Plan.md for why this replaces
the earlier solvers' sparse finite-difference approach.
"""

import numpy as np


def apply_kinetic(psi, G2):
    """-0.5*nabla^2 psi via FFT: -nabla^2 e^{iGx} = |G|^2 e^{iGx}, so the
    kinetic operator is exactly diagonal (multiplication by 0.5*G2) in
    Fourier space. Returns a real array (psi is assumed real, as it can
    always be chosen for a real, Hermitian Hamiltonian; .real discards
    floating-point imaginary noise from the FFT round-trip)."""
    psi_hat = np.fft.fftn(psi)
    return np.fft.ifftn(0.5 * G2 * psi_hat).real


def solve_poisson(rho, G2, G_zero_index=(0, 0, 0)):
    """Hartree potential V_H from a (periodic) charge density rho, via
    nabla^2 V_H = -4*pi*rho -> V_H_G = 4*pi*rho_G / |G|^2 in Fourier
    space. The G=0 (uniform/average) component is formally 0/0 for a
    genuinely periodic charge distribution (undefined absolute potential
    offset) -- set to zero, the standard convention (equivalent to fixing
    the arbitrary zero of potential at the cell average).

    IMPORTANT, characterized empirically (see
    Kinetic_and_Poisson_Validation.ipynb): this convention makes the
    periodic array of charge well-defined (implicitly adding a uniform
    compensating "jellium" background to cancel any net charge -- an
    electron density's total charge is *not* zero by itself, only the
    full nuclear+electron system is), but the resulting field is NOT the
    same as an isolated charge's true 1/r-decaying field -- spurious
    interaction with periodic images (and the compensating background)
    remains, converging away only algebraically (~1/L) as the box grows,
    not exponentially. Validated directly against a Gaussian test charge
    with a known closed-form potential: ~51% error at L=10, still ~7% at
    L=80 (same physical charge/width, just a bigger box). This is a real,
    well-known limitation of naive periodic-FFT Poisson solves for
    non-neutral densities (which is exactly what a single atom/molecule's
    *electron* density is, on its own) -- production isolated-system DFT
    codes use specialized methods (e.g. Martyna-Tuckerman, cutoff-Coulomb,
    wavelet solvers) specifically because of this slow convergence. Out of
    scope for this initial build (see Molecular_DFT_Plan.md's Scope
    decision) -- accepted and documented, not fixed, here: box size is
    chosen generously and the resulting error is reported honestly rather
    than assumed negligible.
    """
    rho_hat = np.fft.fftn(rho)
    G2_safe = G2.copy()
    G2_safe[G_zero_index] = 1.0  # placeholder to avoid 0/0; overwritten below
    V_hat = 4 * np.pi * rho_hat / G2_safe
    V_hat[G_zero_index] = 0.0
    return np.fft.ifftn(V_hat).real
