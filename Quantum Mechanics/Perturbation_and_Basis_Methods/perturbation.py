"""
Rayleigh-Schrodinger perturbation theory (non-degenerate and degenerate),
built directly on a `basis.BasisSet`: H = H0 + H', where H0 is diagonal
in the basis (its eigenvalues are `basis.energies`) and H' is a
perturbing potential sampled on the same grid as the basis functions.

Everything here is checked in `Nondegenerate_Perturbation_Theory.ipynb`
and `Hydrogen_Stark_Effect.ipynb` against direct diagonalization of the
truncated H0+H' matrix -- something the legacy notebook this replaces
never did at all.
"""

import numpy as np
from scipy.integrate import simpson


def matrix_elements(basis, V_prime_samples):
    """Full perturbation matrix H'_mn = <phi_m|V'|phi_n>, one vectorized
    Simpson integral rather than a per-(m,n) loop."""
    weighted = basis.eigenfunctions * V_prime_samples[np.newaxis, :]
    integrand = weighted[:, np.newaxis, :] * basis.eigenfunctions[np.newaxis, :, :]
    return simpson(integrand, x=basis.x, axis=2)


def first_order_energy(H_prime):
    """E_n^(1) = H'_nn, for every n at once."""
    return np.diag(H_prime)


def first_order_state(H_prime, energies, n_index):
    """Coefficients of the 1st-order wavefunction correction
    |n^(1)> = sum_{m!=n} (H'_mn/(E_n-E_m)) |m>, as a coefficient vector
    (zero at index n_index itself -- the standard intermediate
    normalization choice, <n|n^(1)>=0)."""
    energies = np.asarray(energies)
    c1 = np.zeros(len(energies))
    mask = np.arange(len(energies)) != n_index
    c1[mask] = H_prime[mask, n_index] / (energies[n_index] - energies[mask])
    return c1


def second_order_energy(H_prime, energies, n_index):
    """E_n^(2) = sum_{m!=n} |H'_mn|^2/(E_n-E_m)."""
    energies = np.asarray(energies)
    mask = np.arange(len(energies)) != n_index
    return np.sum(H_prime[mask, n_index] ** 2 / (energies[n_index] - energies[mask]))


def exact_diagonalization(energies, H_prime, lam=1.0):
    """Diagonalize H0+lam*H' directly in the truncated basis --
    'exact' up to basis-truncation error, the ground truth perturbation
    theory is checked against. Returns (eigenvalues, eigenvectors)
    sorted ascending, eigenvectors as columns (numpy.linalg.eigh
    convention)."""
    H_total = np.diag(energies) + lam * H_prime
    return np.linalg.eigh(H_total)


def track_state(evecs, n_index):
    """Which column of `evecs` continues adiabatically from unperturbed
    basis state n_index -- the eigenvector with the largest overlap
    (|evecs[n_index, :]|), robust to any incidental level reordering
    rather than assuming the n_index-th eigenvalue is still the right
    one."""
    return int(np.argmax(np.abs(evecs[n_index, :])))


def degenerate_perturbation(H_prime_submatrix):
    """Diagonalize H' restricted to a degenerate subspace -- the
    standard degenerate perturbation theory recipe: the correct
    zeroth-order states are whichever basis of the subspace diagonalizes
    H' there, and the 1st-order energy shifts are its eigenvalues."""
    return np.linalg.eigh(H_prime_submatrix)
