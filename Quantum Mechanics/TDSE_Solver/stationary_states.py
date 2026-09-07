"""
Time-independent Schrodinger equation: sparse finite-difference Hamiltonian
assembly and diagonalization for the lowest few bound states.

This is a genuinely different numerical method from propagator.py's
spectral split-operator time stepper -- an ordinary 3-point finite
difference Laplacian, Kronecker-summed across axes exactly like the
legacy `2D/3D Schrodinger Equation ... .py` scripts attempted, but built
with scipy.sparse operations directly (no per-cell Python loop) and using
eigsh (H is real symmetric, so this is both faster and more numerically
appropriate than the legacy scripts' `eigs`).

Not part of the propagation pipeline: this module exists purely as a
validation/initial-condition tool -- (a) an independent check on
propagator.py, since propagating one of these eigenstates should change
it only by e^{-i*E*t} (already exercised in Phase 2 with the *analytic*
harmonic-oscillator solution; this module lets the same check run against
*any* potential, not just the harmonic well), and (b) a way to build a
physically meaningful initial wavefunction (e.g. Phase 7's bound
wavepacket) from a genuine eigenstate rather than an arbitrary guess.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _laplacian_1d(n, h, boundary):
    """Standard 3-point d^2/dx^2 stencil, (psi[i+1]-2psi[i]+psi[i-1])/h^2.

    'box': psi is exactly 0 just outside the array (grid.py's Dirichlet
      convention) -- dropped from the stencil, not wrapped or padded, the
      standard interior-point Dirichlet finite-difference Laplacian.
    'periodic': wraps around (adds the two corner entries), the standard
      circulant finite-difference Laplacian.
    """
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    L = sp.diags([off, main, off], offsets=[-1, 0, 1], format='lil')
    if boundary == 'periodic':
        L[0, -1] = 1.0
        L[-1, 0] = 1.0
    return (L / h ** 2).tocsr()


def laplacian_matrix(grid):
    """ndim finite-difference Laplacian on `grid`, as a Kronecker sum of
    1D operators -- ordering matches numpy's default (C-order) ravel/
    reshape, the same order grid.coords/psi arrays already use, so
    `laplacian_matrix(grid) @ psi.ravel()` reshapes back to grid.shape
    directly."""
    ops_1d = [_laplacian_1d(n, h, grid.boundary) for n, h in zip(grid.shape, grid.dx)]
    total = None
    for axis in range(grid.ndim):
        term = None
        for other_axis in range(grid.ndim):
            factor = ops_1d[axis] if other_axis == axis else sp.identity(grid.shape[other_axis], format='csr')
            term = factor if term is None else sp.kron(term, factor, format='csr')
        total = term if total is None else total + term
    return total


def hamiltonian(grid, V, mass=1.0):
    """H = -1/(2*mass) * laplacian + V, as a sparse CSC matrix of shape
    (n, n) with n = prod(grid.shape). Real and symmetric (both boundary
    conventions' Laplacians are), so eigsh is the appropriate solver.

    mass=1.0 (default) is the electron TDSE and is bit-identical to the
    original -1/2 laplacian form. Nuclear_Dynamics/ passes a nuclear
    reduced mass mu here to get vibrational levels on a Born-Oppenheimer
    PES."""
    L = laplacian_matrix(grid)
    return (-(0.5 / mass) * L + sp.diags(V.ravel())).tocsc()


def lowest_states(grid, V, k=6, mass=1.0):
    """Lowest `k` eigenstates of hamiltonian(grid, V, mass). Returns
    (energies, states) with energies sorted ascending and states a list
    of k arrays each shaped like grid.shape (already normalized in the
    plain sum-of-squares sense scipy's eigsh returns; use observables.norm
    if a continuum-normalized wavefunction is needed)."""
    H = hamiltonian(grid, V, mass=mass)
    energies, vectors = spla.eigsh(H, k=k, which='SA')
    order = np.argsort(energies)
    energies = energies[order]
    states = [vectors[:, i].reshape(grid.shape) for i in order]
    return energies, states
