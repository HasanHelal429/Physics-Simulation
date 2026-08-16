"""Shared 2D Poisson's-equation solver for electrostatics: a sparse
5-point Laplacian (built via the central-difference-stencil +
Kronecker-sum technique from `legacy/numerical.py`, credited below) and a
direct sparse linear solve for the potential V, given a charge density
rho and a set of fixed-potential ("conductor") cells -- rather than
iterative relaxation run for an arbitrary number of iterations with no
convergence check.
"""
from functools import reduce
from itertools import cycle
from math import factorial

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _difference(derivative, accuracy=1):
    """Central-difference stencil coefficients and offsets for a given
    derivative order, from the Vandermonde-inversion technique at
    http://web.media.mit.edu/~crtaylor/calculator.html (credited in the
    original `legacy/numerical.py`)."""
    derivative += 1
    radius = accuracy + derivative // 2 - 1
    points = range(-radius, radius + 1)
    coefficients = np.linalg.inv(np.vander(points))
    return coefficients[-derivative] * factorial(derivative - 1), points


def _operator(shape, *differences):
    """N-D finite-difference operator as a sparse matrix, built from 1D
    stencils via Kronecker sums (credit to Philip Zucker for the
    kronsum argument-order fix, per the original `legacy/numerical.py`)."""
    differences = zip(shape, cycle(differences))
    factors = (sp.diags(*diff, shape=(dim,) * 2) for dim, diff in differences)
    return reduce(lambda a, f: sp.kronsum(f, a, format='csc'), factors)


def laplacian_operator(shape, dx, dy):
    """Sparse 2D 5-point Laplacian matrix for a `shape`-shaped grid."""
    d2, pts = _difference(2, accuracy=1)
    return _operator(shape, (d2 / dx**2, pts), (d2 / dy**2, pts))


def solve_poisson(rho, dx, dy, fixed_mask=None, fixed_values=None, eps0=1.0):
    """Solve laplacian(V) = -rho/eps0 by direct sparse linear solve.
    `fixed_mask` (bool array, same shape as rho) marks cells held at a
    fixed potential (conductors / Dirichlet boundary); `fixed_values`
    gives their potential there. Cells outside `fixed_mask` are solved
    for directly from the interior 5-point Laplacian equations."""
    shape = rho.shape
    N = rho.size
    L = laplacian_operator(shape, dx, dy).tolil()
    rhs = (-rho / eps0).ravel()

    if fixed_mask is not None:
        idx = np.flatnonzero(fixed_mask.ravel())
        L[idx, :] = 0.0
        L[idx, idx] = 1.0
        rhs[idx] = fixed_values.ravel()[idx] if hasattr(fixed_values, 'ravel') else fixed_values

    V = spla.spsolve(L.tocsc(), rhs)
    return V.reshape(shape)


def efield(V, dx, dy):
    """E = -grad(V) via central differences (one-sided at the edges)."""
    dVdx, dVdy = np.gradient(V, dx, dy)
    return -dVdx, -dVdy
