"""
Implicit (backward-Euler) viscous diffusion of a velocity component on the
MAC grid, solved via warm-started CG rather than a prefactorized direct
solve. See MAC_Grid_Solver_Plan.md: the operator (I - dt*nu*L) is
dt-dependent, and dt is adaptive, so there is no static factorization to
reuse the way pressure.py's dt-independent Poisson matrix has -- L is
symmetric positive definite, so CG warm-started from the previous step's
field converges in a handful of iterations regardless of how often dt
changes.

A face-aligned velocity component's own (normal) axis is on-node at the
domain edge -- e.g. u's left/right walls sit exactly at u's own x=0/x=lx
samples -- so a Dirichlet condition there pins that row/column exactly
(operators.laplacian_matrix's 'dirichlet_exact'). Its tangential axis is
offset half a cell from the domain edge -- e.g. u's top/bottom walls,
where u has no sample exactly on the wall -- and needs the same
ghost-extrapolation Dirichlet (or Neumann) treatment pressure.py's Poisson
solve uses.

`build_diffusion_operator` separates the dt-*independent* pieces (the
geometry/BC-only sparse Laplacian, its RHS contribution, and which rows are
pinned) from `diffuse_component`'s per-step dt-dependent work. Profiling
showed the naive approach -- calling operators.laplacian_matrix fresh every
step, as an earlier version of this module did -- spends 45-60% of its time
on matrix assembly alone at grid sizes from 64x64 up to Phase 7's cylinder
channel, despite the mask and BCs never changing during a run. Precomputing
once and reusing removes that entirely.

A second round of profiling (forced by Phase 7's much larger cylinder-channel
grid) found that even after that fix, re-forming I - dt*nu*L and re-pinning
it via operators.pin_rows's LIL round trip *every step* was still ~95% of
diffuse_component's cost -- the CG solve itself was a few milliseconds,
matrix bookkeeping was ~250ms. The fix: A = I - dt*nu*L has exactly the same
sparsity pattern as L (multiplying by a scalar and adding to the diagonal
never changes which entries are nonzero, and L already has a diagonal entry
in every row), and operators.laplacian_matrix's own internal pinning already
collapses every pinned row of L down to a single diagonal entry (=1) with
nothing else in it. So each pinned row of A, before any correction, is just
`1 - dt*nu*1` at that one diagonal position -- there is no stray off-diagonal
content to zero out, only that one value to overwrite back to exactly 1.
`build_diffusion_operator` locates every row's diagonal position in L's CSR
data array once; `diffuse_component` then forms A's `data` array directly
via plain numpy indexing (multiply, add, overwrite) and hands scipy the
unchanged `indices`/`indptr` from L, skipping sparse arithmetic and the LIL
round trip entirely.
"""

from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import operators as op

DiffusionOperator = namedtuple('DiffusionOperator',
                                ['L', 'rhs_bc', 'pinned', 'pinned_values', 'shape', 'diag_pos', 'pinned_diag_pos'])


def _edge_spec(value, exact):
    """An edge value is either the string 'neumann' (zero-gradient -- e.g.
    an outflow's normal component, or a free-slip wall's tangential one)
    or a fixed value to pin exactly. `exact` selects which Dirichlet flavor
    a fixed value uses: True for a component's own (on-node) normal axis,
    False for its tangential (half-cell-offset) axis."""
    if isinstance(value, str) and value == 'neumann':
        return 'neumann'
    return ('dirichlet_exact' if exact else 'dirichlet', value)


def build_diffusion_operator(mask, dx, dy, normal_axis, normal_bc, tangential_bc):
    """
    Precompute the dt-independent pieces of one velocity component's
    implicit diffusion solve, reused for every step of a run.

    mask: this component's fluid mask (u_mask or v_mask).
    normal_axis: 0 if the component's own direction is x (u), 1 if y (v).
    normal_bc: (lo, hi) values along the normal axis, each either the
        string 'neumann' (e.g. an outflow) or a fixed Dirichlet-exact wall
        value (scalar or an array matching that edge's length).
    tangential_bc: (lo, hi) values along the other axis, each either the
        string 'neumann' or a fixed Dirichlet-ghost value.
    """
    nx, ny = mask.shape
    norm_lo, norm_hi = _edge_spec(normal_bc[0], exact=True), _edge_spec(normal_bc[1], exact=True)
    tan_lo, tan_hi = _edge_spec(tangential_bc[0], exact=False), _edge_spec(tangential_bc[1], exact=False)

    if normal_axis == 0:
        bc = {'left': norm_lo, 'right': norm_hi, 'bottom': tan_lo, 'top': tan_hi}
        edge_slices = [(0, slice(None), normal_bc[0]), (-1, slice(None), normal_bc[1])]
    elif normal_axis == 1:
        bc = {'bottom': norm_lo, 'top': norm_hi, 'left': tan_lo, 'right': tan_hi}
        edge_slices = [(slice(None), 0, normal_bc[0]), (slice(None), -1, normal_bc[1])]
    else:
        raise ValueError(f"normal_axis must be 0 or 1, got {normal_axis!r}")

    L, rhs_bc = op.laplacian_matrix(nx, ny, dx, dy, mask=mask, bc=bc)

    pinned = ~mask.copy()
    pinned_values = np.zeros((nx, ny))
    for i_sl, j_sl, value in edge_slices:
        if isinstance(value, str) and value == 'neumann':
            continue  # a real unknown, solved via the matrix's own Neumann stencil -- not pinned
        pinned[i_sl, j_sl] = True
        pinned_values[i_sl, j_sl] = value

    n = nx * ny
    L = L.tocsr()
    L.sort_indices()
    diag_pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        start, end = L.indptr[i], L.indptr[i + 1]
        diag_pos[i] = start + np.searchsorted(L.indices[start:end], i)

    pinned_flat = pinned.ravel()
    pinned_diag_pos = diag_pos[pinned_flat]

    return DiffusionOperator(L=L, rhs_bc=rhs_bc, pinned=pinned_flat,
                              pinned_values=pinned_values.ravel(), shape=(nx, ny),
                              diag_pos=diag_pos, pinned_diag_pos=pinned_diag_pos)


def diffuse_component(field, diff_op, dt, nu, x0=None, rtol=1e-10):
    """
    Implicitly diffuse one velocity component by dt: solve
    (I - dt*nu*L) field_new = field + dt*nu*rhs_bc  for field_new, using
    the precomputed `diff_op` from `build_diffusion_operator`.

    x0: warm-start guess (typically the previous step's field); defaults
        to `field`.
    """
    nx, ny = diff_op.shape
    n = nx * ny
    L = diff_op.L

    data = (-dt * nu) * L.data
    data[diff_op.diag_pos] += 1.0
    data[diff_op.pinned_diag_pos] = 1.0
    A = sp.csr_matrix((data, L.indices, L.indptr), shape=(n, n))

    b = field.ravel() + dt * nu * diff_op.rhs_bc.ravel()
    b[diff_op.pinned] = diff_op.pinned_values[diff_op.pinned]

    x0_flat = (x0 if x0 is not None else field).ravel()
    field_new, info = spla.cg(A, b, x0=x0_flat, rtol=rtol)
    if info != 0:
        raise RuntimeError(f"diffusion CG did not converge (info={info})")
    return field_new.reshape(nx, ny)
