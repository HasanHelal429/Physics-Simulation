"""
Discrete differential operators on the MAC grid (see grid.py). Plain-array
divergence/gradient/vorticity for diagnostics and physics, plus a generic
sparse Laplacian assembler used by both pressure.py's Poisson solve
(cell-centered fields, 'dirichlet'/ghost-extrapolated domain edges) and
diffusion.py's implicit viscous solve (face-aligned velocity components,
'dirichlet_exact'/pinned domain edges along a component's own axis, where
the wall sits exactly on a grid sample rather than half a cell beyond it).
"""

import numpy as np
import scipy.sparse as sp


def divergence(u, v, dx, dy):
    """Discrete divergence at cell centers, shape (nx, ny)."""
    return (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy


def gradient(p, dx, dy, bc=None, u_mask=None, v_mask=None):
    """Discrete pressure gradient at interior u/v-faces. At a domain-edge
    face, the gradient is exactly 0 by default -- correct for a Neumann
    wall/inflow edge, where dp/dn=0 is the boundary condition itself, so
    that velocity (already set by the BC) is rightly left untouched by
    projection. An outflow edge is different: `bc` (same format as
    `laplacian_matrix`) can mark it ('dirichlet', value), and the gradient
    there is a real one-sided ghost-extrapolated estimate (mirroring the
    Laplacian's own Dirichlet-ghost treatment), since dp/dn genuinely isn't
    zero there and that face's velocity is a free unknown projection must
    actually correct.

    u_mask/v_mask: pass these (grid.u_mask/v_mask) whenever there's an
    obstacle. A face touching an obstacle is not a real degree of freedom
    (its velocity is pinned to 0 downstream regardless), but the plain
    interior formula above doesn't know that and will compute a spurious
    correction there from the obstacle-interior's pinned p=0 -- which then
    contaminates the *neighboring fluid* cell's divergence too, since that
    cell's own divergence() reads this face as if it were a real value.
    Zeroing the correction at masked faces (mirroring the domain-edge
    convention: don't correct a face that isn't a free unknown) fixes both."""
    nx, ny = p.shape
    bc = bc or {}
    grad_u = np.zeros((nx + 1, ny))
    grad_u[1:-1, :] = (p[1:, :] - p[:-1, :]) / dx
    grad_v = np.zeros((nx, ny + 1))
    grad_v[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy

    def outflow_value(edge):
        spec = bc.get(edge, 'neumann')
        if spec == 'neumann':
            return None
        kind, value = spec
        if kind != 'dirichlet':
            raise ValueError(f"unexpected bc kind {kind!r} for edge {edge!r} in gradient()")
        return value

    left, right = outflow_value('left'), outflow_value('right')
    if left is not None:
        grad_u[0, :] = 2 * (p[0, :] - left) / dx
    if right is not None:
        grad_u[-1, :] = -2 * (p[-1, :] - right) / dx

    bottom, top = outflow_value('bottom'), outflow_value('top')
    if bottom is not None:
        grad_v[:, 0] = 2 * (p[:, 0] - bottom) / dy
    if top is not None:
        grad_v[:, -1] = -2 * (p[:, -1] - top) / dy

    if u_mask is not None:
        grad_u[~u_mask] = 0.0
    if v_mask is not None:
        grad_v[~v_mask] = 0.0

    return grad_u, grad_v


def vorticity(u, v, dx, dy, cell_mask=None):
    """Vorticity zeta = dv/dx - du/dy at interior grid corners, shape (nx-1, ny-1).

    cell_mask: pass this (grid.cell_mask) whenever there's an obstacle.
    This is a plain finite-difference stencil with no idea where the
    obstacle is, so near it, differencing a real fluid velocity against a
    solid-interior value that's pinned to exactly 0 (not a real flow
    quantity) produces a large spurious "vorticity" -- visually obvious
    wherever the true vorticity is small (e.g. the stagnation point at the
    front of a cylinder) and easy to miss wherever it's swamped by large
    genuine values (e.g. a shedding wake). `cell_mask` sets corners
    touching any solid cell to NaN (transparent in most colormaps) instead
    of plotting that artifact as if it were real flow structure."""
    dv_dx = (v[1:, 1:-1] - v[:-1, 1:-1]) / dx
    du_dy = (u[1:-1, 1:] - u[1:-1, :-1]) / dy
    zeta = dv_dx - du_dy
    if cell_mask is not None:
        corner_fluid = (cell_mask[:-1, :-1] & cell_mask[1:, :-1] &
                         cell_mask[:-1, 1:] & cell_mask[1:, 1:])
        zeta = np.where(corner_fluid, zeta, np.nan)
    return zeta


def pin_rows(matrix, rhs_flat, pinned_flat, values_flat):
    """Overwrite `pinned_flat` rows of a sparse matrix to the identity, and
    the corresponding entries of `rhs_flat` to `values_flat`, so those
    unknowns solve to exactly the given value regardless of everything
    else. Used both inside `laplacian_matrix` (obstacle-interior and
    dirichlet_exact cells) and by diffusion.py (which must redo this on
    `I - dt*nu*L`, since composing with the identity breaks the pin `L`
    alone had)."""
    if not pinned_flat.any():
        return matrix, rhs_flat
    matrix = matrix.tolil()
    matrix[pinned_flat, :] = 0
    matrix[pinned_flat, pinned_flat] = 1.0
    matrix = matrix.tocsr()
    rhs_flat = rhs_flat.copy()
    rhs_flat[pinned_flat] = values_flat[pinned_flat]
    return matrix, rhs_flat


def laplacian_matrix(nx, ny, dx, dy, mask=None, bc=None):
    """
    Assemble the sparse 5-point Laplacian for a field on an (nx, ny) grid.

    mask: (nx,ny) bool, True = active/fluid. An inactive neighbor
    (obstacle) is always treated as a zero-flux (Neumann) face: dropped
    from the stencil with no diagonal penalty. This is not configurable
    via `bc`, which only governs the 4 domain edges.

    bc: optional dict with keys a subset of {'left','right','bottom','top'},
    each one of:
      'neumann' (default) -- zero-gradient.
      ('dirichlet', value) -- a boundary half a cell beyond the outermost
        sample (the finite-volume convention for a cell-centered field
        like pressure), enforced by ghost extrapolation.
      ('dirichlet_exact', value) -- an on-node boundary: the outermost
        row/column of the array itself sits exactly on the wall (e.g. a
        face-aligned velocity component's own axis -- u's x=0/lx edges).
        That row/column is pinned exactly to `value` (which may be a
        scalar or an array matching the edge's length, e.g. an inflow
        profile) rather than ghost-extrapolated.

    Returns (matrix, rhs): a scipy.sparse.csr_matrix of shape (nx*ny, nx*ny)
    and an (nx,ny) array of the RHS contribution from ghost-extrapolated
    Dirichlet edges (zero elsewhere) that must be added to the Poisson
    right-hand side before solving. Fully inactive (obstacle-interior)
    cells and dirichlet_exact edges are pinned to the identity (with RHS
    set to 0 or `value` respectively) so the returned system solves
    directly to the right answer there even in isolation, e.g. `matrix @
    field.ravel() + rhs.ravel()` reproduces `value` at a pinned dirichlet_exact
    cell for any `field`.
    """
    if mask is None:
        mask = np.ones((nx, ny), dtype=bool)
    bc = bc or {}
    n = nx * ny

    def flat(i, j):
        return i * ny + j

    inv_dx2 = 1.0 / dx ** 2
    inv_dy2 = 1.0 / dy ** 2

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    diag = np.zeros((nx, ny))
    rhs = np.zeros((nx, ny))
    rows, cols, data = [], [], []

    directions = [(-1, 0, inv_dx2, 'left'), (1, 0, inv_dx2, 'right'),
                  (0, -1, inv_dy2, 'bottom'), (0, 1, inv_dy2, 'top')]
    edge_slices = {'left': (0, slice(None)), 'right': (-1, slice(None)),
                   'bottom': (slice(None), 0), 'top': (slice(None), -1)}

    for di, dj, invh2, edge in directions:
        ni, nj = ii + di, jj + dj
        in_grid = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)

        neighbor_active = np.zeros((nx, ny), dtype=bool)
        neighbor_active[in_grid] = mask[ni[in_grid], nj[in_grid]]

        interior_link = mask & in_grid & neighbor_active
        rows.append(flat(ii[interior_link], jj[interior_link]))
        cols.append(flat(ni[interior_link], nj[interior_link]))
        data.append(np.full(int(interior_link.sum()), invh2))
        diag[interior_link] -= invh2

        at_domain_edge = mask & ~in_grid
        edge_spec = bc.get(edge, 'neumann')
        if edge_spec == 'neumann':
            pass  # dropped neighbor, no diagonal penalty: zero-flux mirror
        else:
            kind, value = edge_spec
            if kind == 'dirichlet_exact':
                pass  # this row/column is pinned outright below; its own
                      # stencil equation here is moot, it gets overwritten
            elif kind == 'dirichlet':
                diag[at_domain_edge] -= 2 * invh2
                rhs[at_domain_edge] += 2 * invh2 * value
            else:
                raise ValueError(f"unknown bc kind {kind!r} for edge {edge!r}")
        # obstacle-adjacent faces (in_grid but not neighbor_active) always
        # just drop the neighbor, regardless of `bc` (which only governs
        # domain edges): zero-flux mirror, matching a solid wall.

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    data.append(diag.ravel())

    matrix = sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )

    pinned = ~mask.copy()
    pinned_values = np.zeros((nx, ny))
    for edge, spec in bc.items():
        if isinstance(spec, tuple) and spec[0] == 'dirichlet_exact':
            sl = edge_slices[edge]
            pinned[sl] = True
            pinned_values[sl] = spec[1]

    matrix, rhs_flat = pin_rows(matrix, rhs.ravel(), pinned.ravel(), pinned_values.ravel())
    rhs = rhs_flat.reshape(nx, ny)

    return matrix, rhs
