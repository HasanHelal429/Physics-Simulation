"""
Pressure projection (Chorin 1968): solve the discrete Poisson equation for
pressure and use its gradient to make an intermediate velocity field
divergence-free -- the Helmholtz-Hodge decomposition, u* = u_div_free +
grad(phi), applied by solving for phi (called p here, scaled by dt/rho)
and subtracting its gradient back out.

Unlike diffusion.py's dt-dependent (I - dt*nu*L) operator, the pressure
Laplacian depends only on geometry (dx, dy, the obstacle mask, and which
edges are walls vs. outflow) -- all fixed for the duration of a run -- so
it is prefactorized once (scipy.sparse.linalg.splu) and reused for every
projection, a large and valid performance win.

Boundary conditions: Neumann (zero normal flow) at every wall and obstacle
face by default; Dirichlet p=0 at any edge named in `outflow_edges`. If
there is no outflow at all, every boundary is Neumann, and the assembled
matrix is exactly singular -- adding a constant to p doesn't change
grad(p) -- so cell (0,0) is pinned to p=0 to remove the resulting
one-dimensional null space. Any other solution of the same system differs
from this one only by a constant additive shift; see Validation.ipynb for
the check that this holds and that the compatibility condition
(sum(div(u*)) ~ 0, required for the closed system to be solvable at all)
is satisfied whenever a real u* is projected in a closed domain.
"""

from collections import namedtuple

import numpy as np
import scipy.sparse.linalg as spla

import operators as op
import grid as g

PressureSystem = namedtuple('PressureSystem',
                             ['lu', 'mask', 'u_mask', 'v_mask', 'dx', 'dy', 'singular', 'pin_index', 'bc'])


def build_pressure_system(mask, dx, dy, outflow_edges=(), pin_index=0):
    """Assemble and prefactorize the pressure Poisson matrix for a static
    obstacle/domain geometry. `pin_index` is the flat (row-major) index of
    the reference cell pinned to p=0 when the system is otherwise singular
    (no outflow edges); it's exposed mainly so Validation.ipynb can pin a
    different cell and check the two solutions agree up to a constant."""
    nx, ny = mask.shape
    bc = {edge: ('dirichlet', 0.0) for edge in outflow_edges}
    matrix, _ = op.laplacian_matrix(nx, ny, dx, dy, mask=mask, bc=bc)

    singular = len(outflow_edges) == 0
    if singular:
        n = nx * ny
        pinned = np.zeros(n, dtype=bool)
        pinned[pin_index] = True
        matrix, _ = op.pin_rows(matrix, np.zeros(n), pinned, np.zeros(n))

    # MMD_AT_PLUS_A (minimum-degree ordering on A+A^T) gives noticeably less
    # fill-in than SuperLU's default COLAMD for this 2D Poisson stencil --
    # measured ~45% fewer nonzeros in the L/U factors and a ~2x faster
    # triangular solve, which matters since solve_pressure() runs every
    # step for the life of a run, while factorization is a one-time cost.
    lu = spla.splu(matrix.tocsc(), permc_spec='MMD_AT_PLUS_A')
    u_mask, v_mask = g.face_masks(mask)
    return PressureSystem(lu=lu, mask=mask, u_mask=u_mask, v_mask=v_mask, dx=dx, dy=dy,
                           singular=singular, pin_index=pin_index, bc=bc)


def solve_pressure(system, rhs):
    """Solve L p = rhs for the static geometry captured in `system`."""
    rhs_flat = rhs.ravel().copy()
    if system.singular:
        rhs_flat[system.pin_index] = 0.0
    return system.lu.solve(rhs_flat).reshape(system.mask.shape)


def project_velocity(u, v, system, dt, rho=1.0):
    """
    Full projection step:
      solve  L p = (rho/dt) div(u*)
      correct  u_new = u* - (dt/rho) grad(p)
    Returns (u_new, v_new, p, max_div_after) -- the last a concrete
    diagnostic of how close to divergence-free the corrected field is,
    something Simple_Fluid's fixed-iteration Jacobi solve could never
    actually guarantee reached (near) zero.
    """
    div_u_star = op.divergence(u, v, system.dx, system.dy)
    p = solve_pressure(system, (rho / dt) * div_u_star)
    grad_u, grad_v = op.gradient(p, system.dx, system.dy, bc=system.bc,
                                  u_mask=system.u_mask, v_mask=system.v_mask)
    u_new = u - (dt / rho) * grad_u
    v_new = v - (dt / rho) * grad_v
    max_div_after = np.max(np.abs(op.divergence(u_new, v_new, system.dx, system.dy)))
    return u_new, v_new, p, max_div_after
