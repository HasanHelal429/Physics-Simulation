"""
Conservative finite-volume convection on the MAC grid, time-stepped with
3-stage explicit SSP-RK3 (Shu & Osher 1988). See MAC_Grid_Solver_Plan.md
for why this replaces semi-Lagrangian advection: flux-conservative form
matters for trustworthy long-time-integrated statistics (drag, shedding
frequency), and centered-difference convection is unconditionally unstable
under forward Euler, RK2, or AB2 -- RK3 is the minimum explicit RK order
with a genuine CFL-bounded stability region for this operator.

By default every routine here assumes periodic boundaries (every shift
wraps around) -- exact for none of the Phase 6/7 scenarios, but the
standard, simplest way to validate a convection operator in isolation
(Phase 2's tests). For a real domain, `convective_tendency`/`advect`
accept optional tangential ghost values (see `_corner_uv`); boundary.py
supplies them per Phase 5.

The *normal*-direction wraparound (u's own x-neighbors, v's own
y-neighbors) only needs fixing at an outflow edge. At a Dirichlet-exact
wall/inflow, it's harmless: it only ever contaminates that boundary
component's own face value, which diffusion.py pins outright downstream
regardless of what advection computed there, and which pressure.py's
gradient() never perturbs either. But an outflow edge's normal velocity is
*not* pinned -- it's a genuine free unknown -- so periodic wraparound there
would mix in data from clear across the domain instead of the correct
zero-gradient extrapolation; `outflow_u_x`/`outflow_v_y` flags below select
that extrapolation instead of wraparound, per edge.

The *tangential* direction is a different story regardless of BC type --
e.g. periodic wrap in u's y-roll would mix data from the far wall into
u-rows that are genuine, physically meaningful interior unknowns near the
near wall, not a discarded/pinned value -- which is what `u_ghost_y`/
`v_ghost_x` fix.

Momentum convective term, conservative form (Kim & Moin 1985):
  d(u)/dt = -d(u*u)/dx - d(u*v)/dy
  d(v)/dt = -d(u*v)/dx - d(v*v)/dy
discretized so that the u*u and v*v terms use values interpolated to cell
centers (the natural location for the "other half" of a MAC face's control
volume) and the cross term u*v uses values interpolated to grid corners,
shared between the two momentum components.
"""

import numpy as np


def _corner_uv(u, v, u_ghost_y=None, v_ghost_x=None):
    """u*v interpolated to every grid corner (i*dx, j*dy), i=0..nx, j=0..ny,
    shape (nx+1, ny+1).

    u_ghost_y: optional (bottom, top) pair, each shape (nx+1,) -- the
    physically correct value of u one cell beyond y=0 and y=ny (mirror or
    BC-consistent, e.g. a no-slip wall's Dirichlet-ghost), used in place of
    periodic wraparound for u's tangential (y) direction.
    v_ghost_x: optional (left, right) pair, each shape (ny+1,) -- the same
    idea for v's tangential (x) direction.
    If omitted, both fall back to periodic wraparound (Phase 2's isolated
    tests)."""
    if u_ghost_y is None:
        u_corner = 0.5 * (np.roll(u, 1, axis=1) + u)                        # (nx+1, ny)
        u_corner = np.concatenate([u_corner, u_corner[:, :1]], axis=1)      # (nx+1, ny+1)
    else:
        u_bottom, u_top = u_ghost_y
        u_ext = np.concatenate([u_bottom[:, None], u, u_top[:, None]], axis=1)  # (nx+1, ny+2)
        u_corner = 0.5 * (u_ext[:, :-1] + u_ext[:, 1:])                     # (nx+1, ny+1)

    if v_ghost_x is None:
        v_corner = 0.5 * (np.roll(v, 1, axis=0) + v)                        # (nx, ny+1)
        v_corner = np.concatenate([v_corner, v_corner[:1, :]], axis=0)      # (nx+1, ny+1)
    else:
        v_left, v_right = v_ghost_x
        v_ext = np.concatenate([v_left[None, :], v, v_right[None, :]], axis=0)  # (nx+2, ny+1)
        v_corner = 0.5 * (v_ext[:-1, :] + v_ext[1:, :])                     # (nx+1, ny+1)

    return u_corner * v_corner


def convective_tendency(u, v, dx, dy, u_ghost_y=None, v_ghost_x=None,
                         outflow_u_x=(False, False), outflow_v_y=(False, False)):
    """-div(velocity (x) velocity), i.e. (d(u)/dt, d(v)/dt) from convection
    alone. u: shape (nx+1, ny). v: shape (nx, ny+1). `u_ghost_y`/`v_ghost_x`
    as in `_corner_uv`. `outflow_u_x`/`outflow_v_y` are (lo, hi) booleans:
    True selects zero-gradient extrapolation instead of periodic wraparound
    for u's own x-boundary / v's own y-boundary, needed only where that
    edge is a genuine (unpinned) outflow -- see module docstring."""
    # u*u term, at u's own locations: interpolate u to the two flanking
    # cell centers, square, difference.
    u_half = 0.5 * (u[:-1, :] + u[1:, :])                                # (nx, ny), cell centers
    right_ghost = u_half[-1:, :] if outflow_u_x[1] else u_half[:1, :]
    left_ghost = u_half[:1, :] if outflow_u_x[0] else u_half[-1:, :]
    u_half_right = np.concatenate([u_half, right_ghost], axis=0)         # (nx+1, ny): cell right of each u-face
    u_half_left = np.concatenate([left_ghost, u_half], axis=0)           # (nx+1, ny): cell left of each u-face
    d_uu_dx = (u_half_right ** 2 - u_half_left ** 2) / dx

    # v*v term, symmetric construction for v's own locations.
    v_half = 0.5 * (v[:, :-1] + v[:, 1:])                                # (nx, ny), cell centers
    top_ghost = v_half[:, -1:] if outflow_v_y[1] else v_half[:, :1]
    bottom_ghost = v_half[:, :1] if outflow_v_y[0] else v_half[:, -1:]
    v_half_top = np.concatenate([v_half, top_ghost], axis=1)             # (nx, ny+1)
    v_half_bottom = np.concatenate([bottom_ghost, v_half], axis=1)       # (nx, ny+1)
    d_vv_dy = (v_half_top ** 2 - v_half_bottom ** 2) / dy

    # u*v cross term, shared corner values, differenced along the other axis.
    uv_c = _corner_uv(u, v, u_ghost_y, v_ghost_x)                        # (nx+1, ny+1)
    d_uv_dy_for_u = (uv_c[:, 1:] - uv_c[:, :-1]) / dy                     # (nx+1, ny), matches u
    d_uv_dx_for_v = (uv_c[1:, :] - uv_c[:-1, :]) / dx                     # (nx, ny+1), matches v

    du_dt = -(d_uu_dx + d_uv_dy_for_u)
    dv_dt = -(d_uv_dx_for_v + d_vv_dy)
    return du_dt, dv_dt


def advect(u, v, dt, dx, dy, ghost_fn=None, outflow_u_x=(False, False), outflow_v_y=(False, False)):
    """Advance (u, v) by dt under the convective term alone, via 3-stage
    SSP-RK3 (Shu & Osher 1988): strong-stability-preserving, so it doesn't
    introduce new oscillations/overshoot beyond what forward Euler would at
    each of its three sub-stages.

    ghost_fn(u, v) -> (u_ghost_y, v_ghost_x), recomputed against each RK
    sub-stage's own (evolving) field -- needed for Neumann-type tangential
    boundaries (e.g. free-slip, outflow), whose ghost value tracks the
    current interior state rather than being a fixed constant. None
    (default) falls back to periodic wraparound, matching Phase 2's tests.
    outflow_u_x/outflow_v_y: static (don't depend on the evolving field, so
    passed once rather than through ghost_fn) -- see convective_tendency."""
    def tendency(u_, v_):
        gy, gx = ghost_fn(u_, v_) if ghost_fn is not None else (None, None)
        return convective_tendency(u_, v_, dx, dy, u_ghost_y=gy, v_ghost_x=gx,
                                    outflow_u_x=outflow_u_x, outflow_v_y=outflow_v_y)

    du1, dv1 = tendency(u, v)
    u1, v1 = u + dt * du1, v + dt * dv1

    du2, dv2 = tendency(u1, v1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * du2)
    v2 = 0.75 * v + 0.25 * (v1 + dt * dv2)

    du3, dv3 = tendency(u2, v2)
    u_new = u / 3 + (2 / 3) * (u2 + dt * du3)
    v_new = v / 3 + (2 / 3) * (v2 + dt * dv3)
    return u_new, v_new


def scalar_convective_tendency(phi, u, v, dx, dy):
    """-div(phi * velocity) for a cell-centered scalar phi transported by a
    *given* (fixed, not self-evolving) velocity field. Used for tracer/dye
    visualization (Phase 7's vorticity-street animation) and, here, to test
    the RK3 time integration on a genuinely linear transport problem,
    isolated from momentum self-advection's Burgers-like nonlinearity."""
    phi_pad_x = np.concatenate([phi[-1:, :], phi, phi[:1, :]], axis=0)   # (nx+2, ny)
    phi_at_u = 0.5 * (phi_pad_x[:-1, :] + phi_pad_x[1:, :])              # (nx+1, ny)
    flux_u = phi_at_u * u
    d_phi_u_dx = (flux_u[1:, :] - flux_u[:-1, :]) / dx                  # (nx, ny)

    phi_pad_y = np.concatenate([phi[:, -1:], phi, phi[:, :1]], axis=1)  # (nx, ny+2)
    phi_at_v = 0.5 * (phi_pad_y[:, :-1] + phi_pad_y[:, 1:])             # (nx, ny+1)
    flux_v = phi_at_v * v
    d_phi_v_dy = (flux_v[:, 1:] - flux_v[:, :-1]) / dy                  # (nx, ny)

    return -(d_phi_u_dx + d_phi_v_dy)


def advect_scalar(phi, u, v, dt, dx, dy):
    """3-stage SSP-RK3 advance of a passive scalar under a fixed velocity field."""
    k1 = scalar_convective_tendency(phi, u, v, dx, dy)
    phi1 = phi + dt * k1
    k2 = scalar_convective_tendency(phi1, u, v, dx, dy)
    phi2 = 0.75 * phi + 0.25 * (phi1 + dt * k2)
    k3 = scalar_convective_tendency(phi2, u, v, dx, dy)
    return phi / 3 + (2 / 3) * (phi2 + dt * k3)
