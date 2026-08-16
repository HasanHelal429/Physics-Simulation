"""Boundary conditions for the D2Q9 lattice: halfway bounce-back (no-slip,
for stationary walls and arbitrary obstacle masks), the Zou & He (1997)
velocity BC (for a moving wall, which plain bounce-back can't represent),
and the Guo, Zheng & Shu (2002) forcing scheme (for body-force-driven
flow, e.g. a pressure-gradient-free Poiseuille channel).
"""
import numpy as np

import lattice as lb


def apply_bounce_back(f_pre_collide, f_star, mask):
    """Halfway bounce-back at solid nodes (mask False): skip collision
    there (retain the raw population that streamed in last step) and
    reflect it by permuting opposite-direction pairs in place, so the
    next streaming step carries it straight back out the way it came.
    No ghost/padding layer is needed -- a single mask=False node is a
    self-contained reflecting wall (verified by tracing a population's
    round trip: it returns as if it bounced off the midpoint between the
    solid node and its fluid neighbor, one full timestep later -- the
    standard 2nd-order "halfway" bounce-back).
    """
    out = f_star.copy()
    solid = ~mask
    out[:, solid] = f_pre_collide[:, solid]
    out[:, solid] = out[lb.OPPOSITE][:, solid]
    return out


def zou_he_lid(f, row, u_wall, cols=slice(None)):
    """Zou & He (1997) velocity BC for a moving top lid at y-index `row`,
    prescribing tangential velocity u_wall and no-penetration (u_y=0).
    Solves the 3 populations left unknown by streaming into this row
    (those with e_y=-1: indices 4, 7, 8) directly from the known ones
    (0,1,2,3,5,6) plus the prescribed velocity, exactly reproducing the
    wall velocity -- unlike plain bounce-back, which only ever gives
    zero velocity.

    `cols` restricts which x-indices get the lid treatment -- the two
    corner columns, where the lid meets a stationary side wall, should
    normally be excluded (left to that wall's bounce-back) rather than
    fed this formula's known-populations, which would otherwise be
    reading a solid node's meaningless reflected values.
    """
    f0, f1, f2, f3 = f[0, cols, row], f[1, cols, row], f[2, cols, row], f[3, cols, row]
    f5, f6 = f[5, cols, row], f[6, cols, row]
    rho = f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)
    f4 = f2
    f7 = f5 + 0.5 * (f1 - f3) - 0.5 * rho * u_wall
    f8 = f6 - 0.5 * (f1 - f3) + 0.5 * rho * u_wall
    f_out = f.copy()
    f_out[4, cols, row] = f4
    f_out[7, cols, row] = f7
    f_out[8, cols, row] = f8
    return f_out


def zou_he_inlet_west(f, col, ux_in, uy_in=0.0, rows=slice(None)):
    """Zou & He (1997) velocity BC for a west (left) inlet at x-index
    `col`, prescribing full velocity (ux_in, uy_in). Unlike `zou_he_lid`
    (which prescribes a *tangential* velocity, with the wall-normal
    component trivially zero), this prescribes the wall-*normal*
    component, which needs Zou & He's extra closure assumption --
    bounce-back of the *non-equilibrium* part of the normal-direction
    population, `f1 - f1eq = f3 - f3eq` -- to fully determine the 3
    unknown populations (e_x=+1: indices 1, 5, 8) and the local density
    (not prescribed at a velocity inlet, so solved for self-consistently).
    Verified by substitution: these satisfy rho, rho*ux_in, rho*uy_in
    exactly for any uy_in, not just uy_in=0.
    """
    f0, f2, f3, f4 = f[0, col, rows], f[2, col, rows], f[3, col, rows], f[4, col, rows]
    f6, f7 = f[6, col, rows], f[7, col, rows]
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux_in)
    f1 = f3 + (2.0 / 3.0) * rho * ux_in
    f5 = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux_in + 0.5 * rho * uy_in
    f8 = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux_in - 0.5 * rho * uy_in
    f_out = f.copy()
    f_out[1, col, rows] = f1
    f_out[5, col, rows] = f5
    f_out[8, col, rows] = f8
    return f_out


def outflow_zero_gradient(f, col):
    """Simple zero-gradient (Neumann) outflow at x-index `col`: copy the
    neighboring interior column's post-streaming populations, so the
    flow leaves without imposing an artificial pressure/velocity and
    without reflecting anything back upstream."""
    f_out = f.copy()
    f_out[:, col, :] = f[:, col - 1, :]
    return f_out


def guo_source(rho, ux, uy, Fx, Fy, tau):
    """Guo, Zheng & Shu (2002) discrete lattice forcing term, added to
    the post-collision distribution so a uniform body force per unit
    mass (Fx, Fy) drives the flow with 2nd-order accuracy (naively
    shifting velocity before collision is only 1st order). `ux, uy` must
    already be the force-corrected "true" velocity (ux_raw + F/(2*rho)),
    not the raw momentum-based value.
    """
    ex, ey = lb.E[:, 0][:, None, None], lb.E[:, 1][:, None, None]
    eu = ex * ux[None, :, :] + ey * uy[None, :, :]
    eF = ex * Fx + ey * Fy
    e_minus_u_dot_F = (ex - ux[None, :, :]) * Fx + (ey - uy[None, :, :]) * Fy
    return (1.0 - 1.0 / (2.0 * tau)) * lb.W[:, None, None] * (3.0 * e_minus_u_dot_F + 9.0 * eu * eF)
