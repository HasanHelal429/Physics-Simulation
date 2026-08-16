"""Orchestrates a full D2Q9 LBM timestep: collision (with optional Guo
forcing), halfway bounce-back at solid nodes, streaming, and an optional
Zou-He moving-lid velocity BC -- plus a Re<->(U,tau) unit-conversion
helper matching this repo's `Re = U*L/nu` convention (see
`MAC_Grid_Solver`).
"""
import lattice as lb
import boundary as bc


def tau_from_reynolds(Re, L, U):
    """nu = U*L/Re (lattice units); tau = 3*nu + 1/2 (Chapman-Enskog)."""
    nu = U * L / Re
    return 3.0 * nu + 0.5


def step(f, tau, mask=None, forcing=None, moving_wall=None):
    """One LBM timestep.

    mask: boolean array (True=fluid) for stationary walls/obstacles, or
        None for a fully periodic domain.
    forcing: optional (Fx, Fy) uniform body force per unit mass (Guo
        scheme).
    moving_wall: optional (row, u_wall) or (row, u_wall, cols) applying a
        Zou-He moving lid at y-index `row` (optionally restricted to a
        column slice, e.g. to exclude corners shared with a side wall)
        after streaming.
    """
    rho, ux, uy = lb.moments(f)
    if forcing is not None:
        Fx, Fy = forcing
        ux = ux + Fx / (2.0 * rho)
        uy = uy + Fy / (2.0 * rho)
    feq = lb.equilibrium(rho, ux, uy)
    f_star = f - (f - feq) / tau
    if forcing is not None:
        f_star = f_star + bc.guo_source(rho, ux, uy, Fx, Fy, tau)
    if mask is not None:
        f_star = bc.apply_bounce_back(f, f_star, mask)
    f_new = lb.stream(f_star)
    if moving_wall is not None:
        row, u_wall, *rest = moving_wall
        cols = rest[0] if rest else slice(None)
        f_new = bc.zou_he_lid(f_new, row, u_wall, cols=cols)
    return f_new


def step_channel(f, tau, mask, inlet_col, u_in, outlet_col, inlet_rows=slice(None)):
    """One LBM timestep for an open channel: bounce-back walls/obstacle
    (via `mask`), a Zou-He velocity inlet at x-index `inlet_col`
    (restricted to `inlet_rows` -- excludes the top/bottom wall rows,
    same reasoning as `zou_he_lid`'s corner exclusion), and a
    zero-gradient outlet at x-index `outlet_col`. Used for the cylinder
    vortex-shedding case, where (unlike the closed lid-driven cavity) the
    domain is genuinely open at both ends."""
    f_star = lb.collide(f, tau)
    f_star = bc.apply_bounce_back(f, f_star, mask)
    f_new = lb.stream(f_star)
    f_new = bc.zou_he_inlet_west(f_new, inlet_col, u_in, 0.0, rows=inlet_rows)
    f_new = bc.outflow_zero_gradient(f_new, outlet_col)
    return f_new
