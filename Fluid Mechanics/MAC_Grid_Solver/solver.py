"""
Top-level time-stepping: advect -> diffuse -> project -> re-enforce the
obstacle mask, plus an adaptive CFL-limited timestep.

Per MAC_Grid_Solver_Plan.md's Chorin sequencing, step 3 ("enforce velocity
BCs on u* before the divergence/Poisson solve") is already satisfied by
diffusion.py's own implicit solve -- both the Dirichlet-exact wall/inflow
values and the obstacle mask are pinned as part of that solve, not as a
separate pass. Step 6 ("re-enforce after correction") is only strictly
needed for the obstacle mask here: pressure.py's gradient() never touches
the Dirichlet-exact wall faces by construction (zero there always), but it
is not mask-aware and can nudge an obstacle-interior face away from zero.

`build_system` precomputes everything that depends only on geometry/BCs
(not on the evolving dt or nu) once per run: the pressure Poisson system
and both velocity components' diffusion operators (see diffusion.py for
why the latter matters -- rebuilding those from scratch every step would
otherwise waste 45-60% of the diffusion cost on redundant matrix assembly).
"""

from collections import namedtuple

import numpy as np

import advection as adv
import diffusion as diff
import pressure as pr
import boundary as bnd

System = namedtuple('System', ['pressure', 'u_diffusion', 'v_diffusion'])


def cfl_timestep(u, v, dx, dy, cfl_number=0.5, dt_max=None, eps=1e-10):
    """dt = cfl_number * min(dx, dy) / max(|u|, |v|, eps), capped at dt_max."""
    speed = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), eps)
    dt = cfl_number * min(dx, dy) / speed
    return dt if dt_max is None else min(dt, dt_max)


def build_system(grid, bc_spec):
    """Precompute the pressure Poisson system and both velocity components'
    diffusion operators for a static (grid, bc_spec) pair -- reused for
    every step of a run."""
    pressure_system = pr.build_pressure_system(grid.cell_mask, grid.dx, grid.dy,
                                                outflow_edges=bnd.outflow_edges(bc_spec))
    u_normal_bc, u_tangential_bc = bnd.diffusion_bc_for_u(bc_spec)
    v_normal_bc, v_tangential_bc = bnd.diffusion_bc_for_v(bc_spec)
    u_diff_op = diff.build_diffusion_operator(grid.u_mask, grid.dx, grid.dy, normal_axis=0,
                                               normal_bc=u_normal_bc, tangential_bc=u_tangential_bc)
    v_diff_op = diff.build_diffusion_operator(grid.v_mask, grid.dx, grid.dy, normal_axis=1,
                                               normal_bc=v_normal_bc, tangential_bc=v_tangential_bc)
    return System(pressure=pressure_system, u_diffusion=u_diff_op, v_diffusion=v_diff_op)


def step(u, v, grid, bc_spec, system, dt, nu, rho=1.0):
    """One full timestep: advect, implicitly diffuse, project, re-enforce
    the obstacle mask. Returns (u_new, v_new, p, max_div_after)."""
    ghost_fn = bnd.advection_ghost_fn(bc_spec)
    outflow_u_x, outflow_v_y = bnd.outflow_normal_flags(bc_spec)
    u_a, v_a = adv.advect(u, v, dt, grid.dx, grid.dy, ghost_fn=ghost_fn,
                           outflow_u_x=outflow_u_x, outflow_v_y=outflow_v_y)

    u_d = diff.diffuse_component(u_a, system.u_diffusion, dt, nu, x0=u_a)
    v_d = diff.diffuse_component(v_a, system.v_diffusion, dt, nu, x0=v_a)

    u_p, v_p, p, max_div = pr.project_velocity(u_d, v_d, system.pressure, dt, rho)
    u_new, v_new = bnd.apply_mask(u_p, v_p, grid.u_mask, grid.v_mask)
    return u_new, v_new, p, max_div
