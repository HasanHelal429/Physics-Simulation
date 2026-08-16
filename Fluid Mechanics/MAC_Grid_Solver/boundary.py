"""
Velocity boundary conditions, translated from a per-edge scenario
specification into the arguments advection.py, diffusion.py, and
pressure.py each need.

A `bc_spec` is a plain dict keyed by edge name ('left', 'right', 'bottom',
'top'), each value a dict {'type': ..., 'value': ...}:
  'no_slip'   -- normal velocity 0 (wall is impermeable); tangential
                 velocity fixed at `value` (default 0, or e.g. a moving
                 lid's speed).
  'free_slip' -- normal velocity 0; tangential velocity Neumann
                 (zero-gradient) -- a symmetry/far-field wall.
  'inflow'    -- normal velocity fixed at `value` (a scalar or an array
                 matching the edge's length, e.g. a Poiseuille profile);
                 tangential velocity fixed at `value` too (default 0).
  'outflow'   -- both components Neumann (zero-gradient); paired with a
                 Dirichlet p=0 pressure condition on the same edge.

Left/right edges: u is normal, v is tangential. Bottom/top edges: v is
normal, u is tangential.
"""

import numpy as np

_NORMAL_AXIS = {'left': 0, 'right': 0, 'bottom': 1, 'top': 1}


def _normal_value(edge_bc):
    t = edge_bc['type']
    if t in ('no_slip', 'free_slip'):
        return 0.0
    if t == 'inflow':
        return edge_bc.get('value', 0.0)
    if t == 'outflow':
        return 'neumann'
    raise ValueError(f"unknown bc type {t!r}")


def _tangential_value(edge_bc):
    t = edge_bc['type']
    if t == 'no_slip':
        return edge_bc.get('value', 0.0)
    if t == 'inflow':
        return edge_bc.get('tangential', 0.0)
    if t in ('free_slip', 'outflow'):
        return 'neumann'
    raise ValueError(f"unknown bc type {t!r}")


def diffusion_bc_for_u(bc_spec):
    """(normal_bc, tangential_bc) for diffusion.diffuse_component(..., normal_axis=0, ...)."""
    normal_bc = (_normal_value(bc_spec['left']), _normal_value(bc_spec['right']))
    tangential_bc = (_tangential_value(bc_spec['bottom']), _tangential_value(bc_spec['top']))
    return normal_bc, tangential_bc


def diffusion_bc_for_v(bc_spec):
    """(normal_bc, tangential_bc) for diffusion.diffuse_component(..., normal_axis=1, ...)."""
    normal_bc = (_normal_value(bc_spec['bottom']), _normal_value(bc_spec['top']))
    tangential_bc = (_tangential_value(bc_spec['left']), _tangential_value(bc_spec['right']))
    return normal_bc, tangential_bc


def outflow_edges(bc_spec):
    """Edge names with type 'outflow', for pressure.build_pressure_system."""
    return tuple(edge for edge, spec in bc_spec.items() if spec['type'] == 'outflow')


def _tangential_ghost(edge_bc, interior):
    """The ghost value one cell beyond a tangential-direction wall, given
    the adjacent interior row/column: Dirichlet mirror (2B - interior) for
    a fixed tangential value, or a plain copy (zero-gradient) for Neumann."""
    t = edge_bc['type']
    if t == 'no_slip':
        return 2 * edge_bc.get('value', 0.0) - interior
    if t == 'inflow':
        return 2 * edge_bc.get('tangential', 0.0) - interior
    if t in ('free_slip', 'outflow'):
        return interior.copy()
    raise ValueError(f"unknown bc type {t!r}")


def advection_ghost_fn(bc_spec):
    """Build the ghost_fn(u, v) -> (u_ghost_y, v_ghost_x) advection.advect
    needs for a non-periodic domain's *tangential* direction (see
    advection.py's module docstring)."""
    def ghost_fn(u, v):
        u_ghost_y = (_tangential_ghost(bc_spec['bottom'], u[:, 0]),
                     _tangential_ghost(bc_spec['top'], u[:, -1]))
        v_ghost_x = (_tangential_ghost(bc_spec['left'], v[0, :]),
                     _tangential_ghost(bc_spec['right'], v[-1, :]))
        return u_ghost_y, v_ghost_x
    return ghost_fn


def outflow_normal_flags(bc_spec):
    """(outflow_u_x, outflow_v_y) for advection.advect's *normal*-direction
    flags: whether u's left/right, or v's bottom/top, edge is an outflow
    (needing zero-gradient extrapolation instead of periodic wraparound,
    since an outflow's normal velocity is a genuine unpinned unknown --
    see advection.py's module docstring)."""
    outflow_u_x = (bc_spec['left']['type'] == 'outflow', bc_spec['right']['type'] == 'outflow')
    outflow_v_y = (bc_spec['bottom']['type'] == 'outflow', bc_spec['top']['type'] == 'outflow')
    return outflow_u_x, outflow_v_y


def apply_mask(u, v, u_mask, v_mask):
    """Zero out obstacle-blocked faces. Diffusion's own implicit solve
    already pins these (and the Dirichlet-exact wall values) as part of
    solving, but pressure.py's gradient() is not mask-aware and can nudge
    an obstacle-interior face away from zero during projection -- this is
    the Chorin sequence's final BC-touchpoint, needed only for the mask."""
    u = u.copy()
    v = v.copy()
    u[~u_mask] = 0.0
    v[~v_mask] = 0.0
    return u, v
