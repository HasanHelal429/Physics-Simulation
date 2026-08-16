"""Shared symbolic-Lagrangian-to-ODE pipeline.

Every notebook in this project follows the same recipe: write down T and U
in terms of generalized coordinates, form L = T - U (or L*exp(b*t) for a
Caldirola-Kanai-style linear-damping trick), derive the Euler-Lagrange
equations, solve for the generalized accelerations, lambdify, and integrate
numerically. This module factors that recipe out once instead of every
notebook hand-rolling it.
"""
import numpy as np
import sympy as sp
from scipy.integrate import odeint


def euler_lagrange(L, coords, t):
    """Derive and solve the Euler-Lagrange equations for a Lagrangian L(t)
    built from `coords`, a list of sp.Function(t) generalized coordinates.

    Returns (sols, coords_d, coords_dd): `sols` maps each coordinate's
    second time-derivative symbol to its solved expression (in terms of
    t and the coordinates/velocities), ready for `build_ode_system`.

    Solved via mass-matrix inversion rather than a generic nonlinear
    `sp.solve`: the Euler-Lagrange equations are always linear in the
    accelerations (only their *coefficients* are messy nonlinear functions
    of the coordinates/velocities), so collecting them into a matrix and
    inverting is exact and dramatically faster than a generic nonlinear
    solve once there are more than 2 coordinates (a naive `sp.solve` on
    the 3-coordinate triple pendulum did not finish in several minutes;
    this approach solves it in seconds).
    """
    coords_d = [sp.diff(c, t) for c in coords]
    coords_dd = [sp.diff(cd, t) for cd in coords_d]
    eqs = [sp.diff(L, c) - sp.diff(sp.diff(L, cd), t) for c, cd in zip(coords, coords_d)]

    mass_matrix, rhs = sp.linear_eq_to_matrix(eqs, coords_dd)
    accel = mass_matrix.LUsolve(rhs)
    # No sp.simplify() here: lambdify doesn't need a simplified expression to
    # evaluate correctly, and simplify() on the messy trig-heavy entries of a
    # >2-DOF matrix inverse can be far slower than the solve itself.
    sols = {cdd: accel[i] for i, cdd in enumerate(coords_dd)}
    return sols, coords_d, coords_dd


def build_ode_system(sols, coords, coords_d, t, params):
    """Lambdify the solved accelerations into a dSdt(S, t, *params)
    callable for scipy.integrate.odeint, with state S interleaved as
    [q1, q1_d, q2, q2_d, ...] in the same order as `coords`.
    """
    coords_dd = [sp.diff(cd, t) for cd in coords_d]
    args = (t, *params, *coords, *coords_d)
    accel_fns = [sp.lambdify(args, sols[cdd]) for cdd in coords_dd]

    def dSdt(S, t, *pvals):
        qs = S[0::2]
        qds = S[1::2]
        accels = [f(t, *pvals, *qs, *qds) for f in accel_fns]
        out = []
        for qd, a in zip(qds, accels):
            out.append(qd)
            out.append(a)
        return out

    return dSdt


def integrate(dSdt, S0, t, params):
    """Thin wrapper around odeint; returns the (len(t), len(S0)) solution array."""
    return odeint(dSdt, S0, t, args=tuple(params))


def energy_fn(T_expr, U_expr, coords, coords_d, params):
    """Lambdify T+U into a callable energy(*coord_vals, *coord_d_vals, *params)
    -> float, for energy-conservation checks."""
    args = (*coords, *coords_d, *params)
    return sp.lambdify(args, T_expr + U_expr)


def linearize(L, coords, t, equilibrium):
    """Small-oscillation mass/stiffness matrices from the same symbolic
    Lagrangian used for the full nonlinear system, evaluated at
    `equilibrium` (dict mapping each coordinate symbol, e.g. the1(t), to
    its equilibrium value -- velocities are assumed zero there).

    Returns (M, K, coords_d) as sympy Matrices (still containing any
    remaining physical-parameter symbols): for L = 1/2 qd^T M qd - 1/2 q^T K q
    near equilibrium, M = Hessian of L w.r.t. velocities, K = -Hessian of L
    w.r.t. displacements (both evaluated at equilibrium, velocities -> 0).
    Normal-mode angular frequencies come from generalized eigenvalues of
    (K, M): squared frequencies solve det(K - w^2 M) = 0.
    """
    coords_d = [sp.diff(c, t) for c in coords]
    n = len(coords)
    M = sp.Matrix(n, n, lambda i, j: sp.diff(L, coords_d[i], coords_d[j]))
    K = sp.Matrix(n, n, lambda i, j: -sp.diff(L, coords[i], coords[j]))

    subs = dict(equilibrium)
    subs.update({cd: 0 for cd in coords_d})
    M = M.subs(subs)
    K = K.subs(subs)
    return M, K, coords_d


def normal_mode_frequencies(M, K, param_values):
    """Numeric normal-mode angular frequencies from linearize()'s (M, K),
    with all remaining free symbols substituted via `param_values` (dict).
    Returns frequencies sorted ascending (only the non-negative, real ones --
    a rigid-body/zero mode shows up as ~0)."""
    Mn = np.array(M.subs(param_values), dtype=float)
    Kn = np.array(K.subs(param_values), dtype=float)
    eigvals = np.linalg.eigvals(np.linalg.solve(Mn, Kn))
    eigvals = np.real(eigvals[np.abs(np.imag(eigvals)) < 1e-8])
    eigvals = np.clip(eigvals, 0, None)
    return np.sort(np.sqrt(eigvals))
