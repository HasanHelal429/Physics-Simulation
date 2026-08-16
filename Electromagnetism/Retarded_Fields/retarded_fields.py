"""Lienard-Wiechert retarded electromagnetic fields for a point charge
moving along an arbitrary specified path r(t). Positions/velocities/
accelerations are 3-vectors (planar motion is just z=0 throughout) --
even for a charge confined to a plane, its radiated B-field points out of
that plane, so the field vectors are inherently 3D.
"""
import math

import numpy as np
from numba import njit, prange
from scipy.optimize import brentq


def _cross3(a, b):
    """3-component cross product, bypassing np.cross's generic N-D
    dispatch (moveaxis/normalize_axis machinery). Profiling showed that
    dispatch overhead, not the actual arithmetic, dominates np.cross's
    cost for plain 3-vectors -- and lw_fields calls it 3 times per
    point evaluated."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def _norm3(v):
    """3-component vector norm, same rationale as `_cross3` for
    np.linalg.norm."""
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def retarded_time(r_func, x, t, c=1.0, v_func=None, guess=None):
    """Solve c*(t - t_ret) = |x - r(t_ret)| for t_ret <= t.

    Always brackets first (the light-travel-time deficit
    g(t_ret) = c*(t-t_ret) - |x-r(t_ret)| is negative at t_ret=t and
    becomes positive far enough in the past for any subluminal path),
    by the same auto-expanding-window search as before.

    Without `v_func`, polishes the root with `brentq` bisection exactly
    as before. With `v_func`, uses safeguarded Newton's method instead:
    g is strictly decreasing in t_ret for any subluminal path
    (g'(t_ret) = -c + n.v(t_ret), and |n.v| <= |v| < c always), so
    Newton with that analytic derivative converges quadratically --
    falling back to a bisection step whenever a Newton step would leave
    the maintained bracket, so it's exactly as robust as bisection in
    the worst case (e.g. near-relativistic beaming, where n.v ~ c makes
    the derivative small and a raw Newton step could overshoot wildly).
    `guess` (e.g. the previous animation frame's t_ret for this same
    observation point, since retarded time moves continuously and
    slowly frame to frame) seeds the first step.
    """
    def g(t_ret):
        return c * (t - t_ret) - _norm3(x - r_func(t_ret))

    window = 1.0
    t_lo = t - window
    g_lo = g(t_lo)
    while g_lo <= 0:
        window *= 2.0
        t_lo = t - window
        g_lo = g(t_lo)
        if window > 1e10:
            raise RuntimeError('could not bracket the retarded time -- is the path superluminal?')

    if v_func is None:
        return brentq(g, t_lo, t)

    t_hi = t
    t_ret = guess if (guess is not None and t_lo < guess < t_hi) else 0.5 * (t_lo + t_hi)
    for _ in range(60):
        R_vec = x - r_func(t_ret)
        R = _norm3(R_vec)
        gval = c * (t - t_ret) - R
        if gval > 0:
            t_lo = t_ret
        elif gval < 0:
            t_hi = t_ret
        else:
            return t_ret
        n = R_vec / R
        gprime = -c + np.dot(n, v_func(t_ret))
        step = gval / gprime if gprime != 0 else None
        t_newton = t_ret - step if step is not None else None
        # Convergence is judged by the size of the Newton step, not by
        # bracket width: Newton typically approaches the root from one
        # side, so the far bound of [t_lo, t_hi] can stay stale and wide
        # even after t_ret itself has converged to machine precision --
        # bracket width is only meaningful as a bisection-progress metric.
        if t_newton is not None and t_lo < t_newton < t_hi:
            if abs(step) < 1e-13 * max(1.0, abs(t_ret)):
                return t_newton
            t_ret = t_newton
        else:
            t_ret = 0.5 * (t_lo + t_hi)
    return t_ret


def _lw_fields_core(r_func, v_func, a_func, x, t, q, c=1.0, eps0=1.0, t_ret_guess=None):
    t_ret = retarded_time(r_func, x, t, c=c, v_func=v_func, guess=t_ret_guess)
    R_vec = x - r_func(t_ret)
    R = _norm3(R_vec)
    n = R_vec / R
    beta = v_func(t_ret) / c
    beta_dot = a_func(t_ret) / c
    denom = (1.0 - np.dot(n, beta))**3

    near = (n - beta) * (1.0 - np.dot(beta, beta)) / (R**2 * denom)
    far = _cross3(n, _cross3(n - beta, beta_dot)) / (c * R * denom)
    E = q / (4 * np.pi * eps0) * (near + far)
    B = _cross3(n, E) / c
    return E, B, t_ret


def lw_fields(r_func, v_func, a_func, x, t, q, c=1.0, eps0=1.0, t_ret_guess=None):
    """Lienard-Wiechert E and B fields at observation point/time (x, t)
    from a point charge q following path r_func(t), with velocity
    v_func(t) and acceleration a_func(t). Returns (E, B) as 3-vectors.

    `t_ret_guess`, if given, seeds the retarded-time solve (see
    `retarded_time`) -- purely a speed hint, doesn't change the result.
    """
    E, B, _ = _lw_fields_core(r_func, v_func, a_func, x, t, q, c=c, eps0=eps0, t_ret_guess=t_ret_guess)
    return E, B


def lw_fields_grid(r_func, v_func, a_func, X, Y, Z, t, q, c=1.0, eps0=1.0, t_ret_guess=None):
    """Vectorized-by-loop helper: evaluate lw_fields at every point of a
    meshgrid (X, Y, Z, all same shape), returning (E_out, B_out,
    t_ret_out) with E_out/B_out shaped (*shape, 3) and t_ret_out shaped
    `shape`. Not internally vectorized (each point needs its own
    retarded-time root-find), just a convenience wrapper for animations.

    Pass the previous call's `t_ret_out` back in as `t_ret_guess` (same
    grid, later `t`) to warm-start every point's Newton solve from
    where it was last frame -- retarded time moves continuously and
    slowly frame to frame for a fixed observation grid, so this cuts
    root-find iterations sharply after the first frame.
    """
    shape = X.shape
    E_out = np.zeros(shape + (3,))
    B_out = np.zeros(shape + (3,))
    t_ret_out = np.empty(shape)
    for idx in np.ndindex(shape):
        x = np.array([X[idx], Y[idx], Z[idx]])
        guess = None if t_ret_guess is None else t_ret_guess[idx]
        E_out[idx], B_out[idx], t_ret_out[idx] = _lw_fields_core(
            r_func, v_func, a_func, x, t, q, c=c, eps0=eps0, t_ret_guess=guess)
    return E_out, B_out, t_ret_out


# --- numba-accelerated grid solve (opt-in fast path) -----------------------
#
# lw_fields_grid above works with any Python callable for r_func/v_func/
# a_func, but pays Python-interpreter and numpy-array-allocation overhead
# on every one of its (grid points) x (Newton iterations) inner steps.
# Profiling a 70x70 grid showed that overhead dominates: JIT-compiling the
# same algorithm (as plain scalar math, no array allocation) and running it
# under numba brought a single-threaded 70x70/5-frame benchmark from
# ~1.1s down to ~0.005s (~230x), with numba's parallel prange loop over
# grid points adding another ~5x on top of that on a 16-thread machine.
#
# The catch: numba's nopython mode can't call arbitrary Python closures, so
# r_func/v_func/a_func must themselves be @numba.njit functions returning
# plain (x, y, z) float tuples (not np.array) -- e.g.:
#
#     @njit
#     def r_func(t):
#         return r_orbit * math.cos(Omega * t), r_orbit * math.sin(Omega * t), 0.0
#
# lw_fields/lw_fields_grid are untouched and still accept any Python
# callable; use lw_fields_grid_jit only where the speed matters (animation
# frame loops) and you're willing to write njit-compatible paths.

@njit(cache=True)
def _retarded_time_jit(r_func, v_func, xo, yo, zo, t, c, guess):
    """Same algorithm as `retarded_time`'s Newton branch, in scalar
    (no-array-allocation) form for numba. `guess` uses NaN, not None, to
    mean "no guess" (NaN != NaN is a cheap, nopython-compatible test)."""
    window = 1.0
    t_lo = t - window
    rx, ry, rz = r_func(t_lo)
    g_lo = c * (t - t_lo) - math.sqrt((xo - rx)**2 + (yo - ry)**2 + (zo - rz)**2)
    while g_lo <= 0:
        window *= 2.0
        t_lo = t - window
        rx, ry, rz = r_func(t_lo)
        g_lo = c * (t - t_lo) - math.sqrt((xo - rx)**2 + (yo - ry)**2 + (zo - rz)**2)

    t_hi = t
    t_ret = guess if (guess == guess and t_lo < guess < t_hi) else 0.5 * (t_lo + t_hi)
    for _ in range(60):
        rx, ry, rz = r_func(t_ret)
        Rx, Ry, Rz = xo - rx, yo - ry, zo - rz
        R = math.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
        gval = c * (t - t_ret) - R
        if gval > 0:
            t_lo = t_ret
        elif gval < 0:
            t_hi = t_ret
        else:
            return t_ret
        nx, ny, nz = Rx / R, Ry / R, Rz / R
        vx, vy, vz = v_func(t_ret)
        gprime = -c + (nx * vx + ny * vy + nz * vz)
        step = 0.0
        if gprime != 0:
            step = gval / gprime
            t_newton = t_ret - step
        else:
            t_newton = t_lo - 1
        if t_lo < t_newton < t_hi:
            if abs(step) < 1e-13 * max(1.0, abs(t_ret)):
                return t_newton
            t_ret = t_newton
        else:
            t_ret = 0.5 * (t_lo + t_hi)
    return t_ret


@njit(cache=True)
def _lw_fields_jit_core(r_func, v_func, a_func, xo, yo, zo, t, q, c, eps0, guess):
    t_ret = _retarded_time_jit(r_func, v_func, xo, yo, zo, t, c, guess)
    rx, ry, rz = r_func(t_ret)
    Rx, Ry, Rz = xo - rx, yo - ry, zo - rz
    R = math.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
    nx, ny, nz = Rx / R, Ry / R, Rz / R
    vx, vy, vz = v_func(t_ret)
    ax, ay, az = a_func(t_ret)
    bx, by, bz = vx / c, vy / c, vz / c
    bdx, bdy, bdz = ax / c, ay / c, az / c

    ndotb = nx * bx + ny * by + nz * bz
    denom = (1.0 - ndotb) ** 3
    b2 = bx * bx + by * by + bz * bz

    near_x = (nx - bx) * (1.0 - b2) / (R * R * denom)
    near_y = (ny - by) * (1.0 - b2) / (R * R * denom)
    near_z = (nz - bz) * (1.0 - b2) / (R * R * denom)

    ex, ey, ez = nx - bx, ny - by, nz - bz
    cx1 = ey * bdz - ez * bdy
    cy1 = ez * bdx - ex * bdz
    cz1 = ex * bdy - ey * bdx
    fx = ny * cz1 - nz * cy1
    fy = nz * cx1 - nx * cz1
    fz = nx * cy1 - ny * cx1
    far_x = fx / (c * R * denom)
    far_y = fy / (c * R * denom)
    far_z = fz / (c * R * denom)

    k = q / (4.0 * math.pi * eps0)
    Ex, Ey, Ez = k * (near_x + far_x), k * (near_y + far_y), k * (near_z + far_z)
    Bx = (ny * Ez - nz * Ey) / c
    By = (nz * Ex - nx * Ez) / c
    Bz = (nx * Ey - ny * Ex) / c
    return Ex, Ey, Ez, Bx, By, Bz, t_ret


@njit(cache=True, parallel=True)
def _lw_fields_grid_jit_core(r_func, v_func, a_func, X, Y, Z, t, q, c, eps0, t_ret_guess):
    shape = X.shape
    E_out = np.empty(shape + (3,))
    B_out = np.empty(shape + (3,))
    t_ret_out = np.empty(shape)
    Xf, Yf, Zf, guess_f = X.ravel(), Y.ravel(), Z.ravel(), t_ret_guess.ravel()
    Ef, Bf, tf = E_out.reshape(-1, 3), B_out.reshape(-1, 3), t_ret_out.ravel()
    n = Xf.shape[0]
    for i in prange(n):
        ex, ey, ez, bx, by, bz, tr = _lw_fields_jit_core(
            r_func, v_func, a_func, Xf[i], Yf[i], Zf[i], t, q, c, eps0, guess_f[i])
        Ef[i, 0], Ef[i, 1], Ef[i, 2] = ex, ey, ez
        Bf[i, 0], Bf[i, 1], Bf[i, 2] = bx, by, bz
        tf[i] = tr
    return E_out, B_out, t_ret_out


def lw_fields_grid_jit(r_func, v_func, a_func, X, Y, Z, t, q, c=1.0, eps0=1.0, t_ret_guess=None):
    """Numba-accelerated, parallelized equivalent of `lw_fields_grid` --
    same (E_out, B_out, t_ret_out) contract and same warm-starting
    convention (feed a previous call's `t_ret_out` back in as
    `t_ret_guess`), but ~2-3 orders of magnitude faster.

    Requires `r_func`, `v_func`, `a_func` to be `@numba.njit` functions
    returning plain `(x, y, z)` float tuples (not `np.array`) -- see the
    module-level comment above `_retarded_time_jit` for an example and
    the rationale. Use this instead of `lw_fields_grid` in animation
    frame loops, where the speed actually matters.
    """
    guess = np.full(X.shape, np.nan) if t_ret_guess is None else t_ret_guess
    return _lw_fields_grid_jit_core(r_func, v_func, a_func, X, Y, Z, t, q, c, eps0, guess)
