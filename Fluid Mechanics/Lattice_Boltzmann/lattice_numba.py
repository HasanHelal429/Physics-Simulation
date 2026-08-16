"""Numba-jitted D2Q9 BGK step, fusing collision + halfway bounce-back +
streaming + an optional Zou-He lid into a single compiled kernel.

This is a performance-only alternative to `lattice.py`/`boundary.py`/
`solver.py`'s vectorized-numpy implementation (same house-style precedent
as `Orbital_Dynamics/N_Body_Gravity`'s numba-accelerated Barnes-Hut/FMM):
the physics and validation live in the numpy version, this module is
cross-checked against it (see `Validation.ipynb`'s numba cross-check
cell) rather than validated independently, and is only used where the
numpy version is too slow to run interactively (the Re=1000 lid-driven
cavity needs tens of thousands of steps on a 255x255 grid).
"""
import numpy as np
from numba import njit

_EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
_EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


@njit(cache=True, fastmath=True)
def step_numba(f, tau, mask, lid_row, u_wall, col_lo, col_hi):
    """One full D2Q9 BGK timestep. `lid_row < 0` disables the Zou-He lid."""
    nx, ny = mask.shape
    f_star = np.empty_like(f)

    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                rho = 0.0
                ux = 0.0
                uy = 0.0
                for k in range(9):
                    fk = f[k, i, j]
                    rho += fk
                    ux += fk * _EX[k]
                    uy += fk * _EY[k]
                ux /= rho
                uy /= rho
                u2 = ux * ux + uy * uy
                for k in range(9):
                    eu = _EX[k] * ux + _EY[k] * uy
                    feq = _W[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
                    f_star[k, i, j] = f[k, i, j] - (f[k, i, j] - feq) / tau
            else:
                for k in range(9):
                    f_star[k, i, j] = f[k, i, j]

    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                tmp0 = f_star[0, i, j]
                tmp1 = f_star[1, i, j]
                tmp2 = f_star[2, i, j]
                tmp3 = f_star[3, i, j]
                tmp4 = f_star[4, i, j]
                tmp5 = f_star[5, i, j]
                tmp6 = f_star[6, i, j]
                tmp7 = f_star[7, i, j]
                tmp8 = f_star[8, i, j]
                tmp = (tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8)
                for k in range(9):
                    f_star[k, i, j] = tmp[_OPP[k]]

    f_new = np.empty_like(f)
    for k in range(9):
        dx = _EX[k]
        dy = _EY[k]
        for i in range(nx):
            si = (i - dx) % nx
            for j in range(ny):
                sj = (j - dy) % ny
                f_new[k, i, j] = f_star[k, si, sj]

    if lid_row >= 0:
        for i in range(col_lo, col_hi):
            f0 = f_new[0, i, lid_row]
            f1 = f_new[1, i, lid_row]
            f2 = f_new[2, i, lid_row]
            f3 = f_new[3, i, lid_row]
            f5 = f_new[5, i, lid_row]
            f6 = f_new[6, i, lid_row]
            rho = f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)
            f_new[4, i, lid_row] = f2
            f_new[7, i, lid_row] = f5 + 0.5 * (f1 - f3) - 0.5 * rho * u_wall
            f_new[8, i, lid_row] = f6 - 0.5 * (f1 - f3) + 0.5 * rho * u_wall

    return f_new


@njit(cache=True, fastmath=True)
def step_channel_numba(f, tau, mask, inlet_col, u_in, outlet_col, row_lo, row_hi):
    """Open-channel D2Q9 BGK step: collision + halfway bounce-back (walls
    and obstacle, via `mask`) + streaming + a Zou-He velocity inlet at
    `inlet_col` (rows [row_lo, row_hi) only, excluding wall rows) + a
    zero-gradient outlet at `outlet_col`. See `solver.step_channel` for
    the numpy reference this is cross-validated against."""
    nx, ny = mask.shape
    f_star = np.empty_like(f)

    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                rho = 0.0
                ux = 0.0
                uy = 0.0
                for k in range(9):
                    fk = f[k, i, j]
                    rho += fk
                    ux += fk * _EX[k]
                    uy += fk * _EY[k]
                ux /= rho
                uy /= rho
                u2 = ux * ux + uy * uy
                for k in range(9):
                    eu = _EX[k] * ux + _EY[k] * uy
                    feq = _W[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
                    f_star[k, i, j] = f[k, i, j] - (f[k, i, j] - feq) / tau
            else:
                for k in range(9):
                    f_star[k, i, j] = f[k, i, j]

    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                tmp0 = f_star[0, i, j]
                tmp1 = f_star[1, i, j]
                tmp2 = f_star[2, i, j]
                tmp3 = f_star[3, i, j]
                tmp4 = f_star[4, i, j]
                tmp5 = f_star[5, i, j]
                tmp6 = f_star[6, i, j]
                tmp7 = f_star[7, i, j]
                tmp8 = f_star[8, i, j]
                tmp = (tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8)
                for k in range(9):
                    f_star[k, i, j] = tmp[_OPP[k]]

    f_new = np.empty_like(f)
    for k in range(9):
        dx = _EX[k]
        dy = _EY[k]
        for i in range(nx):
            si = (i - dx) % nx
            for j in range(ny):
                sj = (j - dy) % ny
                f_new[k, i, j] = f_star[k, si, sj]

    for j in range(row_lo, row_hi):
        f0 = f_new[0, inlet_col, j]
        f2 = f_new[2, inlet_col, j]
        f3 = f_new[3, inlet_col, j]
        f4 = f_new[4, inlet_col, j]
        f6 = f_new[6, inlet_col, j]
        f7 = f_new[7, inlet_col, j]
        rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)
        f_new[1, inlet_col, j] = f3 + (2.0 / 3.0) * rho * u_in
        f_new[5, inlet_col, j] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_in
        f_new[8, inlet_col, j] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_in

    for k in range(9):
        for j in range(ny):
            f_new[k, outlet_col, j] = f_new[k, outlet_col - 1, j]

    return f_new
