"""
Stable Fluids solver (Jos Stam, 1999) on a square grid, implemented with
PyTorch tensors so it runs on GPU when available.

Shared by `Simple Fluid Sim.ipynb` (open domain) and `Fluid Sim Obstacle.ipynb`
(flow around a solid boundary). Every step function takes an optional boolean
`mask` (True = fluid, False = solid); pass mask=None for the open-domain case
to skip the per-cell obstacle boundary pass entirely.
"""

import torch
import torch.nn.functional as F


def add_sources(x, x0, dt):
    return x + x0 * dt


def swap(x, x0):
    return x0, x


def _shift_up(t, fill):
    """out[i, j] = t[i - 1, j]; `fill` at i = 0."""
    out = torch.roll(t, 1, dims=0)
    out[0, :] = fill
    return out


def _shift_down(t, fill):
    """out[i, j] = t[i + 1, j]; `fill` at i = N - 1."""
    out = torch.roll(t, -1, dims=0)
    out[-1, :] = fill
    return out


def _shift_left(t, fill):
    """out[i, j] = t[i, j - 1]; `fill` at j = 0."""
    out = torch.roll(t, 1, dims=1)
    out[:, 0] = fill
    return out


def _shift_right(t, fill):
    """out[i, j] = t[i, j + 1]; `fill` at j = N - 1."""
    out = torch.roll(t, -1, dims=1)
    out[:, -1] = fill
    return out


def set_bnd(b, x, mask=None):
    """Apply boundary conditions to a field.

    b selects the field kind:
      0 - scalar (density, pressure): zero-gradient at walls
      1 - x-velocity: no-penetration (sign-flipped) at left/right walls
      2 - y-velocity: no-penetration (sign-flipped) at top/bottom walls

    If `mask` is given, the same no-penetration/zero-gradient logic is
    applied at the fluid/solid interface, and the field is zeroed out
    everywhere outside the mask. Cells off the grid count as solid, same
    as an obstacle.
    """
    x_bc = x.clone()
    if b == 0:
        x_bc[0, :] = x_bc[1, :]
        x_bc[-1, :] = x_bc[-2, :]
        x_bc[:, 0] = x_bc[:, 1]
        x_bc[:, -1] = x_bc[:, -2]
    elif b == 1:
        x_bc[0, :] = -x_bc[1, :]
        x_bc[-1, :] = -x_bc[-2, :]
        x_bc[:, 0] = x_bc[:, 1]
        x_bc[:, -1] = x_bc[:, -2]
    elif b == 2:
        x_bc[0, :] = x_bc[1, :]
        x_bc[-1, :] = x_bc[-2, :]
        x_bc[:, 0] = -x_bc[:, 1]
        x_bc[:, -1] = -x_bc[:, -2]

    if mask is not None:
        # Fluid-ness and values of each cell's 4 neighbors (off-grid = solid = False).
        fluid_up = _shift_up(mask, False)
        fluid_down = _shift_down(mask, False)
        fluid_left = _shift_left(mask, False)
        fluid_right = _shift_right(mask, False)
        val_up = _shift_up(x_bc, 0.0)
        val_down = _shift_down(x_bc, 0.0)
        val_left = _shift_left(x_bc, 0.0)
        val_right = _shift_right(x_bc, 0.0)

        # A fluid cell touching any non-fluid neighbor is on the obstacle boundary.
        mask_edge = mask & (~fluid_up | ~fluid_down | ~fluid_left | ~fluid_right)

        obstacle_val = x_bc
        if b == 0:
            # Zero-gradient: copy from whichever fluid neighbor exists, in
            # up/down/left/right priority (later overwrites earlier, matching
            # the original per-pixel loop order).
            obstacle_val = torch.where(fluid_up, val_up, obstacle_val)
            obstacle_val = torch.where(fluid_down, val_down, obstacle_val)
            obstacle_val = torch.where(fluid_left, val_left, obstacle_val)
            obstacle_val = torch.where(fluid_right, val_right, obstacle_val)
        elif b == 1:
            # No-penetration across a vertical (up/down) obstacle face only.
            vertical_corridor = fluid_up & fluid_down
            obstacle_val = torch.where(fluid_up & ~vertical_corridor, -val_up, obstacle_val)
            obstacle_val = torch.where(fluid_down & ~vertical_corridor, -val_down, obstacle_val)
        elif b == 2:
            # No-penetration across a horizontal (left/right) obstacle face only.
            horizontal_corridor = fluid_left & fluid_right
            obstacle_val = torch.where(fluid_left & ~horizontal_corridor, -val_left, obstacle_val)
            obstacle_val = torch.where(fluid_right & ~horizontal_corridor, -val_right, obstacle_val)

        x_bc = torch.where(mask_edge, obstacle_val, x_bc)
        x_bc[~mask] = 0

    return x_bc


def diffusion(x, x0, diff, dt, b, mask=None, iterations=20):
    """Implicit (Gauss-Seidel-style, via Jacobi iteration) diffusion of field x0 into x."""
    a = dt * diff * x.shape[0] * x.shape[1]
    x = x0
    for _ in range(iterations):
        laplacian = (torch.roll(x, 1, 0) + torch.roll(x, -1, 0) +
                     torch.roll(x, 1, 1) + torch.roll(x, -1, 1))
        x = (x0 + a * laplacian) / (1 + 4 * a)
        x = set_bnd(b, x, mask)
    return x


def advection(d, d0, u, v, dt, b, mask=None):
    """Semi-Lagrangian advection: backtrace each cell through (u, v) and sample d0 there."""
    N = d.shape[0]
    dt0 = dt * N
    i = torch.arange(N, device=d.device).view(-1, 1).expand(N, N)
    j = torch.arange(N, device=d.device).view(1, -1).expand(N, N)

    x = torch.clamp(i - dt0 * u, 0.5, N - 1.5)
    y = torch.clamp(j - dt0 * v, 0.5, N - 1.5)
    x_norm = (x / (N - 1)) * 2 - 1
    y_norm = (y / (N - 1)) * 2 - 1
    grid = torch.stack((y_norm, x_norm), dim=-1).unsqueeze(0)  # [1, N, N, 2]

    d0_unsq = d0.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
    d_new = F.grid_sample(d0_unsq, grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0]
    return set_bnd(b, d_new, mask)


def velocity_project(ux, uy, dx, mask=None, iterations=20):
    """Project (ux, uy) onto its divergence-free part (Helmholtz decomposition)."""
    div = dx * (torch.roll(ux, -1, 0) - torch.roll(ux, 1, 0) +
                torch.roll(uy, -1, 1) - torch.roll(uy, 1, 1)) / 2
    div = set_bnd(0, div, mask)

    p = torch.zeros_like(div)
    for _ in range(iterations):
        p = (div + torch.roll(p, 1, 0) + torch.roll(p, -1, 0) +
             torch.roll(p, -1, 1) + torch.roll(p, 1, 1)) / 4
        p = set_bnd(0, p, mask)

    ux = ux - 0.5 * (torch.roll(p, 1, 0) - torch.roll(p, -1, 0)) / dx
    uy = uy - 0.5 * (torch.roll(p, 1, 1) - torch.roll(p, -1, 1)) / dx
    ux = set_bnd(1, ux, mask)
    uy = set_bnd(2, uy, mask)
    return ux, uy


def vel_step(ux, uy, ux0, uy0, dt, dx, visc, mask=None):
    ux = add_sources(ux, ux0, dt)
    uy = add_sources(uy, uy0, dt)
    ux, ux0 = swap(ux, ux0)
    uy, uy0 = swap(uy, uy0)
    ux = diffusion(ux, ux0, visc, dt, b=1, mask=mask)
    uy = diffusion(uy, uy0, visc, dt, b=2, mask=mask)
    ux, uy = velocity_project(ux, uy, dx, mask=mask)
    ux, ux0 = swap(ux, ux0)
    uy, uy0 = swap(uy, uy0)
    ux = advection(ux, ux0, ux0, uy0, dt, b=1, mask=mask)
    uy = advection(uy, uy0, ux0, uy0, dt, b=2, mask=mask)
    ux, uy = velocity_project(ux, uy, dx, mask=mask)
    return ux, uy


def density_step(rho, ux, uy, rho0, diff, dt, mask=None):
    rho = add_sources(rho, rho0, dt)
    rho = diffusion(rho, rho0, diff, dt, b=0, mask=mask)
    rho = advection(rho, rho0, ux, uy, dt, b=0, mask=mask)
    return rho


def inject_source_block(rho, ux, uy, cx, cy, size=5, rho_0=1.0, ux_0=0.0, uy_0=0.0):
    """Add density/velocity to a size x size square block at (cx, cy)."""
    rho = rho.clone()
    ux = ux.clone()
    uy = uy.clone()
    rho[cx:cx + size, cy:cy + size] += rho_0
    ux[cx:cx + size, cy:cy + size] += ux_0
    uy[cx:cx + size, cy:cy + size] += uy_0
    return rho, ux, uy


def inject_source_gaussian(rho, ux, uy, cx, cy, rho_0=1.0, ux_0=0.0, uy_0=0.0, xspread=1.0, yspread=1.0):
    """Add a smooth Gaussian-blob source of density/velocity centered at (cx, cy)."""
    N = rho.shape[0]
    x = torch.arange(N, dtype=rho.dtype, device=rho.device).view(-1, 1)
    y = torch.arange(N, dtype=rho.dtype, device=rho.device).view(1, -1)
    gaussian = torch.exp(-(xspread * (x - cx) ** 2 + yspread * (y - cy) ** 2) / (2 * (N // 16) ** 2))
    rho = rho + rho_0 * gaussian
    ux = ux + ux_0 * gaussian
    uy = uy + uy_0 * gaussian
    return rho, ux, uy


def circle_mask(N, center, radius):
    """Boolean fluid mask (True = fluid) that is False inside a circle of `radius` at `center`."""
    x = torch.arange(N).view(-1, 1)
    y = torch.arange(N).view(1, -1)
    return ~((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2)
