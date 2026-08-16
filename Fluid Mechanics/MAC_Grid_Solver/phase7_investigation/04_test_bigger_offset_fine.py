import sys, time
sys.path.insert(0, r"C:\Users\Hasan's Laptop\OneDrive\Physics Simulations\Fluid Mechanics\MAC_Grid_Solver")
import numpy as np
import grid as g
import boundary as bnd
import solver as sv

D = 1.0
Re = 100.0
U_inf = 1.0
nu = U_inf * D / Re
BLOCKAGE = 0.125
UPSTREAM_D, DOWNSTREAM_D = 5, 15
CELLS_PER_D = 25

H = D / BLOCKAGE
L = (UPSTREAM_D + DOWNSTREAM_D) * D
dx_target = D / CELLS_PER_D
nx, ny = int(round(L / dx_target)), int(round(H / dx_target))

for offset_frac in (0.1, 0.2):
    offset = offset_frac * D
    cyl_center = (UPSTREAM_D * D, H / 2 + offset)
    mask = g.circle_obstacle(nx, ny, L, H, cyl_center, D / 2)
    grid_c = g.make_grid(nx, ny, L, H, cell_mask=mask)
    bc = {'left': {'type': 'inflow', 'value': U_inf}, 'right': {'type': 'outflow'},
          'bottom': {'type': 'free_slip'}, 'top': {'type': 'free_slip'}}
    system = sv.build_system(grid_c, bc)
    u = np.full((nx + 1, ny), U_inf)
    v = np.zeros((nx, ny + 1))
    u, v = bnd.apply_mask(u, v, grid_c.u_mask, grid_c.v_mask)
    probe_i = int(round((cyl_center[0] + 4 * D) / grid_c.dx))
    probe_j = ny // 2

    print(f"\n=== offset={offset_frac}D ({offset/grid_c.dx:.1f} cells), grid={nx}x{ny} ===", flush=True)
    t, step = 0.0, 0
    window = []
    t0 = time.time()
    while t < 60.0:
        dt = sv.cfl_timestep(u, v, grid_c.dx, grid_c.dy, cfl_number=0.4)
        u, v, p, max_div = sv.step(u, v, grid_c, bc, system, dt, nu)
        t += dt
        step += 1
        window.append(abs(v[probe_i, probe_j]))
        if len(window) > 200:
            window.pop(0)
        if step % 400 == 0:
            print(f"  step {step}: t={t:.1f} envelope={max(window):.4e} max|div|={max_div:.2e}", flush=True)
    t1 = time.time()
    print(f"  {step} steps in {t1-t0:.1f}s", flush=True)

print("\nDONE", flush=True)
