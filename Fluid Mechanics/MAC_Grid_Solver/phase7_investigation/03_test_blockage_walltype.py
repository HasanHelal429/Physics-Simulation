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
CELLS_PER_D = 10  # coarse/fast -- just checking qualitative onset behavior


def run(blockage, wall_type, T_total, upstream_D=5, downstream_D=15, label=""):
    H = D / blockage
    L = (upstream_D + downstream_D) * D
    dx = D / CELLS_PER_D
    nx, ny = int(round(L / dx)), int(round(H / dx))
    offset = 0.02 * D
    cyl_center = (upstream_D * D, H / 2 + offset)
    mask = g.circle_obstacle(nx, ny, L, H, cyl_center, D / 2)
    grid_c = g.make_grid(nx, ny, L, H, cell_mask=mask)
    bc = {'left': {'type': 'inflow', 'value': U_inf}, 'right': {'type': 'outflow'},
          'bottom': {'type': wall_type}, 'top': {'type': wall_type}}
    system = sv.build_system(grid_c, bc)
    u = np.full((nx + 1, ny), U_inf)
    v = np.zeros((nx, ny + 1))
    u, v = bnd.apply_mask(u, v, grid_c.u_mask, grid_c.v_mask)
    probe_i = int(round((cyl_center[0] + 4 * D) / grid_c.dx))
    probe_j = ny // 2

    print(f"\n=== {label}: blockage={blockage:.3f} (H={H:.1f}D), wall={wall_type}, grid={nx}x{ny} ===", flush=True)
    t, step = 0.0, 0
    window = []
    t0 = time.time()
    while t < T_total:
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


run(blockage=0.125, wall_type='free_slip', T_total=80, label="A: current setup (high blockage, free-slip)")
run(blockage=0.03, wall_type='free_slip', T_total=80, label="B: low blockage, free-slip")
run(blockage=0.125, wall_type='no_slip', T_total=80, label="C: current blockage, no-slip walls")

print("\nDONE", flush=True)
