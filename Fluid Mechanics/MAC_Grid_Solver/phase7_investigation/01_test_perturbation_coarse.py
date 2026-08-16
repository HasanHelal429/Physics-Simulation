import sys, time
sys.path.insert(0, r"C:\Users\Hasan's Laptop\OneDrive\Physics Simulations\Fluid Mechanics\MAC_Grid_Solver")
import numpy as np
import grid as g
import boundary as bnd
import solver as sv
import operators as op

D = 1.0
Re = 100.0
U_inf = 1.0
nu = U_inf * D / Re

BLOCKAGE = 0.125
UPSTREAM_D = 5
DOWNSTREAM_D = 10   # shortened for this test only
CELLS_PER_D = 10    # coarser for this test only

H = D / BLOCKAGE
L = (UPSTREAM_D + DOWNSTREAM_D) * D
dx_target = D / CELLS_PER_D
nx = int(round(L / dx_target))
ny = int(round(H / dx_target))

# Key change: offset the cylinder 2% of D off the channel centerline to
# break the exact top-bottom symmetry that was suppressing shedding
# (confirmed: with the cylinder dead-center, the wake stayed a symmetric
# steady double shear layer, and the "shedding" signal was an
# unsaturated linear instability growing off ~1e-12 floating-point noise).
offset = 0.02 * D
cyl_center = (UPSTREAM_D * D, H / 2 + offset)
cyl_radius = D / 2

mask = g.circle_obstacle(nx, ny, L, H, cyl_center, cyl_radius)
grid_c = g.make_grid(nx, ny, L, H, cell_mask=mask)
print(f"grid: {nx}x{ny}={nx*ny} cells, dx={grid_c.dx:.4f}, cylinder offset={offset}")

bc = {'left': {'type': 'inflow', 'value': U_inf}, 'right': {'type': 'outflow'},
      'bottom': {'type': 'free_slip'}, 'top': {'type': 'free_slip'}}
system = sv.build_system(grid_c, bc)

u = np.full((nx + 1, ny), U_inf)
v = np.zeros((nx, ny + 1))
u, v = bnd.apply_mask(u, v, grid_c.u_mask, grid_c.v_mask)

probe_i = int(round((cyl_center[0] + 4 * D) / grid_c.dx))
probe_j = ny // 2

T_TOTAL = 100.0
t = 0.0
step = 0
t0 = time.time()
probe_history = []
while t < T_TOTAL:
    dt = sv.cfl_timestep(u, v, grid_c.dx, grid_c.dy, cfl_number=0.4)
    u, v, p, max_div = sv.step(u, v, grid_c, bc, system, dt, nu)
    t += dt
    step += 1
    probe_history.append((t, v[probe_i, probe_j]))
    if step % 300 == 0:
        recent = [abs(x[1]) for x in probe_history[-100:]]
        print(f"step {step}: t={t:.1f} max|div|={max_div:.2e} "
              f"probe|v| recent max={max(recent):.4e}")
t1 = time.time()
print(f"\nTotal: {step} steps, {t1-t0:.1f}s ({(t1-t0)/step*1000:.1f} ms/step)")

vals = np.array([abs(x[1]) for x in probe_history])
print(f"probe |v| amplitude: first 10% mean={vals[:len(vals)//10].mean():.4e}, "
      f"last 10% mean={vals[-len(vals)//10:].mean():.4e}")
