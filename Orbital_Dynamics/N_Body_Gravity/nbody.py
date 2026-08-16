"""Shared N-body gravity engine: three interchangeable acceleration solvers
(naive pairwise, Barnes-Hut, FMM) behind the same signature
accel_fn(pos, mass, G, softening) -> (N, 2) accelerations, plus a
solver-agnostic leapfrog integrator and energy/angular-momentum
diagnostics. Positions/velocities are (N, 2) arrays; gravity is the
ordinary softened inverse-square law, not the 2D log-potential, so that
two-body orbits stay closed ellipses (Bertrand's theorem) and results are
directly comparable to the legacy notebooks.

Barnes-Hut and FMM's tree/interaction-list walks are numba-jitted: plain
Python recursion/dict lookups turned out to have a far larger constant
factor than vectorized numpy pairwise summation, so despite better
asymptotic complexity neither solver actually beat pairwise_accel in wall
-clock time until N~5000+ (see N_Body_Gravity_Plan.md's "Performance
optimization" section). Compiling the hot loops removes that overhead
without changing either algorithm.
"""
import numpy as np
from numba import njit


def pairwise_accel(pos, mass, G=1.0, softening=1e-3):
    """Direct O(N^2) summation via numpy broadcasting -- the reference
    solver every other method is validated against."""
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # diff[i,j] = pos[j]-pos[i]
    dist2 = np.sum(diff**2, axis=-1) + softening**2
    inv_dist3 = dist2**-1.5
    return G * np.sum(mass[np.newaxis, :, np.newaxis] * diff * inv_dist3[:, :, np.newaxis], axis=1)


def build_flat_quadtree(pos, mass):
    """Barnes-Hut quadtree (max 1 particle per leaf) as flat numpy arrays
    instead of a linked object tree, so the force walk below can be
    numba-jitted -- numba's nopython mode can't work with Python objects,
    dicts, or lists. Construction stays plain Python/numpy (it's O(N log N)
    node visits, not the O(N log N)-per-particle force walk, so it was
    never the bottleneck); only the expensive part is compiled.

    Since a leaf holds at most one particle, its monopole moment *is*
    that particle exactly -- no separate leaf-vs-internal-node force
    formula is needed, just a per-node `is_leaf` flag and (for leaves) the
    one particle's index, or -1 if the leaf is empty.

    Returns flat arrays indexed by node id: mass, com_x, com_y, half-size,
    a `is_leaf` bool array, a `particle` index array (leaves only, -1 if
    empty), and `children` ((n_nodes, 4) int array, -1 where absent).
    """
    N = len(mass)
    max_nodes = 4 * N + 4
    node_mass = np.zeros(max_nodes)
    node_com_x = np.zeros(max_nodes)
    node_com_y = np.zeros(max_nodes)
    node_half = np.zeros(max_nodes)
    node_is_leaf = np.zeros(max_nodes, dtype=np.bool_)
    node_particle = np.full(max_nodes, -1, dtype=np.int64)
    node_children = np.full((max_nodes, 4), -1, dtype=np.int64)
    counter = [0]

    def build(indices, cx, cy, half):
        idx = counter[0]
        counter[0] += 1
        node_half[idx] = half
        if len(indices) == 0:
            node_is_leaf[idx] = True
            return idx
        m = mass[indices]
        M = m.sum()
        com = (pos[indices] * m[:, None]).sum(axis=0) / M
        node_mass[idx], (node_com_x[idx], node_com_y[idx]) = M, com
        if len(indices) == 1 or half < 1e-12:
            node_is_leaf[idx] = True
            node_particle[idx] = indices[0] if len(indices) == 1 else -1
            return idx
        p = pos[indices]
        for c, (qx, qy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
            qcx, qcy, qhalf = cx + qx * half / 2, cy + qy * half / 2, half / 2
            mask = ((p[:, 0] < cx) if qx < 0 else (p[:, 0] >= cx)) & \
                   ((p[:, 1] < cy) if qy < 0 else (p[:, 1] >= cy))
            node_children[idx, c] = build(indices[mask], qcx, qcy, qhalf)
        return idx

    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 * 1.001 + 1e-12
    build(np.arange(N), cx, cy, half)
    n = counter[0]
    return (node_mass[:n], node_com_x[:n], node_com_y[:n], node_half[:n],
            node_is_leaf[:n], node_particle[:n], node_children[:n])


@njit(cache=True)
def _bh_walk(pos, node_mass, node_com_x, node_com_y, node_half, node_is_leaf, node_particle, node_children,
             theta, G, softening):
    N = pos.shape[0]
    accel = np.zeros((N, 2))
    stack = np.empty(256, dtype=np.int64)
    theta2 = theta * theta
    soft2 = softening * softening
    for i in range(N):
        xi, yi = pos[i, 0], pos[i, 1]
        ax, ay = 0.0, 0.0
        stack[0] = 0
        sp = 1
        while sp > 0:
            sp -= 1
            node = stack[sp]
            m = node_mass[node]
            if m == 0.0:
                continue
            if node_is_leaf[node]:
                if node_particle[node] == i:
                    continue
                dx, dy = node_com_x[node] - xi, node_com_y[node] - yi
                inv_r3 = (dx * dx + dy * dy + soft2) ** -1.5
                ax += G * m * dx * inv_r3
                ay += G * m * dy * inv_r3
                continue
            dx, dy = node_com_x[node] - xi, node_com_y[node] - yi
            dist2 = dx * dx + dy * dy + soft2
            size = 2.0 * node_half[node]
            if size * size < theta2 * dist2:
                inv_r3 = dist2 ** -1.5
                ax += G * m * dx * inv_r3
                ay += G * m * dy * inv_r3
            else:
                for c in range(4):
                    child = node_children[node, c]
                    if child != -1:
                        stack[sp] = child
                        sp += 1
        accel[i, 0], accel[i, 1] = ax, ay
    return accel


def barnes_hut_accel(pos, mass, G=1.0, softening=1e-3, theta=0.5):
    """Barnes-Hut tree gravity: O(N log N) instead of pairwise_accel's
    O(N^2), approximate for a distant cluster of particles as a single
    mass at its center of mass whenever the cell looks small enough from
    the target's point of view (cell_size / distance < theta). Tree build
    is plain Python/numpy; the force walk is numba-jitted."""
    tree = build_flat_quadtree(pos, mass)
    return _bh_walk(pos, *tree, theta, G, softening)


@njit(cache=True)
def _multipole_field_scalar(M, Qxx, Qxy, Qyy, rx, ry, G):
    """Same closed-form monopole+quadrupole field as before (see
    N_Body_Gravity_Plan.md for the sympy derivation), as plain scalars
    instead of numpy arrays so it can be called from numba-jitted loops."""
    r2 = rx * rx + ry * ry
    sqrt_r2 = np.sqrt(r2)
    r3, r5, r7, r9 = r2 * sqrt_r2, r2 * r2 * sqrt_r2, r2**3 * sqrt_r2, r2**4 * sqrt_r2

    a0x = -G * M * rx / r3
    a0y = -G * M * ry / r3
    Hxx = -G * M * (2 * rx * rx - ry * ry) / r5
    Hxy = -3 * G * M * rx * ry / r5
    Hyy = G * M * (rx * rx - 2 * ry * ry) / r5

    Qr2 = Qxx * rx * rx + 2 * Qxy * rx * ry + Qyy * ry * ry
    a0x += G * (-5 * rx * Qr2 + 2 * r2 * (Qxx * rx + Qxy * ry)) / (2 * r7)
    a0y += G * (-5 * ry * Qr2 + 2 * r2 * (Qxy * rx + Qyy * ry)) / (2 * r7)
    Hxx += -G * (2 * Qxx * r2 * r2 - 20 * rx * r2 * (Qxx * rx + Qxy * ry) + 5 * (6 * rx * rx - ry * ry) * Qr2) / (2 * r9)
    Hxy += -G * (2 * Qxy * r2 * r2 + 35 * rx * ry * Qr2
                 - 10 * r2 * (rx * (Qxy * rx + Qyy * ry) + ry * (Qxx * rx + Qxy * ry))) / (2 * r9)
    Hyy += G * (-2 * Qyy * r2 * r2 + 20 * ry * r2 * (Qxy * rx + Qyy * ry) + 5 * (rx * rx - 6 * ry * ry) * Qr2) / (2 * r9)
    return a0x, a0y, Hxx, Hxy, Hyy


@njit(cache=True)
def _m2l_l2l_level(n_cells, R, mass, comx, comy, Qxx, Qxy, Qyy,
                    parent_n_cells, parent_comx, parent_comy,
                    parent_a0x, parent_a0y, parent_Hxx, parent_Hxy, parent_Hyy, G):
    """One level's M2L (interaction-list contributions) + L2L (shifted-down
    inheritance from the parent's already-accumulated local expansion),
    jitted over explicit (i,j) grid loops instead of dict lookups -- the
    same algorithm as the original dict-based version, just compiled."""
    a0x = np.zeros((n_cells, n_cells))
    a0y = np.zeros((n_cells, n_cells))
    Hxx = np.zeros((n_cells, n_cells))
    Hxy = np.zeros((n_cells, n_cells))
    Hyy = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            pi, pj = i // 2, j // 2
            ax, ay, hxx, hxy, hyy = 0.0, 0.0, 0.0, 0.0, 0.0
            for a in range(-R, R + 1):
                npi = pi + a
                if npi < 0 or npi >= parent_n_cells:
                    continue
                for b in range(-R, R + 1):
                    npj = pj + b
                    if npj < 0 or npj >= parent_n_cells:
                        continue
                    for dx in range(2):
                        ci = 2 * npi + dx
                        if ci < 0 or ci >= n_cells:
                            continue
                        for dy in range(2):
                            cj = 2 * npj + dy
                            if cj < 0 or cj >= n_cells:
                                continue
                            if abs(ci - i) <= R and abs(cj - j) <= R:
                                continue
                            m = mass[ci, cj]
                            if m == 0.0:
                                continue
                            rx = comx[i, j] - comx[ci, cj]
                            ry = comy[i, j] - comy[ci, cj]
                            dax, day, dhxx, dhxy, dhyy = _multipole_field_scalar(
                                m, Qxx[ci, cj], Qxy[ci, cj], Qyy[ci, cj], rx, ry, G)
                            ax += dax
                            ay += day
                            hxx += dhxx
                            hxy += dhxy
                            hyy += dhyy
            dx = comx[i, j] - parent_comx[pi, pj]
            dy = comy[i, j] - parent_comy[pi, pj]
            ax += parent_a0x[pi, pj] - (parent_Hxx[pi, pj] * dx + parent_Hxy[pi, pj] * dy)
            ay += parent_a0y[pi, pj] - (parent_Hxy[pi, pj] * dx + parent_Hyy[pi, pj] * dy)
            hxx += parent_Hxx[pi, pj]
            hxy += parent_Hxy[pi, pj]
            hyy += parent_Hyy[pi, pj]
            a0x[i, j], a0y[i, j] = ax, ay
            Hxx[i, j], Hxy[i, j], Hyy[i, j] = hxx, hxy, hyy
    return a0x, a0y, Hxx, Hxy, Hyy


@njit(cache=True)
def _fmm_near_and_l2p(pos_sorted, mass_sorted, cell_i, cell_j, cell_start, cell_count, n_cells, R, G, softening,
                       leaf_a0x, leaf_a0y, leaf_Hxx, leaf_Hxy, leaf_Hyy, leaf_comx, leaf_comy):
    """L2P (evaluate each leaf's accumulated local expansion at its member
    particles) + direct near-field summation over neighboring leaves
    (within R), using a sorted-by-cell particle order and a dense
    (start, count) index per cell -- the standard "cell list" trick from
    molecular dynamics, giving O(1) neighbor lookup instead of a dict."""
    N = pos_sorted.shape[0]
    accel = np.zeros((N, 2))
    soft2 = softening * softening
    for idx in range(N):
        i, j = cell_i[idx], cell_j[idx]
        xi, yi = pos_sorted[idx, 0], pos_sorted[idx, 1]
        dx, dy = xi - leaf_comx[i, j], yi - leaf_comy[i, j]
        ax = leaf_a0x[i, j] - (leaf_Hxx[i, j] * dx + leaf_Hxy[i, j] * dy)
        ay = leaf_a0y[i, j] - (leaf_Hxy[i, j] * dx + leaf_Hyy[i, j] * dy)
        for di in range(-R, R + 1):
            ni = i + di
            if ni < 0 or ni >= n_cells:
                continue
            for dj in range(-R, R + 1):
                nj = j + dj
                if nj < 0 or nj >= n_cells:
                    continue
                cell_id = ni * n_cells + nj
                cnt = cell_count[cell_id]
                if cnt == 0:
                    continue
                start = cell_start[cell_id]
                for k in range(start, start + cnt):
                    if k == idx:
                        continue
                    dxp, dyp = pos_sorted[k, 0] - xi, pos_sorted[k, 1] - yi
                    inv_r3 = (dxp * dxp + dyp * dyp + soft2) ** -1.5
                    ax += G * mass_sorted[k] * dxp * inv_r3
                    ay += G * mass_sorted[k] * dyp * inv_r3
        accel[idx, 0], accel[idx, 1] = ax, ay
    return accel


def fmm_accel(pos, mass, G=1.0, softening=1e-3, levels=4, R=2):
    """Fast Multipole Method: a uniform 2**levels x 2**levels grid of
    cells built bottom-up (P2M at the leaves, M2M shifting monopole +
    quadrupole moments up to each parent's own center of mass), then a
    top-down pass building each cell's *local* expansion (a constant +
    linear-in-position acceleration term, i.e. potential truncated at the
    Hessian) from its interaction list -- cells that are within radius `R`
    of its parent but not within radius `R` of the cell itself (M2L), plus
    whatever its parent already accumulated (L2L). Every pair of particles
    is covered exactly once: near cells (within R) by direct summation at
    the leaves, everything else through exactly one level's M2L
    interaction list.

    Cells are stored as dense per-level numpy arrays (a uniform grid has a
    fixed 2**l x 2**l cell count at level l, so this needs no dict/hash
    lookup at all): P2M and M2M are then plain vectorized numpy (P2M via
    `np.bincount` scatter-sums, M2M via reshape-and-pool -- a parent's 4
    children are exactly a contiguous 2x2 block once cells are laid out on
    a grid), and the M2L/near-field passes -- the parts whose cost scales
    with the interaction-list size times the cell count -- are numba
    -jitted instead of walking Python dicts.

    `R` (the "well-separated" buffer, in cells) must be applied
    identically when searching the parent's neighbors and when excluding
    the target's own neighbors -- using a *smaller* search radius than
    exclusion radius silently drops interactions (some cell pairs are
    excluded from level l's interaction list as "too near" without ever
    having been reachable as a level-l candidate in the first place),
    undercounting mass rather than merely approximating it, which is far
    worse than the underlying multipole truncation error. The classic
    minimal choice R=1 also leaves the *closest* interaction-list pairs
    only about one cell-width apart -- comparable to the cell size itself,
    a marginal separation for a quadrupole-truncated expansion (mean error
    ~2%, worst-case individual-particle error up to ~40% in testing). R=2
    guarantees at least a two-cell-width gap and was ~4-5x more accurate
    in practice (mean error ~0.5%, worst case ~10%), at the cost of a
    larger (but still level-independent, so still O(N)) near-field block.
    """
    N = len(mass)
    n_leaf = 2**levels
    n_leaf_cells = n_leaf * n_leaf
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 * 1.001 + 1e-9
    box_min_x, box_min_y = cx - half, cy - half
    leaf_size = (2 * half) / n_leaf

    i_idx = np.clip(((pos[:, 0] - box_min_x) / leaf_size).astype(np.int64), 0, n_leaf - 1)
    j_idx = np.clip(((pos[:, 1] - box_min_y) / leaf_size).astype(np.int64), 0, n_leaf - 1)
    lin = i_idx * n_leaf + j_idx

    # P2M (vectorized): scatter-sum mass/COM/quadrupole moments per leaf cell
    mass_leaf = np.bincount(lin, weights=mass, minlength=n_leaf_cells).reshape(n_leaf, n_leaf)
    safe_mass = np.where(mass_leaf == 0, 1.0, mass_leaf)
    comx_leaf = np.bincount(lin, weights=mass * pos[:, 0], minlength=n_leaf_cells).reshape(n_leaf, n_leaf) / safe_mass
    comy_leaf = np.bincount(lin, weights=mass * pos[:, 1], minlength=n_leaf_cells).reshape(n_leaf, n_leaf) / safe_mass

    dx = pos[:, 0] - comx_leaf[i_idx, j_idx]
    dy = pos[:, 1] - comy_leaf[i_idx, j_idx]
    r2 = dx * dx + dy * dy
    Qxx_leaf = np.bincount(lin, weights=mass * (3 * dx * dx - r2), minlength=n_leaf_cells).reshape(n_leaf, n_leaf)
    Qyy_leaf = np.bincount(lin, weights=mass * (3 * dy * dy - r2), minlength=n_leaf_cells).reshape(n_leaf, n_leaf)
    Qxy_leaf = np.bincount(lin, weights=mass * 3 * dx * dy, minlength=n_leaf_cells).reshape(n_leaf, n_leaf)

    # M2M (vectorized reshape-pooling): a parent's 4 children are always a
    # contiguous 2x2 block on this grid, so `.reshape(n, 2, n, 2)` groups
    # exactly the right cells without any explicit indexing per parent.
    level_mass = [None] * (levels + 1)
    level_comx = [None] * (levels + 1)
    level_comy = [None] * (levels + 1)
    level_Qxx = [None] * (levels + 1)
    level_Qxy = [None] * (levels + 1)
    level_Qyy = [None] * (levels + 1)
    level_mass[levels], level_comx[levels], level_comy[levels] = mass_leaf, comx_leaf, comy_leaf
    level_Qxx[levels], level_Qxy[levels], level_Qyy[levels] = Qxx_leaf, Qxy_leaf, Qyy_leaf

    for l in range(levels - 1, -1, -1):
        cm, ccx, ccy = level_mass[l + 1], level_comx[l + 1], level_comy[l + 1]
        cqxx, cqxy, cqyy = level_Qxx[l + 1], level_Qxy[l + 1], level_Qyy[l + 1]
        n = cm.shape[0] // 2
        cm_r = cm.reshape(n, 2, n, 2)
        pm = cm_r.sum(axis=(1, 3))
        safe_pm = np.where(pm == 0, 1.0, pm)
        pcx = (cm_r * ccx.reshape(n, 2, n, 2)).sum(axis=(1, 3)) / safe_pm
        pcy = (cm_r * ccy.reshape(n, 2, n, 2)).sum(axis=(1, 3)) / safe_pm
        ddx = ccx - np.repeat(np.repeat(pcx, 2, axis=0), 2, axis=1)
        ddy = ccy - np.repeat(np.repeat(pcy, 2, axis=0), 2, axis=1)
        dd2 = ddx * ddx + ddy * ddy
        pqxx = (cqxx + cm * (3 * ddx * ddx - dd2)).reshape(n, 2, n, 2).sum(axis=(1, 3))
        pqyy = (cqyy + cm * (3 * ddy * ddy - dd2)).reshape(n, 2, n, 2).sum(axis=(1, 3))
        pqxy = (cqxy + cm * 3 * ddx * ddy).reshape(n, 2, n, 2).sum(axis=(1, 3))
        level_mass[l], level_comx[l], level_comy[l] = pm, pcx, pcy
        level_Qxx[l], level_Qxy[l], level_Qyy[l] = pqxx, pqxy, pqyy

    # M2L + L2L, level by level (levels 0-1 have no local expansion of
    # their own -- nothing is ever well-separated from the 1-2 cell root)
    local_a0x = [np.zeros((1, 1)), np.zeros((2, 2))] + [None] * (levels - 1)
    local_a0y = [np.zeros((1, 1)), np.zeros((2, 2))] + [None] * (levels - 1)
    local_Hxx = [np.zeros((1, 1)), np.zeros((2, 2))] + [None] * (levels - 1)
    local_Hxy = [np.zeros((1, 1)), np.zeros((2, 2))] + [None] * (levels - 1)
    local_Hyy = [np.zeros((1, 1)), np.zeros((2, 2))] + [None] * (levels - 1)
    for l in range(2, levels + 1):
        (local_a0x[l], local_a0y[l], local_Hxx[l], local_Hxy[l], local_Hyy[l]) = _m2l_l2l_level(
            2**l, R, level_mass[l], level_comx[l], level_comy[l], level_Qxx[l], level_Qxy[l], level_Qyy[l],
            2**(l - 1), level_comx[l - 1], level_comy[l - 1],
            local_a0x[l - 1], local_a0y[l - 1], local_Hxx[l - 1], local_Hxy[l - 1], local_Hyy[l - 1], G)

    # L2P + near field, via a sorted-by-cell particle order (cell list)
    order = np.argsort(lin, kind='stable')
    lin_sorted = lin[order]
    cell_start = np.zeros(n_leaf_cells, dtype=np.int64)
    cell_count = np.zeros(n_leaf_cells, dtype=np.int64)
    uniq, starts, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    cell_start[uniq], cell_count[uniq] = starts, counts

    accel_sorted = _fmm_near_and_l2p(
        pos[order], mass[order], i_idx[order], j_idx[order], cell_start, cell_count, n_leaf, R, G, softening,
        local_a0x[levels], local_a0y[levels], local_Hxx[levels], local_Hxy[levels], local_Hyy[levels],
        comx_leaf, comy_leaf)
    accel = np.empty((N, 2))
    accel[order] = accel_sorted
    return accel


def leapfrog_step(pos, vel, mass, dt, accel_fn, G=1.0, softening=1e-3):
    """Kick-drift-kick symplectic step; accel_fn is any of the solvers
    below (or pairwise_accel), all sharing this same call signature."""
    acc = accel_fn(pos, mass, G, softening)
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = accel_fn(pos_new, mass, G, softening)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new


def integrate(pos0, vel0, mass, dt, n_steps, accel_fn, G=1.0, softening=1e-3):
    """Runs n_steps of leapfrog_step, returning (n_steps+1, N, 2) position
    and velocity trajectories (including the initial condition)."""
    N = pos0.shape[0]
    pos_t = np.zeros((n_steps + 1, N, 2))
    vel_t = np.zeros((n_steps + 1, N, 2))
    pos_t[0], vel_t[0] = pos0, vel0
    pos, vel = pos0.copy(), vel0.copy()
    for i in range(1, n_steps + 1):
        pos, vel = leapfrog_step(pos, vel, mass, dt, accel_fn, G, softening)
        pos_t[i], vel_t[i] = pos, vel
    return pos_t, vel_t


def energy(pos, vel, mass, G=1.0, softening=1e-3):
    """Total kinetic + softened potential energy (a single snapshot)."""
    KE = 0.5 * np.sum(mass * np.sum(vel**2, axis=-1))
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + softening**2)
    m_outer = mass[:, np.newaxis] * mass[np.newaxis, :]
    iu = np.triu_indices(len(mass), k=1)
    PE = -G * np.sum(m_outer[iu] / dist[iu])
    return KE + PE


def angular_momentum(pos, vel, mass):
    """Total z-component of angular momentum (a single snapshot)."""
    return np.sum(mass * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
