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
from numba import njit, prange


def pairwise_accel(pos, mass, G=1.0, softening=1e-3):
    """Direct O(N^2) summation via numpy broadcasting -- the reference
    solver every other method is validated against."""
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # diff[i,j] = pos[j]-pos[i]
    dist2 = np.sum(diff**2, axis=-1) + softening**2
    inv_dist3 = dist2**-1.5
    return G * np.sum(mass[np.newaxis, :, np.newaxis] * diff * inv_dist3[:, :, np.newaxis], axis=1)


@njit(cache=True)
def _build_tree_numba(pos, mass, max_nodes, min_half=1e-12):
    """Non-recursive, in-place-partitioning quadtree builder (max 1
    particle/leaf, or more if `min_half` is reached first -- see
    `node_start`/`node_count`/`order` below): a plain-Python recursive
    builder (this function replaces) turned out to dominate wall-clock
    time for *both* Barnes-Hut and adaptive FMM once their force-walk
    steps were themselves numba-jitted (profiling: ~84ms/5000 particles,
    unrelated to which solver used the tree -- construction was shared,
    and simply hadn't been the bottleneck until the walk got fast).

    The cost driver was numpy fancy-indexing (`indices[mask]`), which
    allocates a new array at every one of the ~3*N tree nodes. This
    builder instead partitions a single shared `order` array in place via
    a counting sort at each node (like quicksort's partition step) and
    pushes child (start, count) ranges onto an explicit stack instead of
    recursing -- both numba-nopython-friendly and allocation-free per
    node. Measured 45-60x faster than the old recursive builder for
    identical tree structure and moments (same node count, same
    accelerations through the same downstream walk, verified bit-for-bit
    modulo floating-point-order differences).

    Returns flat arrays indexed by node id: mass, com_x, com_y, half-size,
    `is_leaf`, `particle` (single-particle leaves only, -1 otherwise),
    `children` ((n,4), -1 where absent), `parent` (-1 for the root),
    `start`/`count` (this node's particles are `order[start:start+count]`
    -- for a normal 1-particle leaf that's just `[particle]`; for a
    degenerate multi-particle leaf, the *whole* group, with no separate
    side-table needed), `order` (the partitioned particle-index array
    itself), and quadrupole moments `Qxx`/`Qxy`/`Qyy` (needed by adaptive
    FMM, unused by Barnes-Hut).
    """
    N = pos.shape[0]
    order = np.arange(N)

    node_mass = np.zeros(max_nodes)
    node_com_x = np.zeros(max_nodes)
    node_com_y = np.zeros(max_nodes)
    node_half = np.zeros(max_nodes)
    node_is_leaf = np.zeros(max_nodes, dtype=np.bool_)
    node_particle = np.full(max_nodes, -1, dtype=np.int64)
    node_children = np.full((max_nodes, 4), -1, dtype=np.int64)
    node_parent = np.full(max_nodes, -1, dtype=np.int64)
    node_start = np.zeros(max_nodes, dtype=np.int64)
    node_count = np.zeros(max_nodes, dtype=np.int64)
    node_cx = np.zeros(max_nodes)
    node_cy = np.zeros(max_nodes)

    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    root_cx, root_cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    root_half = max(x_max - x_min, y_max - y_min) / 2 * 1.001 + 1e-12

    node_start[0] = 0
    node_count[0] = N
    node_cx[0] = root_cx
    node_cy[0] = root_cy
    node_half[0] = root_half
    n_nodes = 1

    stack = np.empty(max_nodes, dtype=np.int64)
    stack[0] = 0
    sp = 1

    temp = np.empty(N, dtype=np.int64)
    quad = np.empty(N, dtype=np.int64)

    while sp > 0:
        sp -= 1
        node = stack[sp]
        start, count = node_start[node], node_count[node]
        cx, cy, half = node_cx[node], node_cy[node], node_half[node]

        if count == 0:
            node_is_leaf[node] = True
            continue
        if count == 1:
            node_is_leaf[node] = True
            node_particle[node] = order[start]
            continue
        if half < min_half:
            node_is_leaf[node] = True
            continue

        counts = np.zeros(4, dtype=np.int64)
        for k in range(count):
            p = order[start + k]
            qx = 0 if pos[p, 0] < cx else 1
            qy = 0 if pos[p, 1] < cy else 1
            q = qx + 2 * qy
            quad[start + k] = q
            counts[q] += 1

        offsets = np.zeros(5, dtype=np.int64)
        for c in range(4):
            offsets[c + 1] = offsets[c] + counts[c]
        fill = offsets.copy()
        for k in range(count):
            q = quad[start + k]
            temp[fill[q]] = order[start + k]
            fill[q] += 1
        for k in range(count):
            order[start + k] = temp[k]

        for c in range(4):
            child_count = counts[c]
            child_idx = n_nodes
            n_nodes += 1
            node_start[child_idx] = start + offsets[c]
            node_count[child_idx] = child_count
            qx = c % 2
            qy = c // 2
            node_cx[child_idx] = cx + (2 * qx - 1) * half / 2
            node_cy[child_idx] = cy + (2 * qy - 1) * half / 2
            node_half[child_idx] = half / 2
            node_parent[child_idx] = node
            node_children[node, c] = child_idx
            stack[sp] = child_idx
            sp += 1

    node_Qxx = np.zeros(max_nodes)
    node_Qxy = np.zeros(max_nodes)
    node_Qyy = np.zeros(max_nodes)

    # Bottom-up mass/COM/quadrupole in one pass: decreasing node index is
    # guaranteed child-before-parent (children always get larger ids than
    # their parent, by construction order above), so every child a node
    # needs is already fully finalized by the time that node's own turn
    # comes up in this same loop.
    for node in range(n_nodes - 1, -1, -1):
        if node_is_leaf[node]:
            p = node_particle[node]
            if p != -1:
                node_mass[node] = mass[p]
                node_com_x[node] = pos[p, 0]
                node_com_y[node] = pos[p, 1]
                # a single point particle has zero quadrupole about itself
            elif node_count[node] > 1:
                start, count = node_start[node], node_count[node]
                m_sum, cxs, cys = 0.0, 0.0, 0.0
                for k in range(count):
                    pp = order[start + k]
                    m_sum += mass[pp]
                    cxs += mass[pp] * pos[pp, 0]
                    cys += mass[pp] * pos[pp, 1]
                node_mass[node] = m_sum
                if m_sum > 0:
                    node_com_x[node] = cxs / m_sum
                    node_com_y[node] = cys / m_sum
                    qxx, qxy, qyy = 0.0, 0.0, 0.0
                    for k in range(count):
                        pp = order[start + k]
                        dx = pos[pp, 0] - node_com_x[node]
                        dy = pos[pp, 1] - node_com_y[node]
                        r2 = dx * dx + dy * dy
                        qxx += mass[pp] * (3 * dx * dx - r2)
                        qxy += mass[pp] * 3 * dx * dy
                        qyy += mass[pp] * (3 * dy * dy - r2)
                    node_Qxx[node], node_Qxy[node], node_Qyy[node] = qxx, qxy, qyy
            continue
        M, cxs, cys = 0.0, 0.0, 0.0
        for c in range(4):
            child = node_children[node, c]
            if child == -1:
                continue
            cm = node_mass[child]
            if cm == 0.0:
                continue
            M += cm
            cxs += cm * node_com_x[child]
            cys += cm * node_com_y[child]
        node_mass[node] = M
        if M > 0:
            node_com_x[node] = cxs / M
            node_com_y[node] = cys / M
        qxx, qxy, qyy = 0.0, 0.0, 0.0
        for c in range(4):
            child = node_children[node, c]
            if child == -1:
                continue
            cm = node_mass[child]
            if cm == 0.0:
                continue
            dx = node_com_x[child] - node_com_x[node]
            dy = node_com_y[child] - node_com_y[node]
            d2 = dx * dx + dy * dy
            qxx += node_Qxx[child] + cm * (3 * dx * dx - d2)
            qxy += node_Qxy[child] + cm * 3 * dx * dy
            qyy += node_Qyy[child] + cm * (3 * dy * dy - d2)
        node_Qxx[node], node_Qxy[node], node_Qyy[node] = qxx, qxy, qyy

    return (node_mass[:n_nodes], node_com_x[:n_nodes], node_com_y[:n_nodes], node_half[:n_nodes],
            node_is_leaf[:n_nodes], node_particle[:n_nodes], node_children[:n_nodes],
            node_parent[:n_nodes], node_start[:n_nodes], node_count[:n_nodes], order,
            node_Qxx[:n_nodes], node_Qxy[:n_nodes], node_Qyy[:n_nodes])


def build_flat_quadtree(pos, mass):
    """Barnes-Hut quadtree as flat numpy arrays (see `_build_tree_numba`
    for the actual construction, which both this and `adaptive_fmm_accel`
    share). Kept as a thin wrapper for backward compatibility: returns
    just the 7 fields `_bh_walk` needs, in its original order.
    """
    N = len(mass)
    # See _build_tree_numba's docstring for why 8*N (not the tighter-
    # looking but not-actually-safe 4*N): two particles closer together
    # than typical precision can resolve force recursion all the way to
    # the `min_half` floor, each level allocating 4 node ids even though
    # 3 stay empty -- found via a genuine crash (IndexError) on an
    # unlucky small-N draw where two points landed very close together.
    max_nodes = 8 * N + 4 * 50
    tree = _build_tree_numba(pos, mass, max_nodes)
    return tree[:7]


@njit(cache=True, parallel=True)
def _bh_walk(pos, node_mass, node_com_x, node_com_y, node_half, node_is_leaf, node_particle, node_children,
             theta, G, softening):
    """Each particle's tree walk reads the shared (read-only) tree arrays
    and writes only its own accel[i] row -- embarrassingly parallel across
    particles, so this loop runs under prange across CPU cores. The scratch
    `stack` used to be allocated once outside the loop and reused every
    iteration; that's a data race once iterations run concurrently on
    different threads, so it's now allocated fresh inside the loop (one
    private copy per iteration) instead.
    """
    N = pos.shape[0]
    accel = np.zeros((N, 2))
    theta2 = theta * theta
    soft2 = softening * softening
    for i in prange(N):
        stack = np.empty(256, dtype=np.int64)
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


@njit(cache=True)
def _dual_tree_m2l(node_mass, node_comx, node_comy, node_half, node_is_leaf, node_particle, node_children,
                    node_Qxx, node_Qxy, node_Qyy, theta, G, max_near_pairs, min_sep2):
    """Adaptive FMM's M2L step via dual-tree traversal: instead of
    `fmm_accel`'s uniform grid of same-size cells (simple, but wastes
    depth on empty regions and under-resolves dense clusters), recurse
    over *pairs* of nodes from the same Barnes-Hut-style adaptive tree --
    one from the "target" side, one from the "source" side, starting both
    at the root -- splitting whichever side is coarser until each pair is
    either well-separated (accumulate an M2L contribution into the coarser
    side's local expansion) or both are leaves (record a near-field pair
    for direct softened summation instead).

    A target leaf can receive M2L contributions from many source nodes at
    different levels, so `local_*` are accumulated bottom-up per node here
    and then pushed the rest of the way down to individual leaves by
    `_l2l` afterward, exactly like the uniform-grid FMM's level-by-level
    L2L -- just walking a parent-pointer array instead of a fixed grid.
    """
    n_nodes = node_mass.shape[0]
    local_a0x = np.zeros(n_nodes)
    local_a0y = np.zeros(n_nodes)
    local_Hxx = np.zeros(n_nodes)
    local_Hxy = np.zeros(n_nodes)
    local_Hyy = np.zeros(n_nodes)

    # `max_near_pairs` (a caller-supplied `max_near_pairs_per_particle * N`
    # budget) is likewise not a rigorous bound -- a sufficiently dense
    # collapsed cluster can force far more leaf-leaf pairs into the
    # near-field path than that budget anticipates (same silent-corruption
    # risk as the traversal stack above, same fix: grow instead of guess).
    near_i = np.empty(max_near_pairs, dtype=np.int64)
    near_j = np.empty(max_near_pairs, dtype=np.int64)
    n_near = 0

    # `4*n_nodes+16` looks like a safe upper bound (each node's self-pair
    # split pushes at most 16 child pairs) but isn't one: a cross pair
    # that fails the MAC also recurses, and for a real post-collapse
    # snapshot (many particles converging through a near-common point
    # nearly simultaneously -- exactly the free-fall scenario this was
    # written for) that recursion can push far more node-pairs than that
    # bound anticipates. Numba nopython mode doesn't bounds-check array
    # writes, so overflowing this silently corrupted memory and segfaulted
    # instead of raising -- found by running the actual cold-collapse
    # trajectory, not a synthetic test. Fixed by growing the stack
    # (doubling, amortized O(1) like a dynamic array) whenever fewer than
    # 16 slots remain -- 16 covers the worst case of any single iteration
    # below (the self-pair split; every other branch pushes at most 4) --
    # instead of trying to guess a tighter bound that could still be wrong
    # for some future, even more extreme collapse.
    stack_t = np.empty(4 * n_nodes + 16, dtype=np.int64)
    stack_s = np.empty(4 * n_nodes + 16, dtype=np.int64)
    stack_t[0] = 0
    stack_s[0] = 0
    sp = 1
    theta2 = theta * theta

    while sp > 0:
        if sp + 16 > stack_t.shape[0]:
            new_cap = stack_t.shape[0] * 2
            grown_t = np.empty(new_cap, dtype=np.int64)
            grown_s = np.empty(new_cap, dtype=np.int64)
            grown_t[:sp] = stack_t[:sp]
            grown_s[:sp] = stack_s[:sp]
            stack_t, stack_s = grown_t, grown_s
        sp -= 1
        t, s = stack_t[sp], stack_s[sp]
        ms = node_mass[s]
        if ms == 0.0 or node_mass[t] == 0.0:
            continue

        if t == s:
            # a self-pair only needs splitting once -- (child_i, child_j)
            # for i != j is already handled as an ordinary (t, s) pair by
            # the next iteration, so this just seeds those cross pairs
            # plus each child's own self-pair.
            if node_is_leaf[t]:
                continue
            for ci in range(4):
                ti = node_children[t, ci]
                if ti == -1:
                    continue
                for cj in range(4):
                    si = node_children[t, cj]
                    if si == -1:
                        continue
                    stack_t[sp], stack_s[sp] = ti, si
                    sp += 1
            continue

        dx = node_comx[t] - node_comx[s]
        dy = node_comy[t] - node_comy[s]
        dist2 = dx * dx + dy * dy
        size_sum = 2.0 * node_half[t] + 2.0 * node_half[s]

        # a degenerate multi-particle leaf (node_particle == -1 despite
        # is_leaf) has no single valid particle index to use for a
        # near-field pair -- but it's also smaller than the tree's minimum
        # resolvable size, so treating it as a point mass via M2L for *any*
        # interaction (regardless of whether the theta test would normally
        # accept it) is an excellent approximation, not a hack.
        degenerate = node_is_leaf[t] and node_particle[t] == -1
        degenerate_s = node_is_leaf[s] and node_particle[s] == -1

        # The MAC (size/distance < theta) is scale-invariant: for a
        # pathologically clustered distribution the adaptive tree can
        # recurse to cell sizes and separations many orders of magnitude
        # below the softening length, where it's still happily "satisfied"
        # in a relative sense -- but the *unsoftened* multipole field
        # formula (1/r^7, 1/r^9 terms) is numerically catastrophic there
        # (found empirically: a dist^2 ~ 1e-18 pair blew up to a ~1e14
        # spurious acceleration). Requiring an absolute minimum separation
        # tied to the softening length -- below which softening already
        # regularizes the *direct* near-field formula anyway, so nothing
        # physical is lost by refusing the multipole shortcut there --
        # fixes it: such pairs fall through to the (safe, softened)
        # near-field path instead.
        well_separated = size_sum * size_sum < theta2 * dist2 and dist2 > min_sep2

        if degenerate or degenerate_s or well_separated:
            a0x, a0y, hxx, hxy, hyy = _multipole_field_scalar(ms, node_Qxx[s], node_Qxy[s], node_Qyy[s], dx, dy, G)
            local_a0x[t] += a0x
            local_a0y[t] += a0y
            local_Hxx[t] += hxx
            local_Hxy[t] += hxy
            local_Hyy[t] += hyy
            continue

        if node_is_leaf[t] and node_is_leaf[s]:
            # dual-tree traversal visits both (A,B) and (B,A) as separate
            # stack entries once a self-pair's children are split (needed
            # for M2L, since the two directions feed different local
            # expansions) -- but a near-field pair is symmetric, so only
            # recording it once (smaller particle index first) avoids
            # double-counting the force.
            pt, ps = node_particle[t], node_particle[s]
            if pt < ps:
                if n_near >= near_i.shape[0]:
                    new_cap = near_i.shape[0] * 2
                    grown_i = np.empty(new_cap, dtype=np.int64)
                    grown_j = np.empty(new_cap, dtype=np.int64)
                    grown_i[:n_near] = near_i[:n_near]
                    grown_j[:n_near] = near_j[:n_near]
                    near_i, near_j = grown_i, grown_j
                near_i[n_near] = pt
                near_j[n_near] = ps
                n_near += 1
            continue

        if node_is_leaf[t]:
            for cj in range(4):
                si = node_children[s, cj]
                if si != -1:
                    stack_t[sp], stack_s[sp] = t, si
                    sp += 1
        elif node_is_leaf[s]:
            for ci in range(4):
                ti = node_children[t, ci]
                if ti != -1:
                    stack_t[sp], stack_s[sp] = ti, s
                    sp += 1
        elif node_half[t] >= node_half[s]:
            for ci in range(4):
                ti = node_children[t, ci]
                if ti != -1:
                    stack_t[sp], stack_s[sp] = ti, s
                    sp += 1
        else:
            for cj in range(4):
                si = node_children[s, cj]
                if si != -1:
                    stack_t[sp], stack_s[sp] = t, si
                    sp += 1

    return local_a0x, local_a0y, local_Hxx, local_Hxy, local_Hyy, near_i[:n_near], near_j[:n_near]


@njit(cache=True)
def _l2l(node_parent, node_comx, node_comy, local_a0x, local_a0y, local_Hxx, local_Hxy, local_Hyy):
    """Top-down local-expansion propagation for the adaptive tree: since
    every child's node id is guaranteed larger than its parent's (an
    invariant of `_build_tree_numba`'s construction order), a single
    increasing-index sweep is already parent-before-child -- no explicit
    BFS/recursion needed, unlike a general tree."""
    n_nodes = node_parent.shape[0]
    for node in range(1, n_nodes):
        p = node_parent[node]
        if p == -1:
            continue
        dx = node_comx[node] - node_comx[p]
        dy = node_comy[node] - node_comy[p]
        local_a0x[node] += local_a0x[p] - (local_Hxx[p] * dx + local_Hxy[p] * dy)
        local_a0y[node] += local_a0y[p] - (local_Hxy[p] * dx + local_Hyy[p] * dy)
        local_Hxx[node] += local_Hxx[p]
        local_Hxy[node] += local_Hxy[p]
        local_Hyy[node] += local_Hyy[p]


def adaptive_fmm_accel(pos, mass, G=1.0, softening=1e-3, theta=0.5, max_near_pairs_per_particle=200):
    """FMM on an adaptive (Barnes-Hut-style) tree instead of `fmm_accel`'s
    uniform grid: a dual-tree M2L traversal (`_dual_tree_m2l`) plus the
    same L2L/L2P pattern, monopole+quadrupole accuracy throughout. Built
    to handle clustered mass distributions -- like a gravitational
    collapse -- that leave a uniform grid either too coarse (few particles
    per leaf cell, most of the M2L budget spent on near-empty cells) or
    forced to an impractically deep fixed level count everywhere just to
    resolve the densest region. Tree construction is the numba-jitted
    `_build_tree_numba` (shared with Barnes-Hut) plus its quadrupole
    moments; the traversal is `_dual_tree_m2l` above.

    NOTE: raising min_half to the softening scale (tried first) was the
    wrong lever -- it forced entire dense *resolvable* regions into a
    single degenerate leaf (up to 288 particles sharing one approximate
    far-field value in testing), which is a much worse approximation than
    a numerically fragile but individually-resolved tree. Keeping min_half
    at its tiny default and relying on min_sep2 (below) to guard the M2L
    step specifically is the right fix -- see `_dual_tree_m2l`.
    """
    N = len(mass)
    (node_mass, node_comx, node_comy, node_half, node_is_leaf, node_particle, node_children,
     node_parent, node_start, node_count, order, node_Qxx, node_Qxy, node_Qyy) = _build_tree_numba(
        pos, mass, max_nodes=8 * N + 4 * 50)

    local_a0x, local_a0y, local_Hxx, local_Hxy, local_Hyy, near_i, near_j = _dual_tree_m2l(
        node_mass, node_comx, node_comy, node_half, node_is_leaf, node_particle, node_children,
        node_Qxx, node_Qxy, node_Qyy, theta, G,
        max_near_pairs=max_near_pairs_per_particle * N, min_sep2=(10 * softening)**2)
    _l2l(node_parent, node_comx, node_comy, local_a0x, local_a0y, local_Hxx, local_Hxy, local_Hyy)

    accel = np.zeros((N, 2))
    n_nodes = len(node_mass)
    for node in range(n_nodes):
        if node_is_leaf[node]:
            p = node_particle[node]
            if p != -1:
                accel[p, 0] += local_a0x[node]
                accel[p, 1] += local_a0y[node]
            elif node_count[node] > 1:
                # degenerate multi-particle leaf: every member gets the
                # leaf's shared far-field local expansion (a fine
                # approximation -- the leaf is smaller than the tree's
                # minimum resolvable size), plus a small brute-force sum
                # for their mutual interactions (the group is tiny).
                idxs = order[node_start[node]:node_start[node] + node_count[node]]
                accel[idxs, 0] += local_a0x[node]
                accel[idxs, 1] += local_a0y[node]
                accel[idxs] += pairwise_accel(pos[idxs], mass[idxs], G=G, softening=softening)

    if len(near_i) > 0:
        dx = pos[near_j, 0] - pos[near_i, 0]
        dy = pos[near_j, 1] - pos[near_i, 1]
        inv_r3 = (dx * dx + dy * dy + softening**2) ** -1.5
        fx = G * mass[near_j] * dx * inv_r3
        fy = G * mass[near_j] * dy * inv_r3
        np.add.at(accel, near_i, np.stack([fx, fy], axis=1))
        fx2 = G * mass[near_i] * (-dx) * inv_r3
        fy2 = G * mass[near_i] * (-dy) * inv_r3
        np.add.at(accel, near_j, np.stack([fx2, fy2], axis=1))
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


@njit(cache=True, parallel=True)
def _potential_energy_numba(pos, mass, G, softening):
    """Direct O(N^2) pairwise sum, but looped rather than materializing
    the full (N,N) diff/distance/mass-outer-product arrays numpy
    broadcasting would build -- at the particle counts the collapse
    notebooks now run (N=15000+), those transient arrays are multiple
    GB each and dominate wall-clock time (originally ~7s/call at
    N=16000, called 150-300x in a conservation-check loop -- the actual
    cause of a real KeyboardInterrupt/near-hang found while committing
    Galaxy_Collision.ipynb). This is per-particle work under `prange`,
    same pattern as `_bh_walk`."""
    N = pos.shape[0]
    soft2 = softening**2
    PE = 0.0
    for i in prange(N):
        pe_i = 0.0
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy + soft2)
            pe_i -= mass[i] * mass[j] / dist
        PE += pe_i
    return G * PE


def energy(pos, vel, mass, G=1.0, softening=1e-3):
    """Total kinetic + softened potential energy (a single snapshot)."""
    KE = 0.5 * np.sum(mass * np.sum(vel**2, axis=-1))
    PE = _potential_energy_numba(pos, mass, G, softening)
    return KE + PE


def angular_momentum(pos, vel, mass):
    """Total z-component of angular momentum (a single snapshot)."""
    return np.sum(mass * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
