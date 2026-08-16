"""2D Lennard-Jones molecular dynamics: the microscopic pairwise force
that produces van der Waals-type non-ideal gas behavior, replacing
Ideal_Gas's hard-sphere elastic collisions. Fully vectorized force
calculation (no per-pair Python loop, unlike the legacy prototype this
replaces), a proper velocity-Verlet integrator, and the same wall
-momentum pressure-tracking convention as ideal_gas.py, so the two
gases are directly comparable at the same N, box size, and temperature.
"""
import numpy as np
import torch
from scipy.integrate import quad


def lj_potential(r, epsilon, sigma):
    """Lennard-Jones pair potential U(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)."""
    sr6 = (sigma / r)**6
    return 4 * epsilon * (sr6**2 - sr6)


def lj_force_magnitude(r, epsilon, sigma):
    """Radial force F(r) = -dU/dr = 24*epsilon*(2*(sigma/r)^12 - (sigma/r)^6)/r.
    Positive (repulsive, r < 2^(1/6)*sigma) or negative (attractive,
    beyond that) by the usual convention that the force on particle i is
    F(r)*r_hat, r_hat pointing away from particle j.
    """
    sr6 = (sigma / r)**6
    return 24 * epsilon * (2 * sr6**2 - sr6) / r


def lj_force(r, ids_pairs, epsilon, sigma, cutoff=None):
    """Total Lennard-Jones force on every particle from all pairs in
    ids_pairs (shape (M,2)): vectorized over every pair at once (no
    per-pair Python loop -- the legacy prototype's `for idx in
    range(close_pairs.shape[0])` loop defeated the point of using GPU
    tensors), scattered to each particle with index_add_.

    r: (2, N) torch tensor of positions. Returns (fx, fy): (N,) tensors.

    `cutoff`, if given, uses the *shifted-force* potential: subtracts
    F(cutoff) from the force for every pair within cutoff (dropping pairs
    beyond it -- needed when `ids_pairs` comes from `build_neighbor_pairs`,
    whose cell-list candidates can include a few pairs just past the true
    cutoff). A hard, unshifted cutoff makes the force jump discontinuously
    as a pair crosses the cutoff distance -- energy isn't conserved
    through a discontinuity, and this showed up as ~20% energy drift over
    10,000 steps in testing (vs. 0.65% without any cutoff). Shifting so
    the force goes smoothly to zero exactly at the cutoff (matching
    lj_potential_energy's shift, so the two stay a consistent
    force/potential pair) fixes that -- see Validation.ipynb.
    """
    i, j = ids_pairs[:, 0], ids_pairs[:, 1]
    rij = r[:, i] - r[:, j]
    dist = torch.sqrt(rij[0]**2 + rij[1]**2)
    if cutoff is not None:
        within = dist <= cutoff
        i, j, dist, rij = i[within], j[within], dist[within], rij[:, within]
        Fmag = lj_force_magnitude(dist, epsilon, sigma) - lj_force_magnitude(cutoff, epsilon, sigma)
    else:
        Fmag = lj_force_magnitude(dist, epsilon, sigma)
    Fvec = Fmag / dist * rij

    fx = torch.zeros(r.shape[1], device=r.device, dtype=r.dtype)
    fy = torch.zeros(r.shape[1], device=r.device, dtype=r.dtype)
    fx.index_add_(0, i, Fvec[0])
    fy.index_add_(0, i, Fvec[1])
    fx.index_add_(0, j, -Fvec[0])
    fy.index_add_(0, j, -Fvec[1])
    return fx, fy


def lj_potential_energy(r, ids_pairs, epsilon, sigma, cutoff=None):
    """Total Lennard-Jones potential energy, summed over all pairs (or
    all within `cutoff`, if given). With `cutoff` set, uses the shifted
    potential U(r) - U(rc) + (r-rc)*F(rc) that's the energy consistent
    with `lj_force`'s shifted-force cutoff (both vanish continuously at
    r=cutoff, rather than U alone being shifted -- shifting only U still
    leaves the force discontinuous and doesn't fix energy conservation).
    """
    i, j = ids_pairs[:, 0], ids_pairs[:, 1]
    rij = r[:, i] - r[:, j]
    dist = torch.sqrt(rij[0]**2 + rij[1]**2)
    if cutoff is not None:
        dist = dist[dist <= cutoff]
        U_c = lj_potential(cutoff, epsilon, sigma)
        F_c = lj_force_magnitude(cutoff, epsilon, sigma)
        U = lj_potential(dist, epsilon, sigma) - U_c + (dist - cutoff) * F_c
    else:
        U = lj_potential(dist, epsilon, sigma)
    return torch.sum(U)


def build_neighbor_pairs(r, L, cutoff, max_per_cell=None):
    """Candidate (i,j) pairs within `cutoff` of each other, found via a
    uniform cell list instead of checking all N*(N-1)/2 pairs: cell size
    = cutoff, so any two particles closer than cutoff must share a cell
    or be in one of its 8 neighbors. Turns lj_force's cost from O(N^2)
    to roughly O(N) at fixed density -- verified by direct comparison
    against brute-force all-pairs (exact match across 20 randomized
    trials) and by scaling measurement (pairs-per-particle stayed ~
    constant, ~5.0, from N=200 to N=51,200 -- a 256x range).

    Fully vectorized (no per-particle or per-cell Python loop): particles
    are bucketed into cells via an argsort + cumulative-count "rank
    within group" trick, then every particle's 3x3 neighborhood is
    gathered by fixed-shape tensor indexing. Candidates can include a
    few pairs just past the true cutoff (adjacent-cell corners); callers
    (`lj_force`, `lj_potential_energy` with `cutoff` set) filter by exact
    distance.

    `max_per_cell` bounds each cell's particle count for the fixed-shape
    gather; a cell that overflows it silently drops the excess particles
    from that cell's outgoing pairs (undercounting some interactions) --
    the default is generous (4x the uniform-density expectation) but
    isn't a hard guarantee for a strongly clustered/non-uniform gas.
    """
    device = r.device
    N = r.shape[1]
    n_cells = max(1, int(L / cutoff))
    cell_size = L / n_cells

    cx = torch.clamp((r[0] / cell_size).long(), 0, n_cells - 1)
    cy = torch.clamp((r[1] / cell_size).long(), 0, n_cells - 1)
    cell_id = cx * n_cells + cy

    if max_per_cell is None:
        max_per_cell = max(8, int(4 * N / n_cells**2) + 8)

    order = torch.argsort(cell_id)
    sorted_cell_id = cell_id[order]
    changed = torch.ones_like(sorted_cell_id, dtype=torch.bool)
    changed[1:] = sorted_cell_id[1:] != sorted_cell_id[:-1]
    idx = torch.arange(N, device=device)
    group_start = torch.where(changed, idx, torch.zeros(N, dtype=torch.long, device=device))
    group_start = torch.cummax(group_start, dim=0).values
    slot = idx - group_start

    valid = slot < max_per_cell
    bucket = torch.full((n_cells * n_cells, max_per_cell), -1, dtype=torch.long, device=device)
    bucket[sorted_cell_id[valid], slot[valid]] = order[valid]

    offsets = torch.tensor([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], device=device)
    raw_ncx = cx.unsqueeze(1) + offsets[:, 0]
    raw_ncy = cy.unsqueeze(1) + offsets[:, 1]
    offset_valid = (raw_ncx >= 0) & (raw_ncx < n_cells) & (raw_ncy >= 0) & (raw_ncy < n_cells)
    ncx = raw_ncx.clamp(0, n_cells - 1)
    ncy = raw_ncy.clamp(0, n_cells - 1)
    neighbor_cell_id = ncx * n_cells + ncy  # (N, 9)

    candidates = bucket[neighbor_cell_id]  # (N, 9, max_per_cell)
    candidates = candidates.masked_fill(~offset_valid.unsqueeze(-1), -1)
    candidates = candidates.reshape(N, -1)

    i_idx = idx.unsqueeze(1).expand_as(candidates)
    valid_cand = candidates >= 0
    i_flat = i_idx[valid_cand]
    j_flat = candidates[valid_cand]

    keep = i_flat < j_flat  # dedupe: each unordered pair appears from both i's and j's neighborhoods
    return torch.stack([i_flat[keep], j_flat[keep]], dim=1)


def _reflect_wall(r_axis, v_axis, L, hi, lo, p, track_pressure):
    """Elastic reflection off whichever of the two walls on this axis
    are active. Unlike ideal_gas.py's version of this helper, this one
    also mirrors the position back inside the box (`r = 2*L - r` / `r =
    -r`), not just flipping velocity -- Lennard-Jones forces need a
    physically valid (in-box) position for the next force evaluation,
    whereas Ideal_Gas's collision-only physics never depended on that.
    """
    if hi:
        hit = r_axis > L
        r_axis[hit] = 2 * L - r_axis[hit]
        v_axis[hit] = -torch.abs(v_axis[hit])
        if track_pressure:
            p[hit] += 2 * torch.abs(v_axis[hit])
    if lo:
        hit = r_axis < 0
        r_axis[hit] = -r_axis[hit]
        v_axis[hit] = torch.abs(v_axis[hit])
        if track_pressure:
            p[hit] += 2 * torch.abs(v_axis[hit])


def motion(r, v, ts, dt, epsilon, sigma, L=1.0,
           reflect_x_lo=True, reflect_x_hi=True, reflect_y_lo=True, reflect_y_hi=True,
           track_pressure=True, thermostat_T=None, thermostat_every=None,
           cutoff=None, neighbor_rebuild_every=1):
    """Advance an N-particle 2D Lennard-Jones gas for `ts` steps of size
    `dt` using velocity-Verlet (energy-conserving to a bounded,
    non-drifting error -- unlike the legacy prototype's ad hoc
    half-explicit-Euler scheme).

    r, v: (2, N) torch tensors (position, velocity; mass = 1 throughout).
    epsilon, sigma: Lennard-Jones parameters.
    L: box size; walls (where reflecting) sit at coordinate 0 and L.
    reflect_*: which of the four box walls elastically reflect particles
        (a wall set to False lets particles pass straight through).
    track_pressure: accumulate wall momentum-transfer per step (same
        convention as ideal_gas.py's `motion`, for direct comparison).
    thermostat_T, thermostat_every: if both given, rescale velocities
        toward mean-KE-per-particle = thermostat_T (mass=1, k=1 units,
        so 2D equipartition gives <0.5*v^2> = thermostat_T) every
        `thermostat_every` steps -- a simple velocity-rescaling
        thermostat for holding T fixed during an equation-of-state sweep
        (Phase 3). Leave both None for a true microcanonical run (Phase
        1's energy-conservation check needs this).
    cutoff: if given, forces beyond this pair distance are dropped (LJ
        is already negligible past a few sigma -- 2.5*sigma is the usual
        MD convention) and pairs are found via build_neighbor_pairs's
        cell list instead of all-pairs combinations, turning the O(N^2)
        force cost into roughly O(N) at fixed density. None (default)
        keeps the original exact all-pairs behavior, unchanged.
    neighbor_rebuild_every: with `cutoff` set, how often (in steps) to
        rebuild the neighbor list -- rebuilding every step (the default)
        is always correct since particles move very little per step at
        the dt this module needs anyway (see Validation.ipynb); a larger
        value trades a small risk of missing a newly-close pair between
        rebuilds for less list-rebuild overhead.

    Returns (rs, vs, ps): trajectories of shape (ts, 2, N), and either a
    (ts,) pressure-proxy tensor or None if track_pressure is False.
    """
    device = r.device
    N = r.shape[1]
    if cutoff is None:
        ids_pairs = torch.combinations(torch.arange(N, device=device), 2)
    else:
        ids_pairs = build_neighbor_pairs(r, L, cutoff)

    rs = torch.zeros((ts, *r.shape), device=device)
    vs = torch.zeros((ts, *v.shape), device=device)
    ps = torch.zeros(ts, device=device) if track_pressure else None
    rs[0] = r
    vs[0] = v

    ax, ay = lj_force(r, ids_pairs, epsilon, sigma, cutoff=cutoff)

    for i in range(1, ts):
        r = r.clone()
        r[0] = r[0] + v[0] * dt + 0.5 * ax * dt**2
        r[1] = r[1] + v[1] * dt + 0.5 * ay * dt**2

        v = v.clone()
        p = torch.zeros(N, device=device) if track_pressure else None
        _reflect_wall(r[0], v[0], L, reflect_x_hi, reflect_x_lo, p, track_pressure)
        _reflect_wall(r[1], v[1], L, reflect_y_hi, reflect_y_lo, p, track_pressure)
        if track_pressure:
            ps[i] = torch.sum(p)

        if cutoff is not None and i % neighbor_rebuild_every == 0:
            ids_pairs = build_neighbor_pairs(r, L, cutoff)
        fx_new, fy_new = lj_force(r, ids_pairs, epsilon, sigma, cutoff=cutoff)
        v[0] = v[0] + 0.5 * (ax + fx_new) * dt
        v[1] = v[1] + 0.5 * (ay + fy_new) * dt
        ax, ay = fx_new, fy_new

        if thermostat_T is not None and thermostat_every is not None and i % thermostat_every == 0:
            mean_ke = 0.5 * torch.sum(v**2) / N
            v = v * torch.sqrt(thermostat_T / (mean_ke + 1e-30))

        rs[i] = r
        vs[i] = v

    return rs, vs, ps


def second_virial_coefficient(T, epsilon, sigma, k=1.0, r_max_factor=15.0):
    """The 2D Lennard-Jones second virial coefficient,
    B2(T) = -pi * integral_0^inf (exp(-U(r)/(k*T)) - 1) * r dr
    (the 2D analog of the standard 3D formula's 4*pi*r^2 weighting),
    computed numerically. This is the first-principles, low-density
    prediction for how far the equation of state P = n*k*T*(1 + B2(T)*n
    + ...) departs from ideal (B2=0) -- used to validate the simulated
    equation of state without an ad hoc Lennard-Jones-to-van-der-Waals
    -constants mapping (unlike the a,b constants, B2 is an exact
    low-density result for a given pair potential).
    """
    def integrand(r):
        U = lj_potential(r, epsilon, sigma)
        return (np.exp(-U / (k * T)) - 1) * r
    integral, _ = quad(integrand, 1e-3 * sigma, r_max_factor * sigma, limit=200)
    return -np.pi * integral
