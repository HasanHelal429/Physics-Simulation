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


def lj_force(r, ids_pairs, epsilon, sigma):
    """Total Lennard-Jones force on every particle from all pairs in
    ids_pairs (shape (M,2)): vectorized over every pair at once (no
    per-pair Python loop -- the legacy prototype's `for idx in
    range(close_pairs.shape[0])` loop defeated the point of using GPU
    tensors), scattered to each particle with index_add_.

    r: (2, N) torch tensor of positions. Returns (fx, fy): (N,) tensors.
    """
    i, j = ids_pairs[:, 0], ids_pairs[:, 1]
    rij = r[:, i] - r[:, j]
    dist = torch.sqrt(rij[0]**2 + rij[1]**2)
    Fmag = lj_force_magnitude(dist, epsilon, sigma)
    Fvec = Fmag / dist * rij

    fx = torch.zeros(r.shape[1], device=r.device, dtype=r.dtype)
    fy = torch.zeros(r.shape[1], device=r.device, dtype=r.dtype)
    fx.index_add_(0, i, Fvec[0])
    fy.index_add_(0, i, Fvec[1])
    fx.index_add_(0, j, -Fvec[0])
    fy.index_add_(0, j, -Fvec[1])
    return fx, fy


def lj_potential_energy(r, ids_pairs, epsilon, sigma):
    """Total Lennard-Jones potential energy, summed over all pairs."""
    i, j = ids_pairs[:, 0], ids_pairs[:, 1]
    rij = r[:, i] - r[:, j]
    dist = torch.sqrt(rij[0]**2 + rij[1]**2)
    return torch.sum(lj_potential(dist, epsilon, sigma))


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
           track_pressure=True, thermostat_T=None, thermostat_every=None):
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

    Returns (rs, vs, ps): trajectories of shape (ts, 2, N), and either a
    (ts,) pressure-proxy tensor or None if track_pressure is False.
    """
    device = r.device
    N = r.shape[1]
    ids_pairs = torch.combinations(torch.arange(N, device=device), 2)

    rs = torch.zeros((ts, *r.shape), device=device)
    vs = torch.zeros((ts, *v.shape), device=device)
    ps = torch.zeros(ts, device=device) if track_pressure else None
    rs[0] = r
    vs[0] = v

    ax, ay = lj_force(r, ids_pairs, epsilon, sigma)

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

        fx_new, fy_new = lj_force(r, ids_pairs, epsilon, sigma)
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
