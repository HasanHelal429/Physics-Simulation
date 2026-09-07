"""
Potential-energy landscapes V(r) for the split-operator propagator
(propagator.py), plus a couple of closed-form reference solutions used
only to validate the propagator against known physics, not part of the
propagation pipeline itself.
"""

import math

import numpy as np
from scipy import special


def harmonic_well(grid, omega, center=None):
    """Isotropic quantum harmonic oscillator, V(r) = 1/2 * omega^2 * |r-center|^2
    (atomic units). Its exact eigenstates (harmonic_eigenstate below) are
    what Validation.ipynb uses to check the propagator against a case
    where V and T=-1/2 laplacian genuinely don't commute -- unlike a free
    particle (V=0), so there's a real O(dt^2) splitting error to measure."""
    if center is None:
        center = np.zeros(grid.ndim)
    center = np.broadcast_to(center, (grid.ndim,))
    r2 = sum((x - c) ** 2 for x, c in zip(grid.coords, center))
    return 0.5 * omega ** 2 * r2


def harmonic_eigenstate(grid, omega, n=0, center=None):
    """Exact eigenstate of harmonic_well(grid, omega, center): mode `n`
    (Hermite polynomial H_n) along axis 0, ground state along every other
    axis. Returns (psi, energy) with energy = omega*(n+1/2) + omega/2*(ndim-1).

    This is a genuine closed-form solution of the full (kinetic +
    potential) time-independent Schrodinger equation, so propagating it
    and checking psi(t) == psi(0)*exp(-i*energy*t) is a strong,
    independent test of strang_step -- it doesn't rely on any numerical
    diagonalization routine (stationary_states.py, Phase 4) being correct
    first.
    """
    if center is None:
        center = np.zeros(grid.ndim)
    center = np.broadcast_to(center, (grid.ndim,))
    psi = np.ones(grid.shape, dtype=complex)
    energy = 0.0
    for axis, (x, c) in enumerate(zip(grid.coords, center)):
        this_n = n if axis == 0 else 0
        xi = np.sqrt(omega) * (x - c)
        herm = special.hermite(this_n, monic=False)
        norm = (omega / np.pi) ** 0.25 / np.sqrt(float(2 ** this_n * math.factorial(this_n)))
        psi = psi * norm * herm(xi) * np.exp(-xi ** 2 / 2)
        energy += omega * (this_n + 0.5)
    return psi, energy


def rectangular_barrier(grid, V0, x_min, x_max, axis=0):
    """V0 for x_min <= x_axis <= x_max along `axis`, else 0 -- a rectangular
    potential barrier, uniform across every other axis (so it's usable as
    a 1D barrier or as a wall segment in a higher-dimensional slit
    geometry). See rectangular_barrier_transmission for the exact 1D
    plane-wave transmission probability through this shape."""
    x = grid.axes[axis]
    barrier_1d = np.where((x >= x_min) & (x <= x_max), V0, 0.0)
    shape = [1] * grid.ndim
    shape[axis] = len(x)
    return np.broadcast_to(barrier_1d.reshape(shape), grid.shape).astype(float).copy()


def rectangular_barrier_transmission(E, V0, a):
    """Exact transmission probability of a monochromatic plane wave of
    energy E through a 1D rectangular barrier of height V0, width a
    (atomic units) -- the standard textbook result (e.g. Griffiths,
    Introduction to Quantum Mechanics), covering both the tunneling
    (E<V0, using sinh) and over-barrier (E>V0, using sin -- and showing
    resonant perfect transmission whenever sqrt(2(E-V0))*a is a multiple
    of pi) regimes, plus the E=V0 limit.

    A real (non-monochromatic) Gaussian wavepacket has a nonzero energy
    spread (~1/(4*sigma0) in this convention), so a wavepacket-integrated
    transmission probability will differ from this pointwise formula by
    an amount that grows wherever T(E) is highly curved -- most visibly in
    the deep-tunneling regime, where T is exponentially sensitive to E.
    That's an expected, physical systematic (see Validation.ipynb, Phase
    5), not a sign either number is wrong.
    """
    scalar_input = np.ndim(E) == 0
    E = np.atleast_1d(np.asarray(E, dtype=float))
    T = np.empty_like(E)

    tunneling = E < V0
    over = E > V0
    at_top = ~(tunneling | over)

    kappa = np.sqrt(2 * (V0 - E[tunneling]))
    T[tunneling] = 1.0 / (1.0 + (V0 ** 2 * np.sinh(kappa * a) ** 2) / (4 * E[tunneling] * (V0 - E[tunneling])))

    k2 = np.sqrt(2 * (E[over] - V0))
    T[over] = 1.0 / (1.0 + (V0 ** 2 * np.sin(k2 * a) ** 2) / (4 * E[over] * (E[over] - V0)))

    T[at_top] = 1.0 / (1.0 + V0 * a ** 2 / 2)

    return float(T[0]) if scalar_input else T


def soft_coulomb_well(grid, strength, softening, center=None):
    """V(r) = -strength / sqrt(r^2 + softening^2) -- an attractive
    Coulomb-like well, softened near r=0 so the singularity (which no
    finite grid could resolve anyway) doesn't appear; reduces to the
    ordinary Coulomb potential -strength/r for r >> softening."""
    if center is None:
        center = np.zeros(grid.ndim)
    center = np.broadcast_to(center, (grid.ndim,))
    r2 = sum((x - c) ** 2 for x, c in zip(grid.coords, center))
    return -strength / np.sqrt(r2 + softening ** 2)


def double_well(grid, barrier_height, half_separation, curvature=None, axis=0):
    """Symmetric quartic double well along `axis`:

        V(x) = barrier_height * ( (x/half_separation)^2 - 1 )^2

    minima at x = +-half_separation (V = 0), a barrier of height
    `barrier_height` at x = 0. Near a minimum the well is harmonic with
    omega = sqrt(8 * barrier_height / (mass * half_separation^2)).

    The lowest two eigenstates form a near-degenerate symmetric/
    antisymmetric pair split by the tunneling gap dE; a state localized in
    one well (their equal superposition) oscillates to the other well with
    period 2*pi/dE -- the toy model of the covalent bond and of ammonia
    inversion. `curvature`, if given, rescales so the small-oscillation
    omega equals `curvature` instead (overrides `barrier_height`'s role in
    setting the well stiffness, keeping the barrier position fixed)."""
    x = grid.coords[axis]
    a = half_separation
    if curvature is not None:
        barrier_height = curvature ** 2 * a ** 2 / 8.0
    return barrier_height * ((x / a) ** 2 - 1.0) ** 2


def gamow_barrier(grid, well_depth, well_radius, coulomb_strength, axis=0,
                  softening=0.4, edge=0.5):
    """Spherically-symmetric alpha-decay model potential along the radial
    coordinate `axis` (assumed >= 0): a nuclear well of depth `well_depth`
    for r < well_radius, smoothly joined (over a width `edge`) to a softened
    repulsive Coulomb tail  coulomb_strength / sqrt(r^2 + softening^2).

    The smooth join (rather than a hard step) keeps the eigenstates free of
    high-k kinks, so the split-operator propagator stays accurate at a
    reasonable time step. A state with 0 < E < the barrier maximum is
    quasi-bound: trapped inside but able to tunnel out with a finite width
    Gamma, lifetime tau = 1/Gamma ~ the WKB / Gamow rate. The lifetime is
    exponentially sensitive to the barrier -- the origin of the
    Geiger-Nuttall law."""
    r = grid.coords[axis]
    coulomb = coulomb_strength / np.sqrt(r ** 2 + softening ** 2)
    switch = 0.5 * (1.0 + np.tanh((r - well_radius) / edge))     # 0 inside -> 1 outside
    return (1.0 - switch) * (-well_depth) + switch * coulomb


def double_slit_barrier(grid, V0, x_min, x_max, slit_width, slit_separation, axis=0, slit_axis=1):
    """A wall of height V0 spanning x_min<=x_axis<=x_max along `axis`,
    pierced by two slits of width `slit_width` centered at
    +-slit_separation/2 along `slit_axis` -- 0 everywhere else (including
    outside the wall's thickness). Analytic reference: for an incident
    wave of de Broglie wavelength lambda=2*pi/k0, the interference fringe
    spacing on a screen a distance L past the wall is `lambda*L/
    slit_separation` in the small-angle (paraxial) limit -- valid near
    the pattern's center, where the fringe position is close to linear
    in the diffraction angle; it grows away from center as the exact
    path-difference condition (m*lambda = d*sin(theta)) departs from
    that linear approximation. See Validation.ipynb (Phase 6)."""
    x = grid.coords[axis]
    y = grid.coords[slit_axis]
    in_wall_x = (x >= x_min) & (x <= x_max)
    in_slit = (np.abs(y - slit_separation / 2) <= slit_width / 2) | (np.abs(y + slit_separation / 2) <= slit_width / 2)
    return np.where(in_wall_x & ~in_slit, V0, 0.0)


def morse_well(grid, D_e, a, R_e, E_min=0.0, axis=0):
    """Morse potential V(R) = E_min + D_e (1 - e^{-a(R-R_e)})^2 along `axis`
    (broadcast over any others). V(R_e) = E_min, V(inf) = E_min + D_e. Its
    vibrational spectrum has the closed form
    E_v = omega_e (v+1/2) - omega_e x_e (v+1/2)^2 with
    omega_e = a sqrt(2 D_e / mu), omega_e x_e = a^2 / (2 mu) -- used to
    validate the reduced-mass grid Hamiltonian in Nuclear_Dynamics/."""
    x = grid.axes[axis]
    v1d = E_min + D_e * (1.0 - np.exp(-a * (x - R_e))) ** 2
    shape = [1] * grid.ndim
    shape[axis] = len(x)
    return np.broadcast_to(v1d.reshape(shape), grid.shape).astype(float).copy()


def _fit_morse(R, E):
    """Least-squares Morse fit to tabulated (R, E); returns (D_e, a, R_e, E_min)."""
    from scipy.optimize import curve_fit

    R = np.asarray(R, float)
    E = np.asarray(E, float)
    i_min = int(np.argmin(E))
    E_min0 = E[i_min]
    R_e0 = R[i_min]
    D_e0 = max(E.max() - E_min0, 1e-3)

    def model(RR, D_e, a, R_e, E_min):
        return E_min + D_e * (1.0 - np.exp(-a * (RR - R_e))) ** 2

    try:
        popt, _ = curve_fit(model, R, E, p0=[D_e0, 1.0, R_e0, E_min0], maxfev=20000)
        D_e, a, R_e, E_min = popt
        if D_e < 0:
            D_e, a = abs(D_e), abs(a)
    except Exception:
        D_e, a, R_e, E_min = D_e0, 1.0, R_e0, E_min0
    return float(D_e), float(abs(a)), float(R_e), float(E_min)


def potential_from_samples(grid, R_samples, E_samples, fill="morse", axis=0,
                           return_fit=False):
    """Turn a tabulated Born-Oppenheimer curve E(R) (as produced by
    Diatomic_HF_solver / Molecular_DFT PES scans) into a potential array on
    `grid`.

    Interior of the sampled range: a natural cubic spline through the points.
    Outside it: `fill="morse"` continues the curve with a Morse fit to the
    samples (physically correct steep wall as R -> 0 and flat asymptote as
    R -> inf, and the fit doubles as the Phase-1 closed-form reference);
    `fill="constant"` clamps to the edge values; `fill="spline"` lets the
    cubic spline extrapolate (not recommended past a small margin).

    Returns V with shape grid.shape, or (V, morse_params_dict) if return_fit.
    """
    from scipy.interpolate import CubicSpline

    R_samples = np.asarray(R_samples, float)
    E_samples = np.asarray(E_samples, float)
    order = np.argsort(R_samples)
    R_samples, E_samples = R_samples[order], E_samples[order]
    R_lo, R_hi = R_samples[0], R_samples[-1]

    spline = CubicSpline(R_samples, E_samples, extrapolate=(fill == "spline"))
    D_e, a, R_e, E_min = _fit_morse(R_samples, E_samples)

    def morse(RR):
        return E_min + D_e * (1.0 - np.exp(-a * (RR - R_e))) ** 2

    x = grid.axes[axis]
    v = np.empty_like(x, dtype=float)
    inside = (x >= R_lo) & (x <= R_hi)
    v[inside] = spline(x[inside])
    if fill == "morse":
        v[~inside] = morse(x[~inside])
        # remove any small offset so the spline and Morse agree at the seams
        for edge, mask in ((R_lo, x < R_lo), (R_hi, x > R_hi)):
            if mask.any():
                v[mask] += spline(edge) - morse(edge)
    elif fill == "constant":
        v[x < R_lo] = E_samples[0]
        v[x > R_hi] = E_samples[-1]
    else:
        v[~inside] = spline(x[~inside])

    shape = [1] * grid.ndim
    shape[axis] = len(x)
    V = np.broadcast_to(v.reshape(shape), grid.shape).astype(float).copy()
    if return_fit:
        return V, {"D_e": D_e, "a": a, "R_e": R_e, "E_min": E_min, "E_inf": E_min + D_e}
    return V


def absorbing_boundary(grid, width, eta, order=3, axes=None):
    """Complex absorbing potential (CAP): a real, non-negative ramp W(r)
    that grows over a layer of thickness `width` at the domain edges,
    meant to be subtracted in as an imaginary term when building the
    effective potential passed to propagator.strang_step:

        V_eff = V_real - 1j * W

    propagator.potential_step's existing exp(-i*V_eff*dt) then factors as
    exp(-i*V_real*dt) * exp(-W*dt) -- an ordinary phase times a real decay
    factor -- with no changes needed to propagator.py itself, since it
    never assumed V was real. This is the standard companion to periodic
    (FFT) propagation for open/scattering problems: instead of outgoing
    probability density either reflecting off a hard wall or wrapping
    back around through the FFT's implicit periodicity, it's absorbed
    before it reaches the edge. That absorption necessarily costs norm
    (see observables.norm) -- the lost norm *is* the flux that left.

    `eta` sets the ramp's strength and `order` (2-4 are typical) its
    steepness; both interact with `width` and the wavepacket's momentum
    to determine how much of the incident wave reflects off the CAP
    itself rather than being absorbed -- tuned by direct simulation in
    Validation.ipynb (Phase 3) rather than derived analytically here.
    width=15, eta=15, order=3 (this function's defaults) were found there
    to give reflection/leakage on the order of 1e-9 for a representative
    wavepacket, and are a reasonable starting point elsewhere in this
    project.

    axes: which axes get a layer (default: all of them) -- e.g. leave a
    transverse axis alone in a 2D scattering setup where only the
    propagation axis should absorb.
    """
    if axes is None:
        axes = range(grid.ndim)
    W = np.zeros(grid.shape)
    for axis in axes:
        x = grid.axes[axis]
        L = grid.lengths[axis]
        if grid.boundary == 'periodic':
            edge_lo, edge_hi = -L / 2, L / 2
        else:
            edge_lo, edge_hi = 0.0, L
        dist_lo = np.clip((width - (x - edge_lo)) / width, 0.0, None)
        dist_hi = np.clip((width - (edge_hi - x)) / width, 0.0, None)
        ramp = eta * (dist_lo ** order + dist_hi ** order)
        shape = [1] * grid.ndim
        shape[axis] = len(x)
        W = W + ramp.reshape(shape)
    return W
