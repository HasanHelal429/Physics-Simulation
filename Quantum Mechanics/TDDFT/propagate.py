"""Real-time TDDFT propagator (Stage 1, Python).

Propagates a set of Kohn-Sham orbitals phi_j(r, t) on the periodic 3D grid
from Molecular_DFT/ under the time-dependent KS potential

    i d phi_j/dt = [ -1/2 nabla^2 + v_ext(r,t) + v_KS[rho(r,t)] ] phi_j ,
    rho(r,t) = sum_j f_j |phi_j(r,t)|^2

with adiabatic LDA (v_KS = the same static Slater+PZ81 functional as the
ground-state code, evaluated at the instantaneous density -- see
Molecular_DFT/potentials3d.ks_potential).

Phase 1 uses `strang_step` with a FIXED external potential (no density
feedback) to validate the bare propagator against TDSE_Solver's closed-form
free-particle / harmonic-oscillator results. Phase 2+ uses `etrs_step`,
which adds the enforced-time-reversal-symmetry (ETRS) predictor/corrector
needed once v_KS depends on the evolving density.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Molecular_DFT"))
import potentials3d as pot  # noqa: E402

_SPATIAL = (-3, -2, -1)


# --------------------------------------------------------------------------
# Low-level split-operator pieces. `psi` is complex, shape (..., Nx, Ny, Nz);
# a leading orbital axis is optional and broadcast over. `V` is a real 3D
# array (or scalar) broadcasting against the spatial axes.
# --------------------------------------------------------------------------

def kinetic_step(psi, G2, dt):
    """exp(-i * (G^2/2) * dt) applied in Fourier space (exact kinetic phase)."""
    psi_hat = np.fft.fftn(psi, axes=_SPATIAL)
    psi_hat *= np.exp(-0.5j * G2 * dt)
    return np.fft.ifftn(psi_hat, axes=_SPATIAL)


def potential_step(psi, V, dt):
    """exp(-i V dt) pointwise in real space. Pass dt/2 for a half kick."""
    return np.exp(-1j * V * dt) * psi


def strang_step(psi, V, G2, dt):
    """One symmetric Strang split-step for a FIXED potential V:
    exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2). 2nd-order in dt, exactly unitary.
    """
    psi = potential_step(psi, V, 0.5 * dt)
    psi = kinetic_step(psi, G2, dt)
    psi = potential_step(psi, V, 0.5 * dt)
    return psi


# --------------------------------------------------------------------------
# Density and the self-consistent (ETRS) step
# --------------------------------------------------------------------------

def density(psi, occ):
    """rho(r) = sum_j occ_j |phi_j(r)|^2. psi shape (n_orb, Nx, Ny, Nz)."""
    occ = np.asarray(occ)
    return np.einsum("j,j...->...", occ, np.abs(psi) ** 2)


def _orthonormalize(psi, dx):
    """Modified Gram-Schmidt on an orbital stack (n_orb, Nx, Ny, Nz) w.r.t. the
    dx^3 real-space inner product."""
    out = np.array(psi, dtype=complex)
    for i in range(len(out)):
        for j in range(i):
            ov = np.sum(np.conj(out[j]) * out[i]) * dx ** 3
            out[i] -= ov * out[j]
        out[i] /= np.sqrt(np.real(np.sum(np.abs(out[i]) ** 2) * dx ** 3))
    return out


def imaginary_time_ground_state(grid, V_nuc, N_electrons, method="lda",
                                alpha=pot.ALPHA_SCHWARZ, dtau=None, max_iter=600,
                                inner=4, tol=1e-8, seed=0, verbose=False):
    """Kohn-Sham ground state by imaginary-time propagation -- the FFT-only
    alternative to scf3d's eigsh (which becomes the bottleneck on the larger
    boxes rt-TDDFT strong-field runs need; also what the planned C++ Stage 2
    uses via 05_tdse_gpu's --relax).

    Each outer step: build V_KS[rho], apply a few symmetric imaginary-time
    split-steps exp(-V dtau/2) exp(-T dtau) exp(-V dtau/2) to every orbital
    (damps out the high-energy components), Gram-Schmidt orthonormalize,
    rebuild rho. Converges when the KS eigenvalue sum and the density stop
    moving.

    Returns (psi, occ, eps) -- normalized complex orbitals (n_orb, *grid),
    occupations (2 per orbital, last one 1 if N_electrons is odd), and the
    per-orbital energies from the converged V_KS.
    """
    x, X, Y, Z, dx, G2 = grid
    shape = X.shape
    n_orb = (N_electrons + 1) // 2
    occ = np.full(n_orb, 2.0)
    if N_electrons % 2:
        occ[-1] = 1.0
    if dtau is None:
        dtau = 0.2 * dx ** 2

    rng = np.random.default_rng(seed)
    r2 = X ** 2 + Y ** 2 + Z ** 2
    psi = np.array([np.exp(-r2 / (2.0 + 3.0 * k)) * (1.0 + 0.05 * rng.standard_normal(shape))
                    for k in range(n_orb)], dtype=complex)
    psi = _orthonormalize(psi, dx)

    kin = np.exp(-0.5 * G2 * dtau)
    E_prev = None
    for it in range(1, max_iter + 1):
        rho = density(psi, occ)
        V_eff, _ = pot.ks_potential(rho, V_nuc, G2, method=method, alpha=alpha)
        half = np.exp(-0.5 * V_eff * dtau)
        for _ in range(inner):
            psi = half * psi
            psi = np.fft.ifftn(kin * np.fft.fftn(psi, axes=_SPATIAL), axes=_SPATIAL)
            psi = half * psi
            psi = _orthonormalize(psi, dx)
        eps = np.array([orbital_energy(psi[j], V_eff, G2, dx) for j in range(n_orb)])
        E = float(np.sum(occ * eps))
        dn = np.sum(np.abs(density(psi, occ) - rho)) * dx ** 3
        if verbose and it % 20 == 0:
            print(f"      imag-time it {it:4d}: sum eps = {E:.6f}  dn = {dn:.2e}")
        if E_prev is not None and abs(E - E_prev) < tol and dn < 1e-6:
            break
        E_prev = E
    return psi, occ, eps


def relax_to_self_consistency(psi, occ, V_nuc, grid, method="lda",
                              alpha=pot.ALPHA_SCHWARZ, n_iter=40, tol=1e-9):
    """Pure fixed-point refinement (no density mixing) of a near-converged KS
    state from Molecular_DFT/scf3d: repeatedly diagonalize ks_potential(rho)
    and rebuild rho until int|delta rho| d^3r < tol.

    scf3d.run_scf stops when its *mixed* density change is below tol_n; the
    returned orbitals then diagonalize a V_eff built from the previous
    iteration's density, so they are not quite eigenstates of
    ks_potential(rho_returned). rt-TDDFT's Phase-2 fixed-point test needs
    that identity to hold tightly (otherwise the density breathes at the
    SCF-residual amplitude), so this drives it home -- near the solution the
    unmixed iteration converges fast.
    """
    import scf3d

    x, X, Y, Z, dx, G2 = grid
    shape = X.shape
    occ = np.asarray(occ)
    energies = None
    for _ in range(n_iter):
        rho = density(psi, occ)
        V_eff, _ = pot.ks_potential(rho, V_nuc, G2, method=method, alpha=alpha)
        energies, vecs = scf3d.solve_lowest_states(V_eff, G2, shape, len(occ))
        new = np.array([vecs[:, j].reshape(shape) for j in range(len(occ))], dtype=complex)
        for j in range(len(occ)):
            new[j] /= np.sqrt(np.sum(np.abs(new[j]) ** 2) * dx ** 3)
        drho = np.sum(np.abs(density(new, occ) - rho)) * dx ** 3
        psi = new
        if drho < tol:
            break
    return psi, energies[:len(occ)]


def _v_ks(rho, V_nuc, G2, method, alpha, v_ext):
    if method is None:
        V_eff = V_nuc
    else:
        V_eff, _ = pot.ks_potential(rho, V_nuc, G2, method=method, alpha=alpha)
    return V_eff if v_ext is None else V_eff + v_ext


def etrs_step(psi, occ, V_nuc, G2, dt, method="lda", alpha=pot.ALPHA_SCHWARZ,
              v_ext_now=None, v_ext_next=None, mask=None):
    """One ETRS step for the density-dependent KS Hamiltonian:

        V0        = v_KS[rho(t)]      (+ v_ext_now)
        psi_pred  = Strang(psi, V0, dt)                 # predictor
        V1        = v_KS[rho(psi_pred)] (+ v_ext_next)
        psi(t+dt) = exp(-iV1 dt/2) exp(-iT dt) exp(-iV0 dt/2) psi(t)

    then multiply by `mask` (absorbing boundary) if given. `v_ext_next`
    defaults to `v_ext_now` (static external potential).

    method=None skips the density feedback entirely (V is just V_nuc + the
    time-dependent field), so the predictor is unnecessary -- this is the
    fast single-particle path for the hydrogen HHG run.
    """
    if v_ext_next is None:
        v_ext_next = v_ext_now

    if method is None:
        V0 = V_nuc if v_ext_now is None else V_nuc + v_ext_now
        V1 = V_nuc if v_ext_next is None else V_nuc + v_ext_next
    else:
        V0 = _v_ks(density(psi, occ), V_nuc, G2, method, alpha, v_ext_now)
        psi_pred = strang_step(psi, V0, G2, dt)
        V1 = _v_ks(density(psi_pred, occ), V_nuc, G2, method, alpha, v_ext_next)

    psi = potential_step(psi, V0, 0.5 * dt)
    psi = kinetic_step(psi, G2, dt)
    psi = potential_step(psi, V1, 0.5 * dt)
    if mask is not None:
        psi = psi * mask
    return psi


# --------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------

def norm(psi, dx):
    """Per-orbital norm integral |phi_j|^2 d^3r (array, one value per orbital)."""
    return np.sum(np.abs(psi) ** 2, axis=_SPATIAL) * dx ** 3


def expectation_r(psi, occ, coords, dx):
    """Electronic dipole-ish first moment <r> = integral r rho(r) d^3r / N_e,
    returned as a length-3 vector (x, y, z). coords = (X, Y, Z)."""
    rho = density(psi, occ)
    Ne = np.sum(rho) * dx ** 3
    return np.array([np.sum(c * rho) * dx ** 3 / Ne for c in coords])


def dipole(psi, occ, coords, dx):
    """Electronic dipole moment d = - integral r rho(r) d^3r (length-3 vector)."""
    rho = density(psi, occ)
    return np.array([-np.sum(c * rho) * dx ** 3 for c in coords])


def variance_r(psi, occ, coords, dx):
    """Per-axis spatial variance of the density (length-3 vector)."""
    rho = density(psi, occ)
    Ne = np.sum(rho) * dx ** 3
    out = []
    for c in coords:
        mean = np.sum(c * rho) * dx ** 3 / Ne
        out.append(np.sum((c - mean) ** 2 * rho) * dx ** 3 / Ne)
    return np.array(out)


def _kinetic_expectation(psi_j, G2, dx):
    """<phi_j| -1/2 nabla^2 |phi_j> for one orbital (Parseval on the FFT)."""
    ph = np.fft.fftn(psi_j)
    return np.sum(0.5 * G2 * np.abs(ph) ** 2) * dx ** 3 / ph.size


def orbital_energy(psi_j, V, G2, dx):
    """<phi|H|phi>/<phi|phi> for one orbital and a fixed real potential V."""
    T = _kinetic_expectation(psi_j, G2, dx)
    Vexp = np.sum(V * np.abs(psi_j) ** 2) * dx ** 3
    n = np.sum(np.abs(psi_j) ** 2) * dx ** 3
    return (T + Vexp) / n


def _e_xc(rho, parts, dx):
    """Exchange-correlation energy, same convention as scf3d.run_scf:
    E_x = integral(e_x rho) = (3/4) integral(V_x rho)  (Euler's theorem on the
    rho^{4/3} Slater functional), plus PZ81 E_c = integral(eps_c rho) if present.
    """
    E = 0.75 * np.sum(parts["V_x"] * rho) * dx ** 3
    if parts["eps_c"] is not None:
        E += np.sum(parts["eps_c"] * rho) * dx ** 3
    return E


def ks_total_energy(psi, occ, V_nuc, G2, dx, method="lda", alpha=pot.ALPHA_SCHWARZ):
    """Kohn-Sham total energy E = T_s + E_ext + E_H + E_xc for the current
    orbitals -- the same decomposition scf3d.run_scf's `sum_eps - E_H - E_x/3
    [+ E_c doublecount]` reduces to. Excludes nuclear-nuclear repulsion (add
    it separately; it is constant during a fixed-geometry propagation)."""
    occ = np.asarray(occ)
    rho = density(psi, occ)
    _, parts = pot.ks_potential(rho, V_nuc, G2, method=method, alpha=alpha)
    T_s = sum(occ[j] * _kinetic_expectation(psi[j], G2, dx) for j in range(len(occ)))
    E_ext = np.sum(V_nuc * rho) * dx ** 3
    E_H = 0.5 * np.sum(parts["V_H"] * rho) * dx ** 3
    return T_s + E_ext + E_H + _e_xc(rho, parts, dx)


def propagate(psi0, grid, n_steps, dt, *, fixed_V=None, occ=None, V_nuc=None,
              method="lda", alpha=pot.ALPHA_SCHWARZ, v_ext_fn=None, mask=None,
              record_every=1, observers=None):
    """Drive the propagator for `n_steps` of `dt`.

    fixed_V given -> Phase-1 mode: Strang steps with that fixed potential (psi0
      may be a single orbital or an orbital stack).
    fixed_V None  -> ETRS mode: needs `occ` and `V_nuc`; v_ext_fn(t) -> 3D array
      or None supplies a time-dependent drive.

    `observers` is a dict {name: fn(psi, t) -> scalar/array}; their values are
    collected every `record_every` steps and returned as arrays under
    result["obs"][name], with result["t"] the matching times.
    """
    x, X, Y, Z, dx = grid[:5]
    G2 = grid[5]
    psi = np.asarray(psi0, dtype=complex).copy()
    observers = observers or {}
    rec = {name: [] for name in observers}
    ts = []

    def snap(t):
        ts.append(t)
        for name, fn in observers.items():
            rec[name].append(fn(psi, t))

    snap(0.0)
    for step in range(1, n_steps + 1):
        t = (step - 1) * dt
        if fixed_V is not None:
            psi = strang_step(psi, fixed_V, G2, dt)
        else:
            vn = v_ext_fn(t) if v_ext_fn else None
            vnn = v_ext_fn(t + dt) if v_ext_fn else None
            psi = etrs_step(psi, occ, V_nuc, G2, dt, method=method, alpha=alpha,
                            v_ext_now=vn, v_ext_next=vnn, mask=mask)
        if step % record_every == 0:
            snap(step * dt)

    return {"psi": psi, "t": np.array(ts),
            "obs": {name: np.array(v) for name, v in rec.items()}}
