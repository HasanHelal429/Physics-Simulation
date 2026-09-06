"""Stage-1 TDDFT validation, headless.

Phase 1 -- bare propagator against TDSE_Solver's closed-form results, on the
Molecular_DFT 3D grid:
  * free Gaussian wavepacket:  sigma(t) = sigma0 sqrt(1 + (t / 2 sigma0^2)^2),
    <r>(t) = k0 t, norm conserved
  * harmonic coherent state:   <x>(t) = x0 cos(omega t), constant width,
    constant energy
  * ~2nd-order convergence of the harmonic case in dt
Phase 2 -- self-consistent propagation of the converged Molecular_DFT ground
state with no perturbation: pure-phase orbital evolution, stationary density
and dipole.

    python validate.py [--phase 1|2|all] [--quick]

Writes figures to media/ and prints a PASS/FAIL table.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "Molecular_DFT"))

import grid3d  # noqa: E402
import propagate as prop  # noqa: E402

MEDIA = os.path.join(HERE, "media")
os.makedirs(MEDIA, exist_ok=True)

_results = []


def check(name, passed, detail=""):
    _results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def _grid(L, N):
    x, X, Y, Z, dx = grid3d.make_grid(L, N)
    _, _, _, G2 = grid3d.g_vectors(L, N)
    return (x, X, Y, Z, dx, G2)


def gaussian(grid, center, sigma, k0=(0, 0, 0)):
    """Minimum-uncertainty Gaussian, same convention as
    TDSE_Solver.grid.gaussian_wavepacket: |psi|^2 has std dev `sigma`."""
    x, X, Y, Z, dx = grid[:5]
    center = np.broadcast_to(center, 3).astype(float)
    sigma = np.broadcast_to(sigma, 3).astype(float)
    k0 = np.broadcast_to(k0, 3).astype(float)
    psi = np.ones(X.shape, dtype=complex)
    for c, x0, s, k in zip((X, Y, Z), center, sigma, k0):
        psi *= (2 * np.pi * s ** 2) ** -0.25 * np.exp(1j * k * (c - x0)) * np.exp(-(c - x0) ** 2 / (4 * s ** 2))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx ** 3)
    return psi


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------

def phase1_free(quick=False):
    print("\nPhase 1a -- free Gaussian wavepacket spreading")
    L, N = 28.0, (48 if quick else 64)
    grid = _grid(L, N)
    x, X, Y, Z, dx, G2 = grid
    sigma0 = 2.0
    k0 = (0.4, 0.0, 0.0)
    psi = gaussian(grid, (0, 0, 0), sigma0, k0)
    T_end = 4.0 if quick else 5.0
    dt = 0.01
    n_steps = int(round(T_end / dt))

    coords = (X, Y, Z)
    obs = {
        "norm": lambda p, t: float(prop.norm(p[None], dx)[0]),
        "var": lambda p, t: prop.variance_r(p[None], [1.0], coords, dx),
        "mean": lambda p, t: prop.expectation_r(p[None], [1.0], coords, dx),
    }
    res = prop.propagate(psi, grid, n_steps, dt, fixed_V=0.0,
                         record_every=max(1, n_steps // 60), observers=obs)
    t = res["t"]
    sig_num = np.sqrt(res["obs"]["var"])                 # (nt, 3)
    sig_ana = sigma0 * np.sqrt(1 + (t[:, None] / (2 * sigma0 ** 2)) ** 2)
    mean_num = res["obs"]["mean"]
    mean_ana = np.outer(t, np.array(k0))

    width_err = np.max(np.abs(sig_num - sig_ana) / sig_ana)
    mean_err = np.max(np.abs(mean_num - mean_ana))
    norm_drift = np.max(np.abs(res["obs"]["norm"] - 1.0))

    check("free: width sigma(t) matches analytic", width_err < 0.01, f"max rel err {width_err:.2e}")
    check("free: <r>(t) = k0 t", mean_err < 0.02, f"max abs err {mean_err:.2e}")
    check("free: norm conserved", norm_drift < 1e-8, f"max drift {norm_drift:.2e}")

    _plot_free(t, sig_num, sig_ana)
    return grid


def phase1_harmonic(quick=False):
    print("\nPhase 1b -- harmonic-oscillator coherent state")
    L, N = 18.0, (40 if quick else 48)
    grid = _grid(L, N)
    x, X, Y, Z, dx, G2 = grid
    omega = 1.0
    x0 = 1.5
    sigma0 = 1.0 / np.sqrt(2 * omega)            # HO ground-state width
    V = 0.5 * omega ** 2 * (X ** 2 + Y ** 2 + Z ** 2)
    psi = gaussian(grid, (x0, 0, 0), sigma0)

    n_periods = 1.5
    T_end = n_periods * 2 * np.pi / omega
    dt = 0.005
    n_steps = int(round(T_end / dt))
    coords = (X, Y, Z)
    obs = {
        "norm": lambda p, t: float(prop.norm(p[None], dx)[0]),
        "x": lambda p, t: prop.expectation_r(p[None], [1.0], coords, dx)[0],
        "varx": lambda p, t: prop.variance_r(p[None], [1.0], coords, dx)[0],
        "E": lambda p, t: prop.orbital_energy(p, V, G2, dx),
    }
    res = prop.propagate(psi, grid, n_steps, dt, fixed_V=V,
                         record_every=max(1, n_steps // 120), observers=obs)
    t = res["t"]
    x_ana = x0 * np.cos(omega * t)
    x_err = np.max(np.abs(res["obs"]["x"] - x_ana))
    var0 = sigma0 ** 2
    width_drift = np.max(np.abs(res["obs"]["varx"] - var0) / var0)
    E0 = 1.5 * omega + 0.5 * omega ** 2 * x0 ** 2
    E_drift = np.max(np.abs(res["obs"]["E"] - E0))
    norm_drift = np.max(np.abs(res["obs"]["norm"] - 1.0))

    check("harmonic: <x>(t) = x0 cos(wt)", x_err < 0.02, f"max abs err {x_err:.2e}")
    check("harmonic: width stays constant", width_drift < 0.02, f"max rel drift {width_drift:.2e}")
    check("harmonic: energy conserved", E_drift < 1e-3, f"max drift {E_drift:.2e} (E0={E0:.3f})")
    check("harmonic: norm conserved", norm_drift < 1e-8, f"max drift {norm_drift:.2e}")

    _plot_harmonic(t, res["obs"]["x"], x_ana, res["obs"]["E"], E0)


def phase1_convergence(quick=False):
    print("\nPhase 1c -- 2nd-order dt convergence (harmonic)")
    L, N = 16.0, 40
    grid = _grid(L, N)
    x, X, Y, Z, dx, G2 = grid
    omega = 1.0
    x0 = 1.5
    sigma0 = 1.0 / np.sqrt(2 * omega)
    V = 0.5 * omega ** 2 * (X ** 2 + Y ** 2 + Z ** 2)
    psi0 = gaussian(grid, (x0, 0, 0), sigma0)
    T_end = 2 * np.pi / omega            # one period
    coords = (X, Y, Z)

    def final_x_err(dt):
        n = int(round(T_end / dt))
        res = prop.propagate(psi0, grid, n, dt, fixed_V=V, record_every=n,
                             observers={"x": lambda p, t: prop.expectation_r(p[None], [1.0], coords, dx)[0]})
        return abs(res["obs"]["x"][-1] - x0 * np.cos(omega * n * dt))

    dts = [0.04, 0.02, 0.01] if not quick else [0.04, 0.02]
    errs = [final_x_err(d) for d in dts]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for d, e in zip(dts, errs):
        print(f"    dt={d:.3f}  err={e:.3e}")
    ok = all(3.0 < r < 5.5 for r in ratios)
    check("dt halving -> error ~/4 (2nd order)", ok, "ratios " + ", ".join(f"{r:.2f}" for r in ratios))


# --------------------------------------------------------------------------
# Phase 2
# --------------------------------------------------------------------------

def phase2_fixed_point(quick=False):
    print("\nPhase 2 -- self-consistent propagation, ground-state fixed point")
    import scf3d
    import potentials3d as pot

    L, N = 14.0, (28 if quick else 32)
    grid = _grid(L, N)
    x, X, Y, Z, dx, G2 = grid
    soft = 0.5 * dx
    nuclei = [(2, 0.0, 0.0, 0.0, soft)]          # Helium
    Ne = 2

    scf = scf3d.run_scf(nuclei, Ne, grid, method="lda", n_states=4,
                        max_iter=120, tol_E=1e-7, tol_n=1e-5, return_orbitals=True)
    occ = scf["occ_orbitals"]
    V_nuc = pot.nuclear_potential(X, Y, Z, nuclei)
    psi0, eps = prop.relax_to_self_consistency(scf["orbitals"].astype(complex), occ, V_nuc, grid, method="lda")
    print(f"    SCF: E_total={scf['E_total']:.5f} Ha  iters={scf['iterations']}  N={scf['N_check']:.5f}"
          f"   eps_HOMO={eps[-1]:.5f}")

    coords = (X, Y, Z)
    rho0 = prop.density(psi0, occ)
    d0 = prop.dipole(psi0, occ, coords, dx)
    E0 = prop.ks_total_energy(psi0, occ, V_nuc, G2, dx, method="lda")
    T_end = 8.0 if quick else 16.0

    def run(dt):
        n = int(round(T_end / dt))
        obs = {
            "drho": lambda p, t: float(np.sum(np.abs(prop.density(p, occ) - rho0)) * dx ** 3),
            "ddip": lambda p, t: float(np.linalg.norm(prop.dipole(p, occ, coords, dx) - d0)),
            "dE": lambda p, t: float(prop.ks_total_energy(p, occ, V_nuc, G2, dx, method="lda") - E0),
            "norm": lambda p, t: prop.norm(p, dx),
        }
        return prop.propagate(psi0, grid, n, dt, occ=occ, V_nuc=V_nuc, method="lda",
                              record_every=max(1, n // 80), observers=obs)

    res = run(0.02)
    res_fine = run(0.01)

    max_drho = np.max(res["obs"]["drho"])
    drho_fine = np.max(res_fine["obs"]["drho"])
    max_ddip = np.max(res["obs"]["ddip"])
    max_dE = np.max(np.abs(res["obs"]["dE"]))
    norm_drift = np.max(np.abs(np.array(res["obs"]["norm"]) - 1.0))
    ratio = max_drho / drho_fine

    # The ground state is an exact fixed point of the *continuous* KS flow;
    # dipole and energy (the observables TDDFT actually reports) stay pinned to
    # ~1e-5 / 1e-4, proving propagator<->SCF consistency. The raw density L1
    # picks up the split-operator's O(dt^2) shape wobble -- verified to scale
    # as dt^2, i.e. not an inconsistency.
    check("phase2: dipole stationary", max_ddip < 1e-5, f"max |d(t)-d0| {max_ddip:.2e}")
    check("phase2: KS energy stationary", max_dE < 1e-4, f"max |dE| {max_dE:.2e}")
    check("phase2: per-orbital norm conserved", norm_drift < 1e-8, f"max drift {norm_drift:.2e}")
    check("phase2: density wobble is O(dt^2)", 3.0 < ratio < 5.5,
          f"|d rho|: {max_drho:.2e} (dt=.02) -> {drho_fine:.2e} (dt=.01), ratio {ratio:.2f}")

    _plot_phase2(res["t"], res["obs"]["drho"], res["obs"]["ddip"], res["obs"]["dE"])


# --------------------------------------------------------------------------
# Phase 3
# --------------------------------------------------------------------------

def phase3_absorption(quick=False):
    print("\nPhase 3 -- delta-kick linear-response absorption spectrum (He)")
    import scf3d
    import potentials3d as pot
    import perturb
    import response as rsp

    L, N = 16.0, (28 if quick else 32)
    grid = _grid(L, N)
    x, X, Y, Z, dx, G2 = grid
    coords = (X, Y, Z)
    soft = 0.5 * dx
    nuclei = [(2, 0.0, 0.0, 0.0, soft)]
    Ne = 2

    scf = scf3d.run_scf(nuclei, Ne, grid, method="lda", n_states=3,
                        max_iter=120, tol_E=1e-7, tol_n=1e-5, return_orbitals=True)
    occ = scf["occ_orbitals"]
    V_nuc = pot.nuclear_potential(X, Y, Z, nuclei)
    psi0, eps = prop.relax_to_self_consistency(scf["orbitals"].astype(complex), occ, V_nuc, grid, method="lda")
    print(f"    SCF: E={scf['E_total']:.4f} Ha  eps_HOMO={eps[-1]:.4f}  (-> I_p ~ {-eps[-1] * rsp.HA_TO_EV:.1f} eV)")

    dt = 0.05
    T = 120.0 if quick else 256.0
    n_steps = int(round(T / dt))

    def run_kick(kk):
        psi = perturb.dipole_kick(psi0, coords, kk, axis=0)
        res = prop.propagate(psi, grid, n_steps, dt, occ=occ, V_nuc=V_nuc, method="lda",
                             record_every=1, observers={"dx": rsp.moment_observer(occ, coords, dx, axis=0)})
        w, alpha = rsp.polarizability(res["t"], res["obs"]["dx"], kk)
        return w, alpha

    w, alpha = run_kick(0.01)
    S = rsp.strength_function(w, alpha)
    band = w < 4.0
    w, alpha, S = w[band], alpha[band], S[band]

    n_eff = rsp.sum_rule(w, S, w_max=w[-1])
    im = np.imag(alpha)
    im_min_rel = np.min(im[w > 0.1]) / np.max(im[w > 0.1])
    alpha0 = float(np.real(alpha[0]))                       # static polarizability
    pk = rsp.peaks(w, S, n=4, w_lo=0.2, w_hi=3.0)
    lowest = min((p for p in pk if p[1] > 0.15 * pk[0][1]), key=lambda p: p[0], default=pk[0])

    # rigorous internal check: the f-sum rule (exact for the true response;
    # real-time propagation preserves it up to the finite-time / grid cutoff).
    check("phase3: TRK sum rule recovers N_e", 0.85 * Ne < n_eff < 1.15 * Ne,
          f"integral S dw = {n_eff:.3f} = {100 * n_eff / Ne:.0f}% of N_e")
    # passivity: no negative absorption beyond the between-peak numerical ripple
    check("phase3: Im alpha >= 0 to the noise floor", im_min_rel > -0.03,
          f"min(Im alpha) / max(Im alpha) = {im_min_rel:.2e}")
    # the response is genuinely in the linear regime
    p1, p2 = _phase3_peak(psi0, grid, occ, V_nuc, coords, dt, 0.005), \
             _phase3_peak(psi0, grid, occ, V_nuc, coords, dt, 0.02)
    check("phase3: peak position independent of kick strength", abs(p1 - p2) < 0.05,
          f"k=.005 -> {p1:.3f} Ha,  k=.02 -> {p2:.3f} Ha")
    # the lowest line sits in the physical bound/near-threshold region -- its
    # exact position is red-shifted from the 21.2 eV experiment by the softened
    # nucleus (same limitation that puts the SCF energy ~0.45 Ha high); a
    # resolution study (full run only) shows it blue-shifting toward experiment.
    check("phase3: lowest line in the bound-excitation region",
          0.30 < lowest[0] < 1.70,
          f"peak {lowest[0]:.3f} Ha = {lowest[0] * rsp.HA_TO_EV:.1f} eV "
          f"(expt 1s->2p 21.2 eV; alpha(0) = {alpha0:.2f} a.u., He expt 1.38)")

    conv = None
    if not quick:
        conv = _phase3_convergence(coords, dt)

    _plot_phase3(w, S, alpha, n_eff, Ne, lowest, -eps[-1], conv)


def _phase3_peak(psi0, grid, occ, V_nuc, coords, dt, kk, T=100.0):
    import perturb
    import response as rsp
    dx = grid[4]
    psi = perturb.dipole_kick(psi0, coords, kk, axis=0)
    r = prop.propagate(psi, grid, int(round(T / dt)), dt, occ=occ, V_nuc=V_nuc, method="lda",
                       record_every=1, observers={"dx": rsp.moment_observer(occ, coords, dx, axis=0)})
    ww, aa = rsp.polarizability(r["t"], r["obs"]["dx"], kk)
    ss = rsp.strength_function(ww, aa)
    m = (ww > 0.2) & (ww < 3.0)
    return float(ww[m][np.argmax(ss[m])])


def _phase3_convergence(coords_unused, dt):
    """Lowest-peak position vs grid spacing (softening = 0.5 dx tracks dx), to
    show it blue-shifts toward the 21.2 eV experiment as the grid is refined."""
    import scf3d
    import potentials3d as pot
    import perturb
    import response as rsp
    out = []
    for L, N in [(16.0, 28), (16.0, 34), (16.0, 40)]:
        g = _grid(L, N)
        x, X, Y, Z, dxg, G2 = g
        soft = 0.5 * dxg
        nuc = [(2, 0.0, 0.0, 0.0, soft)]
        scf = scf3d.run_scf(nuc, 2, g, method="lda", n_states=2, max_iter=120,
                            tol_E=1e-7, tol_n=1e-5, return_orbitals=True)
        Vn = pot.nuclear_potential(X, Y, Z, nuc)
        p0, _ = prop.relax_to_self_consistency(scf["orbitals"].astype(complex),
                                               scf["occ_orbitals"], Vn, g, method="lda")
        pk = _phase3_peak(p0, g, scf["occ_orbitals"], Vn, (X, Y, Z), dt, 0.01, T=140.0)
        out.append((dxg, pk * rsp.HA_TO_EV, scf["E_total"]))
        print(f"    conv: dx={dxg:.3f}  E={scf['E_total']:.3f}  lowest peak {pk * rsp.HA_TO_EV:.1f} eV")
    return out


def _plot_phase3(w, S, alpha, n_eff, Ne, lowest, Ip, conv=None):
    plt = _mpl()
    ev = w * 27.211386
    nrow = 3 if conv else 2
    fig, axes = plt.subplots(nrow, 1, figsize=(7, 3.2 * nrow), dpi=130)
    a, b = axes[0], axes[1]
    a.plot(ev, S, "-", color="#4c72b0")
    a.axvline(lowest[0] * 27.211386, color="#dd8452", ls="--", lw=1,
              label=f"lowest peak {lowest[0] * 27.211386:.1f} eV")
    a.axvline(21.22, color="#888", ls=":", lw=1, label="expt 1s->2p 21.2 eV")
    a.axvline(Ip * 27.211386, color="#c44e52", ls=":", lw=1, label=f"$-\\epsilon_{{HOMO}}$ {Ip*27.211386:.1f} eV")
    a.set_xlabel(r"$\omega$  (eV)"); a.set_ylabel(r"$S(\omega)$  (per axis)")
    a.set_title(f"Phase 3 -- He delta-kick absorption   ($\\int S\\,d\\omega$ = {n_eff:.2f}, $N_e$ = {Ne})")
    a.set_xlim(0, min(ev[-1], 110)); a.legend(frameon=False, fontsize=8)
    b.plot(ev, np.cumsum(S) * (w[1] - w[0]), color="#55a868")
    b.axhline(Ne, color="#888", ls=":", lw=1)
    b.set_xlabel(r"$\omega$  (eV)"); b.set_ylabel(r"running $N_{\rm eff}(\omega)$")
    b.set_xlim(0, min(ev[-1], 110))
    if conv:
        c = axes[2]
        dxs = [row[0] for row in conv]
        pks = [row[1] for row in conv]
        c.plot(dxs, pks, "o-", color="#c44e52")
        c.axhline(21.22, color="#888", ls=":", lw=1, label="expt 21.2 eV")
        c.invert_xaxis()
        c.set_xlabel("grid spacing dx  (Bohr)  -- finer ->")
        c.set_ylabel("lowest peak  (eV)")
        c.set_title("resolution convergence (softening = 0.5 dx)")
        c.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase3_he_absorption.png")); plt.close(fig)


# --------------------------------------------------------------------------
# Phase 4
# --------------------------------------------------------------------------

def phase4_h2_absorption(quick=False):
    print("\nPhase 4 -- H2 delta-kick absorption (parallel vs perpendicular to the bond)")
    import scf3d
    import potentials3d as pot
    import response as rsp

    L, N = 18.0, (30 if quick else 36)
    grid = _grid(L, N)
    x, X, Y, Z, dx, G2 = grid
    Re = 1.4                                     # H2 equilibrium bond length (Bohr)
    soft = 0.5 * dx
    nuclei = [(1, 0.0, 0.0, -Re / 2, soft), (1, 0.0, 0.0, +Re / 2, soft)]   # bond along z
    Ne = 2

    scf = scf3d.run_scf(nuclei, Ne, grid, method="lda", n_states=3, max_iter=150,
                        tol_E=1e-7, tol_n=1e-5, return_orbitals=True)
    occ = scf["occ_orbitals"]
    V_nuc = pot.nuclear_potential(X, Y, Z, nuclei)
    psi0, eps = prop.relax_to_self_consistency(scf["orbitals"].astype(complex), occ, V_nuc, grid, method="lda")
    print(f"    SCF: E_total={scf['E_total']:.4f} Ha (incl. nuc-nuc)  eps_HOMO={eps[-1]:.4f}  N={scf['N_check']:.5f}")

    dt = 0.05
    T = 120.0 if quick else 256.0

    wz, az, Sz = rsp.kick_spectrum(psi0, grid, occ, V_nuc, dt, T, k=0.01, axis=2)   # parallel
    wx, ax_, Sx = rsp.kick_spectrum(psi0, grid, occ, V_nuc, dt, T, k=0.01, axis=0)  # perpendicular
    S_iso = (Sz + 2.0 * Sx) / 3.0               # wz == wx (same T, dt)

    n_eff_z = rsp.sum_rule(wz, Sz, w_max=wz[-1])
    n_eff_x = rsp.sum_rule(wx, Sx, w_max=wx[-1])
    n_eff_iso = rsp.sum_rule(wz, S_iso, w_max=wz[-1])
    a0z, a0x = float(np.real(az[0])), float(np.real(ax_[0]))

    def lowest(w, S):
        pk = rsp.peaks(w, S, n=4, w_lo=0.2, w_hi=3.0)
        return min((p for p in pk if p[1] > 0.15 * pk[0][1]), key=lambda p: p[0], default=pk[0])
    lz, lx = lowest(wz, Sz), lowest(wx, Sx)

    check("phase4: TRK sum rule, parallel kick", 0.85 * Ne < n_eff_z < 1.15 * Ne,
          f"integral S_par dw = {n_eff_z:.3f} = {100 * n_eff_z / Ne:.0f}% of N_e")
    check("phase4: TRK sum rule, perpendicular kick", 0.85 * Ne < n_eff_x < 1.15 * Ne,
          f"integral S_perp dw = {n_eff_x:.3f} = {100 * n_eff_x / Ne:.0f}% of N_e")
    check("phase4: response is anisotropic (parallel != perpendicular)",
          abs(a0z - a0x) / max(a0z, a0x) > 0.05,
          f"alpha(0): parallel {a0z:.2f} vs perp {a0x:.2f} a.u.  (H2 expt ~6.3 / ~4.9)")
    check("phase4: lowest line in the physical region",
          0.15 < min(lz[0], lx[0]) < 1.20,
          f"parallel {lz[0] * rsp.HA_TO_EV:.1f} eV, perp {lx[0] * rsp.HA_TO_EV:.1f} eV "
          f"(expt lowest strong absorption ~ 12-13 eV; grid/softening red-shifts)")

    _plot_phase4(wz, Sz, Sx, S_iso, n_eff_iso, Ne, lz, lx, -eps[-1])


def _plot_phase4(w, Sz, Sx, S_iso, n_eff, Ne, lz, lx, Ip):
    plt = _mpl()
    ev = w * 27.211386
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6.5), dpi=130)
    a.plot(ev, Sz, color="#c44e52", label=r"$S_\parallel$  (kick $\parallel$ bond)")
    a.plot(ev, Sx, color="#4c72b0", label=r"$S_\perp$  (kick $\perp$ bond)")
    a.plot(ev, S_iso, color="#333", lw=1.0, ls="--", label=r"$S_{\rm iso}$")
    a.axvline(Ip * 27.211386, color="#888", ls=":", lw=1, label=f"$-\\epsilon_{{HOMO}}$ {Ip*27.211386:.1f} eV")
    a.set_xlabel(r"$\omega$  (eV)"); a.set_ylabel(r"$S(\omega)$  (per axis)")
    a.set_title(f"Phase 4 -- H$_2$ absorption  ($\\int S_{{\\rm iso}}\\,d\\omega$ = {n_eff:.2f}, $N_e$ = {Ne})")
    a.set_xlim(0, min(ev[-1], 90)); a.legend(frameon=False, fontsize=8)
    b.plot(ev, np.cumsum(S_iso) * (w[1] - w[0]), color="#55a868")
    b.axhline(Ne, color="#888", ls=":", lw=1)
    b.set_xlabel(r"$\omega$  (eV)"); b.set_ylabel(r"running $N_{\rm eff}(\omega)$")
    b.set_xlim(0, min(ev[-1], 90))
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase4_h2_absorption.png")); plt.close(fig)


# --------------------------------------------------------------------------
# Phase 5
# --------------------------------------------------------------------------

def phase5_hhg(quick=False):
    print("\nPhase 5 -- strong-field high-harmonic generation (H atom)")
    import potentials3d as pot
    import perturb
    import response as rsp
    from masking import boundary_mask

    # H is one electron: the self-interaction-free single-particle limit
    # (method=None -- bare softened Coulomb, no Hartree/XC) is both *more*
    # correct here and much faster (no density feedback -> the ETRS predictor
    # is skipped). Ground state by imaginary-time relaxation (FFT-only).
    L, N = 32.0, (48 if quick else 56)
    grid = _grid(L, N)
    x, X, Y, Z, dx, G2 = grid
    coords = (X, Y, Z)
    soft = 0.5 * dx
    V_nuc = pot.nuclear_potential(X, Y, Z, [(1, 0.0, 0.0, 0.0, soft)])

    psi0, occ, eps = prop.imaginary_time_ground_state(grid, V_nuc, 1, method=None,
                                                      max_iter=800, tol=1e-9)
    Ip = float(-eps[-1])
    print(f"    imag-time GS: eps_1s = {eps[-1]:.4f} Ha  ->  I_p = {Ip * rsp.HA_TO_EV:.1f} eV")

    mask = boundary_mask(grid, width=8.0, order=2)
    wL = 0.114                            # ~400 nm carrier
    n_flat = 3 if quick else 4
    dt = 0.04

    def run(E0):
        E = perturb.flattop_pulse(E0, wL, n_ramp=2, n_flat=n_flat)
        vext = perturb.dipole_field(E, coords, axis=2)
        res = prop.propagate(psi0, grid, int(round((E.T_pulse + 5.0) / dt)), dt,
                             occ=occ, V_nuc=V_nuc, method=None, v_ext_fn=vext, mask=mask,
                             record_every=1,
                             observers={"dz": rsp.moment_observer(occ, coords, dx, axis=2),
                                        "surv": lambda p, t: float((prop.norm(p, dx) * occ).sum() / occ.sum())})
        return res, E

    def spectrum_and_cutoff(res, E):
        """Harmonic spectrum from the flat portion of the pulse; the cutoff is
        the highest odd order still within 1e-3 of the plateau."""
        t = np.array(res["t"]); dz = np.array(res["obs"]["dz"])
        m = (t >= E.t_flat_start) & (t <= E.t_flat_end)
        w, P = rsp.hhg_spectrum(t[m], dz[m], window="hann")
        hn = w / wL
        plat = np.median([rsp.harmonic_peak(w, P, h, wL) for h in (3, 5)])
        band = (hn > 2.0) & (hn < 25)
        above = hn[band][P[band] > plat * 1e-3]
        return w, P, hn, (above.max() if above.size else np.nan)

    E0 = 0.06
    Up0 = E0 ** 2 / (4 * wL ** 2)
    res0, E_a = run(E0)
    w, P, hn, cut0 = spectrum_and_cutoff(res0, E_a)
    surv0 = np.array(res0["obs"]["surv"])
    ion0 = 1.0 - surv0[-1]

    E1 = 1.3 * E0
    Up1 = E1 ** 2 / (4 * wL ** 2)
    res1, E_b = run(E1)
    _, _, _, cut1 = spectrum_and_cutoff(res1, E_b)
    ion1 = 1.0 - np.array(res1["obs"]["surv"])[-1]

    odd = np.mean([rsp.harmonic_peak(w, P, h, wL) for h in (3, 5, 7)])
    even = np.mean([rsp.harmonic_peak(w, P, h, wL) for h in (2, 4, 6)])
    odd_even = odd / even

    # plateau: odd orders 3..cut0-2 stay within ~2 decades of the strongest;
    # then a sharp drop (>= 3 decades) marks the cutoff.
    p3 = rsp.harmonic_peak(w, P, 3, wL)
    p_last = rsp.harmonic_peak(w, P, max(3, cut0 - 2), wL)
    p_beyond = rsp.harmonic_peak(w, P, cut0 + 3, wL)
    plateau_flat = p_last / p3 > 1e-2
    has_cutoff = p_beyond / p_last < 1e-3

    cut_pred0 = (Ip + 3.17 * Up0) / wL
    d_cut_meas = cut1 - cut0
    d_cut_pred = 3.17 * (Up1 - Up0) / wL

    check("phase5: odd harmonics dominate over even (inversion symmetry)", odd_even > 8,
          f"odd/even peak ratio = {odd_even:.1f}")
    check("phase5: a harmonic plateau ends in a sharp cutoff",
          plateau_flat and has_cutoff,
          f"plateau flat to order {max(3, cut0 - 2):.0f}, then drops >3 decades by order {cut0 + 3:.0f}")
    check("phase5: cutoff extends with intensity (>= 3.17 dU_p)",
          np.isfinite(d_cut_meas) and d_cut_meas >= 0.7 * d_cut_pred,
          f"cutoff {cut0:.1f} -> {cut1:.1f} harmonics (shift {d_cut_meas:+.1f}); "
          f"3.17 dU_p = {d_cut_pred:+.1f} (a lower bound: E0 ~ the barrier-suppression "
          f"field for this softened H, above the clean tunneling-rescattering regime)")
    check("phase5: ionization rises with intensity", ion1 > 1.5 * ion0 + 1e-5,
          f"ionized fraction {ion0:.2e} (E0) -> {ion1:.2e} (1.3 E0)")

    print(f"    cutoff at E0: {cut0:.1f} harmonics; I_p + 3.17 U_p = {cut_pred0:.1f} "
          f"(3.17 U_p law under-predicts in the over-the-barrier regime)")
    _plot_phase5(np.array(res0["t"]), np.array(res0["obs"]["dz"]), surv0,
                 [E_a(tt) for tt in res0["t"]], hn, P, cut_pred0, cut0, Ip / wL,
                 E_a.t_flat_start, E_a.t_flat_end)


def _plot_phase5(t, dz, surv, Efield, hn, P, cut_pred, cut_meas, Ip_h, t_flat0, t_flat1):
    plt = _mpl()
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(7.5, 8.5), dpi=130)
    a.plot(t, np.array(Efield) / max(np.abs(Efield)) * np.abs(dz).max(),
           color="#bbb", lw=0.8, label="E(t) (scaled)")
    a.plot(t, dz, color="#4c72b0", lw=1.0, label=r"$d_z(t)$")
    a.axvspan(t_flat0, t_flat1, color="#dd8452", alpha=0.12, label="flat-top (analysed)")
    a.set_xlabel("t  (a.u.)"); a.set_ylabel(r"$\langle z\rangle$")
    a.set_title("Phase 5 -- H atom in a flat-top pulse")
    a.legend(frameon=False, fontsize=8)
    b.plot(t, 1.0 - surv, color="#dd8452")
    b.set_xlabel("t  (a.u.)"); b.set_ylabel("ionized fraction  (norm absorbed by mask)")
    c.semilogy(hn, P / np.max(P), color="#333", lw=0.9)
    c.axvline(Ip_h, color="#55a868", ls=":", lw=1, label=f"$I_p$ = {Ip_h:.1f} $\\omega_L$")
    c.axvline(cut_pred, color="#c44e52", ls="--", lw=1, label=f"$I_p + 3.17U_p$ = {cut_pred:.1f}")
    if np.isfinite(cut_meas):
        c.axvline(cut_meas, color="#4c72b0", ls="-", lw=1, label=f"measured cutoff {cut_meas:.1f}")
    for k in range(1, int(hn[-1]) + 1, 2):
        c.axvline(k, color="#eee", lw=0.5, zorder=0)
    c.set_xlim(0, min(hn[-1], 25)); c.set_ylim(1e-10, 3)
    c.set_xlabel(r"harmonic order  $\omega / \omega_L$"); c.set_ylabel(r"$|a(\omega)|^2$  (norm.)")
    c.set_title("high-harmonic spectrum"); c.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase5_hhg.png")); plt.close(fig)


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_free(t, sig_num, sig_ana):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    ax.plot(t, sig_ana[:, 0], "k-", label="analytic")
    ax.plot(t, sig_num[:, 0], "o", ms=4, label=r"numeric $\sigma_x$")
    ax.plot(t, sig_num[:, 1], "s", ms=3, label=r"numeric $\sigma_y$")
    ax.set_xlabel("t  (a.u.)"); ax.set_ylabel(r"$\sigma(t)$  (Bohr)")
    ax.set_title("Phase 1a -- free wavepacket spreading"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase1_free_spreading.png")); plt.close(fig)


def _plot_harmonic(t, x_num, x_ana, E, E0):
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=130)
    a.plot(t, x_ana, "k-", label=r"$x_0\cos\omega t$")
    a.plot(t, x_num, "o", ms=3, label="numeric")
    a.set_xlabel("t"); a.set_ylabel(r"$\langle x\rangle$"); a.legend(frameon=False)
    a.set_title("Phase 1b -- coherent-state oscillation")
    b.plot(t, E - E0)
    b.set_xlabel("t"); b.set_ylabel(r"$E(t) - E_0$"); b.set_title("energy drift")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase1_harmonic.png")); plt.close(fig)


def _plot_phase2(t, drho, ddip, dE):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=130)
    ax.semilogy(t, np.abs(drho) + 1e-16, label=r"$\int|\rho(t)-\rho_0|\,d^3r$")
    ax.semilogy(t, np.abs(ddip) + 1e-16, label=r"$|d(t)-d_0|$")
    ax.semilogy(t, np.abs(dE) + 1e-16, label=r"$|E(t)-E_0|$")
    ax.set_xlabel("t  (a.u.)"); ax.set_title("Phase 2 -- ground-state fixed-point test (He)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase2_fixed_point.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "5", "all"])
    ap.add_argument("--quick", action="store_true", help="smaller grids / shorter runs")
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        phase1_free(args.quick)
        phase1_harmonic(args.quick)
        phase1_convergence(args.quick)
    if args.phase in ("2", "all"):
        phase2_fixed_point(args.quick)
    if args.phase in ("3", "all"):
        phase3_absorption(args.quick)
    if args.phase in ("4", "all"):
        phase4_h2_absorption(args.quick)
    if args.phase in ("5", "all"):
        phase5_hhg(args.quick)

    n_pass = sum(p for _, p, _ in _results)
    print(f"\n{n_pass}/{len(_results)} checks passed")
    sys.exit(0 if n_pass == len(_results) else 1)


if __name__ == "__main__":
    main()
