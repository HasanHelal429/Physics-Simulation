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
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    ap.add_argument("--quick", action="store_true", help="smaller grids / shorter runs")
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        phase1_free(args.quick)
        phase1_harmonic(args.quick)
        phase1_convergence(args.quick)
    if args.phase in ("2", "all"):
        phase2_fixed_point(args.quick)

    n_pass = sum(p for _, p, _ in _results)
    print(f"\n{n_pass}/{len(_results)} checks passed")
    sys.exit(0 if n_pass == len(_results) else 1)


if __name__ == "__main__":
    main()
