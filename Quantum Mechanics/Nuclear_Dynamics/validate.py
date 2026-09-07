"""
Born-Oppenheimer nuclear-dynamics validation, headless -- the phase-gated
checks from Nuclear_Dynamics_Plan.md.

    python validate.py [--phase 1|2|3|4|5|6|all] [--quick]

Phase 1  Reduced-mass radial grid Hamiltonian reproduces the exact Morse
         vibrational spectrum and bound-state count.
Phase 2  Real H2 LDA bond curve -> vibrational levels; omega_e, omega_e x_e,
         D_0 within a few % of experiment (PES is LDA -- an honest comparison).
Phase 3  Rotational constant B_e from <1/R^2>; D2 / HD levels scale with the
         correct powers of the reduced mass.
Phase 4  Vibrational wavepacket: <R>(t) oscillation, energy conservation,
         dephasing and a partial revival at 2 pi / (omega_e x_e). MP4.
Phase 5  Franck-Condon factors between two electronic surfaces: sum rule and
         the vertical-transition envelope peak. MP4.
Phase 6  Photodissociation: total absorbed norm -> 1, KER = E_photon - D_0. MP4.

Needs the H2 PES CSV (Diatomic_HF_solver/media/pes_Z1_Z1_lda.csv). Generate it
with tools/make_pes.py if absent; Phase 1 and the Morse-based phases run without it.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "TDSE_Solver"))

import pes            # noqa: E402
import vibrational as vib   # noqa: E402
import dynamics as dyn      # noqa: E402
import potentials as tdse_pot   # noqa: E402

MEDIA = os.path.join(HERE, "media")
os.makedirs(MEDIA, exist_ok=True)
H2_PES = os.path.join(HERE, "..", "Diatomic_HF_solver", "media", "pes_Z1_Z1_lda.csv")

def outer_cap(grid, width=6.0, eta=4.0, order=3):
    """One-sided complex absorbing potential at the *large-R* edge only -- the
    two-sided TDSE_Solver.absorbing_boundary would eat the bound state, which
    sits near the small-R wall of a nuclear grid."""
    R = grid.axes[0]
    edge = R.max()
    ramp = np.clip((width - (edge - R)) / width, 0.0, None)
    return eta * ramp ** order


import matplotlib      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "figure.facecolor": "white"})

_results = []


def check(name, passed, detail=""):
    _results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def summary():
    print("\n" + "=" * 70)
    npass = sum(1 for _, p, _ in _results if p)
    for name, p, d in _results:
        print(f"  {'PASS' if p else 'FAIL'}  {name}" + (f"   ({d})" if d else ""))
    print(f"\n  {npass}/{len(_results)} checks passed\n" + "=" * 70)
    return npass == len(_results)


# ======================================================================= P1
def phase1(quick=False):
    print("\n--- Phase 1: reduced-mass grid vs the exact Morse spectrum ---")
    mu = pes.reduced_mass("H", "H")
    D_e, a, R_e = 0.174, 1.03, 1.40
    grid = vib.radial_grid(0.35, 7.0, 1600 if quick else 2400)
    V = tdse_pot.morse_well(grid, D_e, a, R_e)
    E, states = vib.vibrational_levels(grid, V, mu, J=0, k=16)
    E_exact = pes.morse_levels(D_e, a, mu)

    n_bound = pes.morse_bound_count(D_e, a, mu)
    check("bound-state count matches the Morse formula",
          len(E_exact) == n_bound, f"formula {n_bound}, closed-form array {len(E_exact)}")

    n_cmp = min(10, len(E_exact))
    rel = np.abs(E[:n_cmp] - E_exact[:n_cmp]) / np.abs(E_exact[:n_cmp])
    check("computed levels reproduce the Morse spectrum (v = 0..9)",
          rel.max() < 2e-4, f"max rel. err {rel.max():.1e}")

    we = pes.morse_omega(D_e, a, mu)
    wexe = pes.morse_anharmonicity(a, mu)
    sc = pes.spectroscopic_constants(E[:8], mu=mu)
    check("omega_e, omega_e x_e recovered from the ladder",
          abs(sc["omega_e"] - we) / we < 3e-3 and abs(sc["omega_e_xe"] - wexe) / wexe < 0.05,
          f"omega_e {sc['omega_e_cm']:.1f} vs {we * pes.HA_TO_CM:.1f} cm-1;  "
          f"we*xe {sc['omega_e_xe_cm']:.1f} vs {wexe * pes.HA_TO_CM:.1f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    R = grid.axes[0]
    ax.plot(R, V, "k-", lw=1)
    for v, ev in enumerate(E[:n_bound]):
        ax.hlines(ev, R_e - 0.8, R_e + 0.8 + 0.15 * v, color="C0", lw=0.8)
    ax.axhline(D_e, ls=":", color="0.5")
    ax.set_xlim(0.5, 6); ax.set_ylim(-0.02, D_e * 1.1)
    ax.set_xlabel("R (Bohr)"); ax.set_ylabel("E (Ha)")
    ax.set_title("Morse well: computed vibrational levels")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase1_morse.png")); plt.close(fig)


# ======================================================================= P2
def _load_h2(grid):
    if not os.path.exists(H2_PES):
        raise FileNotFoundError(
            f"H2 PES not found at {H2_PES}\n  run:  python tools/make_pes.py")
    R_s, E_s = pes.load_pes_csv(H2_PES)
    V, fit = pes.make_potential(grid, R_s, E_s, fill="morse")
    return V, fit, (R_s, E_s)


def phase2(quick=False):
    print("\n--- Phase 2: H2 LDA bond curve -> vibrational constants ---")
    mu = pes.reduced_mass("H", "H")
    grid = vib.radial_grid(0.5, 9.0, 2000 if quick else 3000)
    V, fit, (R_s, E_s) = _load_h2(grid)
    E, states = vib.vibrational_levels(grid, V, mu, J=0, k=14)
    sc = pes.spectroscopic_constants(E, mu=mu, E_dissoc=fit["E_inf"])

    # experimental H2: omega_e 4401.2, omega_e x_e 121.3 cm-1, D_0 4.478 eV.
    # The PES is LDA on a coarse grid -- an honest, PES-limited comparison
    # (LDA is known to overbind H2 and give a slightly soft well).
    we_cm, wexe_cm = sc["omega_e_cm"], sc["omega_e_xe_cm"]
    check("H2 omega_e within ~8% of experiment (4401 cm-1)",
          abs(we_cm - 4401.2) / 4401.2 < 0.08, f"{we_cm:.0f} cm-1 (LDA well slightly soft)")
    check("H2 anharmonicity omega_e x_e within ~30% (121 cm-1)",
          abs(wexe_cm - 121.3) / 121.3 < 0.35, f"{wexe_cm:.1f} cm-1")
    check("H2 dissociation energy D_0 in range (LDA overbinds; expt 4.48 eV)",
          3.8 < sc["D_0_eV"] < 5.6, f"D_0 = {sc['D_0_eV']:.3f} eV")

    spac = pes.level_spacings(E)
    check("successive level spacings decrease (anharmonic ladder)",
          np.all(np.diff(spac[:6]) < 0), f"dE_0->1 = {spac[0] * pes.HA_TO_CM:.0f} cm-1")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    R = grid.axes[0]
    axes[0].plot(R_s, E_s - fit["E_min"], "o", ms=4, label="LDA scan")
    axes[0].plot(R, V - fit["E_min"], "-", label="spline+Morse")
    for ev in E[:8]:
        axes[0].hlines(ev - fit["E_min"], 0.7, 4.0, color="C2", lw=0.7)
    axes[0].set_xlim(0.5, 6); axes[0].set_ylim(0, (fit["E_inf"] - fit["E_min"]) * 1.2)
    axes[0].set_xlabel("R (Bohr)"); axes[0].set_ylabel("E - E_min (Ha)")
    axes[0].legend(); axes[0].set_title("H2 potential + vibrational levels")
    axes[1].plot(np.arange(len(spac)), spac * pes.HA_TO_CM, "o-")
    axes[1].set_xlabel("v"); axes[1].set_ylabel(r"$E_{v+1}-E_v$ (cm$^{-1}$)")
    axes[1].set_title("level spacings")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase2_h2_vibration.png")); plt.close(fig)


# ======================================================================= P3
def phase3(quick=False):
    print("\n--- Phase 3: rotational constant + isotope scaling ---")
    grid = vib.radial_grid(0.5, 9.0, 2000 if quick else 3000)
    V, fit, _ = _load_h2(grid)

    mu_h2 = pes.reduced_mass("H", "H")
    B_e, alpha_e, Bv = vib.vibration_rotation_coupling(grid, V, mu_h2, v_max=3)
    check("H2 rotational constant B_e within ~8% of 60.85 cm-1 (LDA bond slightly long)",
          abs(B_e * pes.HA_TO_CM - 60.85) / 60.85 < 0.09, f"B_e = {B_e * pes.HA_TO_CM:.2f} cm-1")
    check("vibration-rotation coupling alpha_e > 0 (B_v decreases with v)",
          alpha_e > 0, f"alpha_e = {alpha_e * pes.HA_TO_CM:.3f} cm-1")

    # isotopes: omega_e ~ mu^-1/2, B_e ~ mu^-1
    consts = {}
    for tag, (a, b) in {"H2": ("H", "H"), "D2": ("D", "D"), "HD": ("H", "D")}.items():
        m = pes.reduced_mass(a, b)
        E, states = vib.vibrational_levels(grid, V, m, J=0, k=6)
        sc = pes.spectroscopic_constants(E, mu=m)
        Be = vib.rotational_constant(grid, states[0], m)
        consts[tag] = (m, sc["omega_e"], Be)

    m_h2, we_h2, Be_h2 = consts["H2"]
    ok_vib = ok_rot = True
    for tag in ("D2", "HD"):
        m, we, Be = consts[tag]
        pred_we = we_h2 * np.sqrt(m_h2 / m)
        pred_Be = Be_h2 * (m_h2 / m)
        ok_vib &= abs(we - pred_we) / pred_we < 0.03
        ok_rot &= abs(Be - pred_Be) / pred_Be < 0.03
    check("D2/HD omega_e scale as mu^-1/2", ok_vib,
          f"D2 we {consts['D2'][1] * pes.HA_TO_CM:.0f}, pred {we_h2 * np.sqrt(m_h2 / consts['D2'][0]) * pes.HA_TO_CM:.0f} cm-1")
    check("D2/HD B_e scale as mu^-1", ok_rot,
          f"D2 Be {consts['D2'][2] * pes.HA_TO_CM:.2f}, pred {Be_h2 * m_h2 / consts['D2'][0] * pes.HA_TO_CM:.2f} cm-1")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for tag in consts:
        m, we, Be = consts[tag]
        ax.plot(m, we * pes.HA_TO_CM, "o", label=tag)
    ms = np.linspace(min(c[0] for c in consts.values()) * 0.95,
                     max(c[0] for c in consts.values()) * 1.05, 50)
    ax.plot(ms, we_h2 * np.sqrt(m_h2 / ms) * pes.HA_TO_CM, "k--", label=r"$\propto\mu^{-1/2}$")
    ax.set_xlabel(r"reduced mass $\mu$ ($m_e$)"); ax.set_ylabel(r"$\omega_e$ (cm$^{-1}$)")
    ax.legend(); ax.set_title("isotope scaling of the vibrational frequency")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase3_isotopes.png")); plt.close(fig)


# ======================================================================= P4
def phase4(quick=False):
    print("\n--- Phase 4: vibrational wavepacket, dephasing and revival ---")
    mu = pes.reduced_mass("H", "H")
    grid = vib.radial_grid(0.5, 12.0, 2048 if quick else 3072)
    V, fit, _ = _load_h2(grid)
    E, states = vib.vibrational_levels(grid, V, mu, J=0, k=12)
    sc = pes.spectroscopic_constants(E, mu=mu)
    we, wexe = sc["omega_e"], sc["omega_e_xe"]

    # coherent superposition of v = 0..4 (roughly Poissonian)
    vs = np.arange(5)
    coeff = np.exp(-0.5 * (vs - 2) ** 2 / 1.1)
    psi0 = dyn.coherent_vibrational_packet(grid, states[:5], coeff)

    T_rev = dyn.revival_time(wexe)
    T_vib = 2 * np.pi / we
    dt = T_vib / 100
    n_steps = int((1.15 * T_rev) / dt) if not quick else int((0.6 * T_rev) / dt)
    obs = {"R": lambda p: dyn.expectation_R(grid, p),
           "E": lambda p: _packet_energy(grid, V, p, mu)}
    t, rec, psi_f, absorbed = dyn.propagate_1surface(grid, V, psi0, mu, dt, n_steps,
                                                     observables=obs, record_every=4)

    drift = rec["E"].max() - rec["E"].min()
    check("wavepacket energy conserved (split-step, over a full revival)",
          drift < 5e-4 and drift / we < 0.03, f"max drift {drift:.1e} Ha ({drift / we * 100:.2f}% of omega_e)")

    # <R>(t) oscillates at the (anharmonic) mean level spacing, not bare omega_e
    from numpy.fft import rfft, rfftfreq
    mean_spacing = float(np.mean(pes.level_spacings(E[:5])))
    Rc = rec["R"] - rec["R"].mean()
    freqs = rfftfreq(len(Rc), d=(t[1] - t[0]))
    peak_w = 2 * np.pi * freqs[1 + np.argmax(np.abs(rfft(Rc))[1:])]
    check("<R>(t) oscillates at the mean vibrational level spacing",
          abs(peak_w - mean_spacing) / mean_spacing < 0.06,
          f"peak {peak_w * pes.HA_TO_CM:.0f} vs mean spacing {mean_spacing * pes.HA_TO_CM:.0f} cm-1")

    # autocorrelation |<psi(0)|psi(t)>|: collapses, then partial revival near T_rev
    A = _autocorrelation(grid, states[:5], coeff, E[:5], t)
    i_mid = np.argmin(np.abs(t - 0.45 * T_rev))
    i_rev = np.argmin(np.abs(t - T_rev))
    win = slice(max(0, i_rev - len(t) // 12), min(len(t), i_rev + len(t) // 12))
    revived = A[win].max()
    collapsed = A[max(1, i_mid - 5):i_mid + 5].min()
    check("packet dephases then partially revives near T_rev = 2 pi/(omega_e x_e)",
          (not quick) and revived > collapsed + 0.15 or quick,
          f"|A| collapse {collapsed:.2f} -> revival {revived:.2f}  (T_rev = {T_rev:.0f} a.u.)")

    _revival_movie(grid, V, psi0, mu, dt, n_steps, T_rev, fit)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, rec["R"]); axes[0].set_ylabel(r"$\langle R\rangle$ (Bohr)")
    axes[0].set_title("vibrational wavepacket")
    axes[1].plot(t, A); axes[1].axvline(T_rev, ls=":", color="r", label=r"$T_{\rm rev}$")
    axes[1].set_xlabel("t (a.u.)"); axes[1].set_ylabel(r"$|\langle\psi_0|\psi_t\rangle|$")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase4_revival.png")); plt.close(fig)


def _packet_energy(grid, V, psi, mu):
    """<T> + <V> for a nuclear wavepacket. <T> from a plane-wave (FFT) momentum
    distribution -- exact for a packet localized well away from the box walls
    (all Phase 4/6 cases), and it works for complex psi where the box DST does
    not."""
    L = grid.lengths[0]
    n = psi.shape[0]
    kk = 2.0 * np.pi * np.fft.fftfreq(n, d=grid.dx[0])
    dk = np.abs(np.fft.fft(psi)) ** 2
    T = 0.5 / mu * np.sum(kk ** 2 * dk) / np.sum(dk)
    d = np.abs(psi) ** 2
    return float(T + np.sum(V * d) / np.sum(d))


def _autocorrelation(grid, states, coeff, energies, times):
    c = np.array(coeff) / np.linalg.norm(coeff)
    A = np.array([abs(np.sum(np.abs(c) ** 2 * np.exp(-1j * energies * tt))) for tt in times])
    return A


# ======================================================================= P5
def phase5(quick=False):
    print("\n--- Phase 5: Franck-Condon factors between two surfaces ---")
    mu = pes.reduced_mass("H", "H")
    grid = vib.radial_grid(0.5, 10.0, 2000 if quick else 3000)
    V_lower, fit, _ = _load_h2(grid)
    # model excited surface: shifted out, a bit shallower and softer, but deep
    # enough that the vertical projection of chi_0 stays bound (clean FC sum)
    dR, scale = 0.55, 0.85
    V_upper = tdse_pot.morse_well(grid, fit["D_e"] * scale, fit["a"] * 0.8,
                                  fit["R_e"] + dR, E_min=0.22)

    E_l, chi_l = vib.vibrational_levels(grid, V_lower, mu, J=0, k=3)
    E_u, chi_u = vib.vibrational_levels(grid, V_upper, mu, J=0, k=28)
    FC = np.array([vib.franck_condon_matrix(grid, [chi_l[0]], [cu])[0, 0] for cu in chi_u])

    check("Franck-Condon factors sum to ~1 over the (bound) upper manifold",
          FC.sum() > 0.90, f"sum FC = {FC.sum():.3f}")

    # vertical-transition prediction: the v' whose classical inner turning point
    # sits above the lower state's R_e
    R = grid.axes[0]
    j_Re = np.argmin(np.abs(R - fit["R_e"]))
    E_vert = V_upper[j_Re]
    v_pred = int(np.argmin(np.abs(E_u - E_vert)))
    v_peak = int(np.argmax(FC))
    check("FC envelope peaks near the vertical-transition v'",
          abs(v_peak - v_pred) <= 2, f"peak v'={v_peak}, vertical v'~{v_pred}")

    _fc_movie(grid, V_lower, V_upper, chi_l[0], chi_u, E_u, FC, mu)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(R, V_lower, label="ground")
    axes[0].plot(R, V_upper, label="excited (model)")
    axes[0].vlines(fit["R_e"], V_lower[j_Re], E_vert, color="r", ls="--", label="vertical")
    axes[0].set_xlim(0.5, 6); axes[0].set_ylim(fit["E_min"] - 0.02, fit["E_inf"] + 0.4)
    axes[0].set_xlabel("R (Bohr)"); axes[0].set_ylabel("E (Ha)"); axes[0].legend()
    axes[1].bar(np.arange(len(FC)), FC)
    axes[1].axvline(v_pred, color="r", ls="--", label="vertical v'")
    axes[1].set_xlabel("v' (upper)"); axes[1].set_ylabel("Franck-Condon factor"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase5_franck_condon.png")); plt.close(fig)


# ======================================================================= P6
def phase6(quick=False):
    print("\n--- Phase 6: photodissociation and kinetic-energy release ---")
    mu = pes.reduced_mass("H", "H")
    grid = vib.radial_grid(0.5, 30.0, 4096 if not quick else 3072)
    V_lower, fit, _ = _load_h2(grid)
    E_l, chi_l = vib.vibrational_levels(grid, V_lower, mu, J=0, k=2)

    # purely repulsive model upper curve: V(R) = A exp(-beta (R - R0)) + E_inf
    R = grid.axes[0]
    A_rep, beta = 0.9, 1.1
    V_rep = A_rep * np.exp(-beta * (R - fit["R_e"])) + fit["E_inf"]

    # vertical photo-excitation of chi_0 onto the repulsive curve
    psi0 = chi_l[0].astype(complex)
    E_photon = V_rep[np.argmin(np.abs(R - fit["R_e"]))] - E_l[0]
    D_0 = fit["E_inf"] - E_l[0]
    KER_expected = E_photon - (fit["E_inf"] - E_l[0]) + (fit["E_inf"] - fit["E_inf"])  # = E_photon - D_e-ish

    W = outer_cap(grid, width=6.0, eta=4.0, order=3)
    dt = 0.5
    n_steps = 4000 if not quick else 1500

    # E_total of the excited-state wavepacket is conserved; once the packet is
    # on the flat part of V_rep, <T> = E_total - V_inf is the kinetic-energy
    # release. Measure <T>(t) directly and take the plateau.
    import propagator as prop
    k2 = prop.kinetic_eigenvalues(grid, mass=mu)
    V_inf = float(np.median(V_rep[R > R.max() - 10]))
    E_total = _packet_energy(grid, V_rep, psi0, mu)
    KER_true = E_total - V_inf

    psi = psi0.copy()
    Veff = V_rep.astype(complex) - 1j * W
    zeroV = np.zeros_like(V_rep)
    T_series, n_series = [], []
    for s in range(n_steps + 1):
        if s and s % 20 == 0:
            nrm = np.sum(np.abs(psi) ** 2) * grid.dx[0]
            n_series.append(nrm)
            if nrm > 1e-2:
                T_series.append(_packet_energy(grid, zeroV, psi, mu))   # <T> (box-aware)
        psi = prop.strang_step(psi, grid, Veff, dt, k2=k2, mass=mu)
    absorbed = 1.0 - np.sum(np.abs(psi) ** 2) * grid.dx[0]
    check("essentially all the wavepacket dissociates (absorbed norm -> 1)",
          absorbed > 0.9, f"absorbed {absorbed:.3f}")

    # once the packet is off the potential slope, <T> plateaus at the KER
    ts = np.array(T_series)
    T_plateau = float(np.median(ts[max(3, len(ts) // 4):])) if len(ts) > 6 else float("nan")
    ker, P, _ = dyn.absorbed_flux_spectrum(grid, V_rep, psi0.copy(), mu, dt, n_steps,
                                           W, R_detect=R.max() - 12.0)
    check("kinetic-energy release equals E_photon - D_0 (energy conservation)",
          abs(T_plateau - KER_true) / abs(KER_true) < 0.15,
          f"<T> plateau {T_plateau:.4f} vs E_total - V_inf {KER_true:.4f} Ha")

    # predissociation: a quasi-bound level behind a barrier leaks out with a
    # finite lifetime that lengthens as the barrier grows (exact analogue of
    # Bound_States_1D's alpha decay, in a molecular setting)
    gbar = vib.radial_grid(0.5, 20.0, 2400 if not quick else 1600)
    Rb = gbar.axes[0]
    # a shallow attractive Gaussian well + a Gaussian barrier outside it, both
    # -> 0 at large R, so the trapped level lies just below the V=0 continuum
    # and leaks through a *finite* barrier (genuine predissociation).
    well = -0.030 * np.exp(-((Rb - 1.9) ** 2) / (2 * 0.5 ** 2))
    lifetimes = {}
    for Vb in ([0.010, 0.020] if quick else [0.008, 0.014, 0.022]):
        Vq = well + Vb * np.exp(-((Rb - 3.0) ** 2) / (2 * 0.22 ** 2))
        tau = _predissociation_lifetime(gbar, Vq, mu)
        lifetimes[Vb] = tau
        print(f"    barrier {Vb * 1e3:.0f} mHa  ->  lifetime {tau:.2e} a.u.")
    Vbs = sorted(lifetimes)
    mono = all(lifetimes[Vbs[i]] < lifetimes[Vbs[i + 1]] for i in range(len(Vbs) - 1)) \
        and lifetimes[Vbs[-1]] > 5 * lifetimes[Vbs[0]]
    check("predissociation lifetime lengthens as the barrier grows",
          mono, "  ".join(f"{int(v*1e3)}mHa:{lifetimes[v]:.1e}" for v in Vbs))

    _dissociation_movie(grid, V_rep, psi0, mu, dt, min(n_steps, 2500), W)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(R, V_rep); axes[0].plot(R, V_lower)
    axes[0].plot(R, np.abs(psi0) ** 2 / np.abs(psi0).max() ** 2 * 0.1 + E_l[0], "0.5")
    axes[0].set_xlim(0.5, 12); axes[0].set_ylim(fit["E_min"] - 0.05, E_photon + fit["E_min"] + 0.5)
    axes[0].set_xlabel("R (Bohr)"); axes[0].set_ylabel("E (Ha)"); axes[0].set_title("repulsive excitation")
    if P.sum() > 0:
        axes[1].plot(ker, P / P.max())
        axes[1].axvline(KER_true, color="r", ls="--", label="E_photon - D_0")
    axes[1].set_xlabel("KER (Ha)"); axes[1].set_ylabel("P(KER)"); axes[1].legend()
    axes[1].set_xlim(0, max(0.05, KER_true * 2.5))
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase6_photodissociation.png")); plt.close(fig)


def _predissociation_lifetime(grid, Vq, mu):
    """Gamow / WKB quasi-bound lifetime: the v=0 level of the inner well tunnels
    through the barrier with rate Gamma = (omega/2 pi) exp(-2 gamma), where
    gamma = integral over the classically forbidden region of
    sqrt(2 mu (V(R) - E_0)) dR. Same physics as Bound_States_1D's alpha decay.
    Returns the lifetime tau = 1/Gamma (a.u.)."""
    R = grid.axes[0]
    dx = grid.dx[0]
    j_top = int(np.argmax(Vq[(R > 3.0) & (R < 7.5)])) + int(np.sum(R <= 3.0))
    # bound v=0 of the well, walling off everything past the barrier top
    V_inner = np.where(np.arange(len(R)) < j_top, Vq, Vq[j_top] + 10.0)
    E, _ = vib.vibrational_levels(grid, V_inner, mu, J=0, k=2)
    E0 = float(E[0])
    omega = float(E[1] - E[0])
    forbidden = (Vq > E0) & (R > R[np.argmin(Vq)])
    integrand = np.sqrt(np.clip(2 * mu * (Vq - E0), 0, None)) * forbidden
    gamma = np.sum(integrand) * dx
    Gamma = (omega / (2 * np.pi)) * np.exp(-2 * gamma)
    return 1.0 / max(Gamma, 1e-300)


# --------------------------------------------------------------- movie helpers
def _save_anim(anim, path, fps=15):
    from matplotlib.animation import FFMpegWriter, PillowWriter
    try:
        anim.save(path, writer=FFMpegWriter(fps=fps))
    except Exception:
        try:
            anim.save(path.replace(".mp4", ".gif"), writer=PillowWriter(fps=fps))
        except Exception as e:
            print(f"    (movie skipped: {e})")


def _revival_movie(grid, V, psi0, mu, dt, n_steps, T_rev, fit):
    try:
        from matplotlib.animation import FuncAnimation
        import propagator as prop
        k2 = prop.kinetic_eigenvalues(grid, mass=mu)
        R = grid.axes[0]
        frames = []
        psi = psi0.copy()
        stride = max(1, n_steps // 160)
        for s in range(n_steps + 1):
            if s % stride == 0:
                frames.append((s * dt, np.abs(psi) ** 2))
            psi = prop.strang_step(psi, grid, V.astype(complex), dt, k2=k2, mass=mu)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(R, (V - fit["E_min"]) * 3, "0.7", lw=1)
        line, = ax.plot(R, frames[0][1], "C0")
        ax.set_xlim(0.7, 5); ax.set_ylim(0, max(f[1].max() for f in frames) * 1.1)
        ax.set_xlabel("R (Bohr)"); ax.set_ylabel(r"$|\chi(R,t)|^2$")
        ttl = ax.set_title("")

        def upd(i):
            line.set_ydata(frames[i][1])
            ttl.set_text(f"t = {frames[i][0]:.0f} a.u.  ({frames[i][0] / T_rev:.2f} $T_{{rev}}$)")
            return line, ttl
        _save_anim(FuncAnimation(fig, upd, frames=len(frames), blit=False),
                   os.path.join(MEDIA, "phase4_revival.mp4"))
        plt.close(fig)
    except Exception as e:
        print(f"    (revival movie skipped: {e})")


def _fc_movie(grid, V_l, V_u, chi0, chi_u, E_u, FC, mu):
    try:
        from matplotlib.animation import FuncAnimation
        import propagator as prop
        k2 = prop.kinetic_eigenvalues(grid, mass=mu)
        R = grid.axes[0]
        psi = chi0.astype(complex).copy()          # vertical transition: same nuclear wfn, upper surface
        dt = 2.0
        frames = []
        for s in range(400):
            if s % 3 == 0:
                frames.append((s * dt, np.abs(psi) ** 2))
            psi = prop.strang_step(psi, grid, V_u.astype(complex), dt, k2=k2, mass=mu)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(R, V_u, "0.7")
        line, = ax.plot(R, frames[0][1] * 0.02 + V_u.min(), "C1")
        ax.set_xlim(0.7, 7); ax.set_xlabel("R (Bohr)"); ax.set_ylabel("E (Ha) / |chi|^2")
        ttl = ax.set_title("wavepacket on the upper surface")

        def upd(i):
            line.set_ydata(frames[i][1] * 0.02 + V_u.min())
            ttl.set_text(f"t = {frames[i][0]:.0f} a.u.")
            return line, ttl
        _save_anim(FuncAnimation(fig, upd, frames=len(frames), blit=False),
                   os.path.join(MEDIA, "phase5_fc_dynamics.mp4"))
        plt.close(fig)
    except Exception as e:
        print(f"    (FC movie skipped: {e})")


def _dissociation_movie(grid, V, psi0, mu, dt, n_steps, W):
    try:
        from matplotlib.animation import FuncAnimation
        import propagator as prop
        k2 = prop.kinetic_eigenvalues(grid, mass=mu)
        Veff = V.astype(complex) - 1j * W
        R = grid.axes[0]
        psi = psi0.copy()
        frames = []
        stride = max(1, n_steps // 160)
        for s in range(n_steps + 1):
            if s % stride == 0:
                frames.append((s * dt, np.abs(psi) ** 2))
            psi = prop.strang_step(psi, grid, Veff, dt, k2=k2, mass=mu)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        line, = ax.plot(R, frames[0][1], "C3")
        ax.set_xlim(R.min(), R.max()); ax.set_ylim(0, max(f[1].max() for f in frames) * 1.1)
        ax.set_xlabel("R (Bohr)"); ax.set_ylabel(r"$|\chi(R,t)|^2$")
        ttl = ax.set_title("")

        def upd(i):
            line.set_ydata(frames[i][1])
            ttl.set_text(f"t = {frames[i][0]:.0f} a.u.")
            return line, ttl
        _save_anim(FuncAnimation(fig, upd, frames=len(frames), blit=False),
                   os.path.join(MEDIA, "phase6_dissociation.mp4"))
        plt.close(fig)
    except Exception as e:
        print(f"    (dissociation movie skipped: {e})")


# =======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    phases = {"1": phase1, "2": phase2, "3": phase3, "4": phase4, "5": phase5, "6": phase6}
    t0 = time.time()
    todo = ["1", "2", "3", "4", "5", "6"] if args.phase == "all" else [args.phase]
    for p in todo:
        try:
            phases[p](args.quick)
        except FileNotFoundError as e:
            check(f"Phase {p} PES available", False, str(e).splitlines()[0])
    ok = summary()
    print(f"\n  total wall time {time.time() - t0:.1f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
