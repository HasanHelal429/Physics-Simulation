"""
Tensor-network validation, headless -- the phase-gated checks from
Tensor_Networks_Plan.md.

    python validate.py [--phase 1|2|3|4|5|6|all] [--quick]

Phase 1  MPS <-> ED round-trip fidelity vs chi; canonical-form identities.
Phase 2  DMRG vs ED (1e-8) and the analytic TFIM infinite-chain energy;
         discarded weight bounds the energy error.
Phase 3  Heisenberg: spin-1/2 vs Bethe ansatz; spin-1 Haldane gap ~ 0.41 J,
         4-fold open-chain edge manifold, string order with vanishing Neel order.
Phase 4  TFIM criticality: block entanglement entropy vs the conformal formula,
         slope gives the central charge c = 1/2.
Phase 5  TEBD quench: light-cone velocity 2 min(g,1); linear entanglement
         growth; imaginary-time TEBD reproduces the DMRG ground-state energy.
Phase 6  1D Hubbard Mott gap via a chemical-potential scan vs Lieb-Wu;
         spin/charge correlation spreading; Hydrogen-chain dissociation.

Writes figures/movies to media/ and prints a PASS/FAIL table.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import models          # noqa: E402
import ed              # noqa: E402
import mps as mps_mod  # noqa: E402
import dmrg            # noqa: E402
import tebd            # noqa: E402

MEDIA = os.path.join(HERE, "media")
os.makedirs(MEDIA, exist_ok=True)

import matplotlib      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

_results = []


def check(name, passed, detail=""):
    _results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def summary():
    print("\n" + "=" * 70)
    npass = sum(1 for _, p, _ in _results if p)
    for name, p, detail in _results:
        print(f"  {'PASS' if p else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    print(f"\n  {npass}/{len(_results)} checks passed")
    print("=" * 70)
    return npass == len(_results)


# ======================================================================= P1
def phase1(quick=False):
    print("\n--- Phase 1: MPS <-> ED round-trip + canonical identities ---")
    n = 10 if not quick else 8
    w, v = ed.ground_state("tfim", n, g=1.2)
    psi_exact = v[:, 0]

    chis = [1, 2, 4, 8, 16, 32]
    infid = []
    for chi in chis:
        m = mps_mod.MPS.from_statevector(psi_exact, 2, n, chi_max=chi)
        m.canonicalize(chi_max=chi)
        m.normalize()
        f = abs(np.vdot(psi_exact, m.to_statevector())) ** 2
        infid.append(abs(1.0 - f))
    monotone = all(infid[i + 1] <= infid[i] * 1.5 + 1e-14 for i in range(len(infid) - 1))
    check("round-trip infidelity decreases with chi", monotone,
          "  ".join(f"chi{c}:{e:.1e}" for c, e in zip(chis, infid)))
    check("exact at full Schmidt rank (chi=32)", infid[-1] < 1e-10, f"1-F = {infid[-1]:.2e}")

    # canonical-form identity: left-canonical A -> sum_s A_s^dag A_s = I
    m = mps_mod.MPS.random(n, 2, 16, seed=3)
    m.move_center_to(n - 1)
    max_err = 0.0
    for i in range(n - 1):
        A = m.M[i]
        g = np.tensordot(A.conj(), A, axes=([0, 1], [0, 1]))
        max_err = max(max_err, np.abs(g - np.eye(g.shape[0])).max())
    check("left-canonical isometry  sum A^dag A = I", max_err < 1e-12, f"max dev {max_err:.2e}")

    # entanglement entropy MPS vs ED
    s_mps = mps_mod.MPS.from_statevector(psi_exact, 2, n, chi_max=64)
    s_mps.canonicalize(); s_mps.normalize()
    e_mps = s_mps.entanglement_entropy(n // 2 - 1)
    e_ed = ed.entanglement_entropy(psi_exact, n // 2, 2, n)
    check("entanglement entropy matches ED", abs(e_mps - e_ed) < 1e-9,
          f"MPS {e_mps:.6f}  ED {e_ed:.6f}")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.semilogy(chis, infid, "o-")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"$1 - |\langle\psi_{\rm ED}|\psi_{\rm MPS}\rangle|^2$")
    ax.set_title(f"MPS compression of the TFIM ground state (N={n}, g=1.2)")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase1_compression.png")); plt.close(fig)


# ======================================================================= P2
def phase2(quick=False):
    print("\n--- Phase 2: DMRG vs ED and the analytic TFIM energy ---")
    n = 12 if not quick else 10
    for g in (0.5, 1.0, 1.5):
        w, _ = ed.ground_state("tfim", n, g=g)
        res = dmrg.run_dmrg(models.tfim_mpo(n, g), n, 2, chi_max=32,
                            n_sweeps=16, tol=1e-12, verbose=False)
        check(f"DMRG = ED  (N={n}, g={g})", abs(res.energy - w[0]) < 1e-8,
              f"dE = {abs(res.energy - w[0]):.1e}")

    # approach to the thermodynamic limit
    Ns = [16, 32, 64] if not quick else [16, 32]
    e_site, e_exact = [], models.tfim_energy_per_site(1.0)
    for N in Ns:
        res = dmrg.run_dmrg(models.tfim_mpo(N, 1.0), N, 2, chi_max=48,
                            n_sweeps=20, tol=1e-11, verbose=False)
        e_site.append(res.energy / N)
    trend = abs(e_site[-1] - e_exact) < abs(e_site[0] - e_exact)
    check("energy/site -> analytic infinite-chain value at g=1", trend,
          f"N{Ns[-1]}: {e_site[-1]:.5f}  vs  {e_exact:.5f}")

    # discarded weight bounds the energy error
    N = 40
    w_ref = None
    rows = []
    for chi in [4, 8, 16, 32]:
        res = dmrg.run_dmrg(models.tfim_mpo(N, 1.0), N, 2, chi_max=chi,
                            n_sweeps=14, tol=1e-12, verbose=False)
        rows.append((chi, res.energy, res.max_discarded))
    e_best = min(r[1] for r in rows)
    ok = all((r[1] - e_best) <= 50 * r[2] * N + 1e-6 or r[2] < 1e-12 for r in rows)
    check("energy error tracks discarded weight", ok,
          "  ".join(f"chi{c}:dE={e - e_best:.1e},w={d:.1e}" for c, e, d in rows))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(Ns, e_site, "o-", label="DMRG")
    axes[0].axhline(e_exact, ls="--", color="k", label="free-fermion")
    axes[0].set_xlabel("N"); axes[0].set_ylabel("E / N"); axes[0].set_title("TFIM g=1"); axes[0].legend()
    axes[1].loglog([r[2] for r in rows], [r[1] - e_best for r in rows], "o-")
    axes[1].set_xlabel("discarded weight"); axes[1].set_ylabel(r"$E-E_{\rm best}$")
    axes[1].set_title(f"N={N}, chi scan")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase2_dmrg.png")); plt.close(fig)


# ======================================================================= P3
def phase3(quick=False):
    print("\n--- Phase 3: Heisenberg chains, Haldane gap, edge modes ---")
    # --- spin-1/2 vs Bethe ansatz (1/N^2 finite-size extrapolation) ---
    Ns = [24, 40, 60] if not quick else [16, 28]
    es = []
    for N in Ns:
        res = dmrg.run_dmrg(models.heisenberg_mpo(N, 0.5), N, 2, chi_max=56,
                            n_sweeps=16, tol=1e-10, chi_schedule=[16, 32, 48, 56],
                            verbose=False)
        es.append(res.energy / N)
    slope, inter = np.polyfit(1.0 / np.array(Ns) ** 2, es, 1)
    check("spin-1/2 energy/site -> Bethe ansatz (1/4 - ln2)",
          abs(inter - models.HEISENBERG_HALF_E0) < 4e-3,
          f"extrap {inter:.5f}  vs  {models.HEISENBERG_HALF_E0:.5f}")

    # --- spin-1 Haldane gap: spin-1/2-capped chain, singlet-triplet splitting ---
    bulks = [16, 32] if not quick else [10, 18]
    gaps = []
    for nb in bulks:
        spins = [0.5] + [1.0] * nb + [0.5]
        dims = [2] + [3] * nb + [2]
        init = mps_mod.MPS.random_mixed(dims, 8, seed=2)
        lv = dmrg.excited_states(models.heisenberg_chain_mpo(spins), nb + 2, 3,
                                 chi_max=40, n_levels=2, n_sweeps=10, tol=1e-8,
                                 chi_schedule=[12, 24, 40], init=init, time_budget=150)
        gaps.append(lv[1].energy - lv[0].energy)
        print(f"    capped bulk={nb}: gap = {gaps[-1]:.4f}")
    g_slope, g_inf = np.polyfit(1.0 / np.array(bulks), gaps, 1)
    check("spin-1 Haldane gap ~ 0.4105 J (capped-chain extrapolation)",
          abs(g_inf - models.HALDANE_GAP) < 0.05,
          f"extrap {g_inf:.4f}  (finite {gaps[-1]:.4f})  vs  {models.HALDANE_GAP:.4f}")

    # --- string order vs Neel order (uniform spin-1 chain) ---
    N = 24 if not quick else 14
    res = dmrg.run_dmrg(models.heisenberg_mpo(N, 1.0), N, 3, chi_max=48,
                        n_sweeps=12, tol=1e-9, chi_schedule=[16, 32, 48], verbose=False)
    psi = res.mps
    _, _, Sz1, _, _, _ = models.spin_operators(1.0)
    i0, j0 = N // 2 - 5, N // 2 + 5
    string = _string_order(psi, Sz1, i0, j0)
    neel = abs(psi.correlator(Sz1, i0, Sz1, j0).real)
    check("spin-1 string order finite, Neel order strongly suppressed",
          abs(string) > 0.28 and neel < 0.35 * abs(string),
          f"O_string = {string:.3f}   |<Sz Sz>|_(r={j0-i0}) = {neel:.4f}  (ratio {neel/abs(string):.2f})")

    # --- fractional edge spins: weak uniform field polarizes the edge modes ---
    res_h = dmrg.run_dmrg(models.heisenberg_mpo(N, 1.0, hz=0.05), N, 3, chi_max=48,
                          n_sweeps=10, tol=1e-8, chi_schedule=[16, 32, 48], verbose=False)
    prof = res_h.mps.density_profile(Sz1)
    edge_spin = prof[:3].sum()
    check("edge magnetization ~ 1/2 per end, localized (weak field)",
          0.20 < edge_spin < 0.80 and abs(prof[N // 2]) < 0.06,
          f"sum<Sz>_(first 3 sites) = {edge_spin:.3f},  bulk <Sz> = {prof[N // 2]:.1e}")

    # --- 4-fold quasi-degenerate ground manifold (uncapped, small N via ED) ---
    Nd = 10
    wED, _ = ed.ground_state("heisenberg", Nd, S=1.0, k=6)
    splits = np.sort(wED.real) - wED.real.min()
    check("open spin-1 chain: 4 near-degenerate levels below the bulk gap",
          splits[3] < 0.4 and splits[4] > 1.8 * splits[3],
          "splits " + ", ".join(f"{s:.3f}" for s in splits[:6]))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    xx = np.linspace(0, 1.0 / bulks[0] * 1.1, 20)
    axes[0].plot(1.0 / np.array(bulks), gaps, "o", ms=7)
    axes[0].plot(xx, g_inf + g_slope * xx, "--", color="k", label=f"extrap {g_inf:.3f}")
    axes[0].axhline(models.HALDANE_GAP, color="r", ls=":", label="0.4105 J")
    axes[0].set_xlabel("1/N$_{\\rm bulk}$"); axes[0].set_ylabel("singlet-triplet gap")
    axes[0].set_title("spin-1 Haldane gap"); axes[0].legend()
    axes[1].plot(range(N), prof, "o-")
    axes[1].set_xlabel("site"); axes[1].set_ylabel(r"$\langle S^z_i\rangle$")
    axes[1].set_title(f"edge spins (h={0.05})")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase3_haldane.png")); plt.close(fig)


def _string_order(psi, Sz, i, j):
    """<Sz_i (prod_{i<k<j} e^{i pi Sz_k}) Sz_j> by transfer-matrix contraction."""
    d = Sz.shape[0]
    expmat = np.diag(np.exp(1j * np.pi * np.diag(Sz)))
    psi.move_center_to(i)
    A = psi.M[i]
    left = np.tensordot(A.conj(), np.tensordot(Sz, A, axes=(1, 1)), axes=([0, 1], [1, 0]))
    for k in range(i + 1, j):
        Ak = psi.M[k]
        op = expmat if k != j else np.eye(d)
        tmp = np.tensordot(op, Ak, axes=(1, 1))         # (d_out, Dl, Dr)
        left = np.tensordot(left, tmp, axes=(1, 1))      # (Dbra, d_out, Dr)
        left = np.tensordot(Ak.conj(), left, axes=([0, 1], [0, 1]))
    Aj = psi.M[j]
    r = np.tensordot(Sz, Aj, axes=(1, 1))
    r = np.tensordot(Aj.conj(), r, axes=([1, 2], [0, 2]))
    return float(np.real(np.tensordot(left, r, axes=([0, 1], [0, 1]))))


# ======================================================================= P4
def phase4(quick=False):
    print("\n--- Phase 4: TFIM criticality and the central charge ---")
    N = 64 if not quick else 40
    res = dmrg.run_dmrg(models.tfim_mpo(N, 1.0), N, 2, chi_max=80,
                        n_sweeps=24, tol=1e-10, chi_schedule=[16, 32, 48, 64, 80],
                        verbose=False)
    psi = res.mps
    S = psi.entanglement_profile()
    L = np.arange(1, N)
    chord = np.log((N / np.pi) * np.sin(np.pi * L / N))
    lo, hi = N // 4, 3 * N // 4
    slope, inter = np.polyfit(chord[lo:hi], S[lo:hi], 1)
    c_est = 6.0 * slope
    check("block entropy slope gives central charge c = 1/2",
          abs(c_est - 0.5) < 0.07, f"c = {c_est:.4f}")

    # finite-entanglement scaling: correlation length grows with chi
    xis = []
    chis = [6, 12, 24, 48] if not quick else [6, 12, 24]
    for chi in chis:
        r = dmrg.run_dmrg(models.tfim_mpo(N, 1.0), N, 2, chi_max=chi,
                          n_sweeps=16, tol=1e-10, verbose=False)
        xis.append(_corr_length(r.mps, models.PAULI_X.astype(complex)))
    grows = all(xis[i + 1] >= xis[i] - 1e-6 for i in range(len(xis) - 1))
    check("correlation length grows with chi (finite-entanglement scaling)",
          grows, "  ".join(f"chi{c}:xi={x:.1f}" for c, x in zip(chis, xis)))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(chord, S, ".", ms=4)
    axes[0].plot(chord[lo:hi], inter + slope * chord[lo:hi], "-", color="k",
                 label=f"slope c/6, c={c_est:.3f}")
    axes[0].set_xlabel(r"$\ln[(N/\pi)\sin(\pi \ell/N)]$"); axes[0].set_ylabel(r"$S(\ell)$")
    axes[0].set_title(f"TFIM critical, N={N}"); axes[0].legend()
    axes[1].plot(chis, xis, "o-")
    axes[1].set_xlabel(r"$\chi$"); axes[1].set_ylabel(r"$\xi$ (sites)")
    axes[1].set_title("finite-entanglement scaling")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase4_criticality.png")); plt.close(fig)


def _corr_length(psi, op, rmax=None):
    N = psi.n
    c = N // 2
    rmax = rmax or min(20, N // 2 - 2)
    rs = np.arange(2, rmax)
    vals = np.array([abs(psi.correlator(op, c - r // 2, op, c + (r - r // 2)).real
                         - psi.expectation_1site(op, c - r // 2).real
                         * psi.expectation_1site(op, c + (r - r // 2)).real) for r in rs])
    good = vals > 1e-10
    if good.sum() < 3:
        return 0.0
    sl, _ = np.polyfit(rs[good], np.log(vals[good]), 1)
    return -1.0 / sl if sl < 0 else 0.0


# ======================================================================= P5
def phase5(quick=False):
    print("\n--- Phase 5: TEBD quench dynamics ---")
    N = 32 if not quick else 20
    g0, g1 = 0.5, 2.0
    # ground state at g0 by DMRG
    gs = dmrg.run_dmrg(models.tfim_mpo(N, g0), N, 2, chi_max=32, n_sweeps=14,
                       tol=1e-10, verbose=False).mps
    mpo1 = models.tfim_mpo(N, g1)
    bond1 = models.two_site_hamiltonian("tfim", N, g=g1)
    c = N // 2
    Z = models.PAULI_Z.astype(complex)

    tmax = 2.2 if not quick else 1.4
    dt = 0.03
    nsteps = int(tmax / dt)
    rmax = min(10, N // 2 - 2)
    psi = gs.copy()
    conn = {r: [] for r in range(1, rmax)}
    times, ent, energy = [], [], []
    gates_full = tebd.make_gates(bond1, dt)
    gates_half = tebd.make_gates(bond1, dt / 2)
    js = list(range(c + 1, c + rmax))
    for step in range(nsteps + 1):
        if step:
            tebd.sweep(psi, (gates_full, gates_half), chi_max=96, cutoff=1e-8)
        if step % 4 == 0:
            times.append(step * dt)
            ent.append(psi.entanglement_entropy(c - 1))
            energy.append(psi.expectation_mpo(mpo1).real)
            dens = psi.density_profile(Z)
            cij = psi.correlators_from(Z, c, Z, js)
            for r, j, cc in zip(range(1, rmax), js, cij):
                conn[r].append(cc.real - dens[c] * dens[j])
    times = np.array(times)

    # light-cone: dC(r,t) = |C(r,t) - C(r,0)|. For each snapshot the front is
    # the largest r whose signal exceeds a fixed fraction of the global max;
    # v is the slope of that front vs time (a robust wavefront tracker).
    dC = np.array([np.abs(np.array(conn[r]) - conn[r][0]) for r in range(1, rmax)])  # (r, t)
    thr = max(0.06 * dC.max(), 1e-4)
    rs_axis = np.arange(1, rmax)
    front_t, front_r = [], []
    for it_, tt in enumerate(times):
        hit = rs_axis[dC[:, it_] > thr]
        if len(hit):
            front_t.append(tt); front_r.append(hit.max())
    # keep the causal rising part (drop the saturated tail once the front hits rmax)
    front_t = np.array(front_t); front_r = np.array(front_r)
    rise = front_r < rmax - 1
    if rise.sum() >= 4:
        v_fit = np.polyfit(front_t[rise], front_r[rise], 1)[0]
    elif len(front_r) >= 4:
        v_fit = np.polyfit(front_t, front_r, 1)[0]
    else:
        v_fit = np.nan
    v_lr = models.lieb_robinson_velocity("tfim", g=g1)
    check("correlation front velocity ~ Lieb-Robinson 2 min(g,1)",
          np.isfinite(v_fit) and abs(v_fit - v_lr) < 0.7 * v_lr,
          f"v_fit = {v_fit:.2f}   v_LR = {v_lr:.2f}")

    # linear entanglement growth in the mid window
    ent = np.array(ent)
    k = len(times) // 2
    lin = np.polyfit(times[2:k], ent[2:k], 1)
    resid = np.std(ent[2:k] - np.polyval(lin, times[2:k]))
    check("entanglement entropy grows ~linearly after the quench",
          lin[0] > 0.2 and resid < 0.06, f"dS/dt = {lin[0]:.3f}, resid {resid:.3f}")

    drift = np.max(np.abs(np.array(energy) - energy[0]))
    check("energy conserved under the post-quench Hamiltonian", drift < 5e-3,
          f"max drift {drift:.1e}")

    # imaginary-time TEBD reproduces DMRG ground state
    _, e_imag = tebd.ground_state_imag("tfim", N, 2, chi_max=32,
                                       dt_schedule=(0.1, 0.03, 0.01), steps_per=120, g=g1, J=1.0)
    e_dmrg = dmrg.run_dmrg(mpo1, N, 2, chi_max=32, n_sweeps=16, tol=1e-10, verbose=False).energy
    check("imaginary-time TEBD = DMRG ground-state energy",
          abs(e_imag - e_dmrg) < 2e-4, f"TEBD {e_imag:.6f}  DMRG {e_dmrg:.6f}")

    # movie: the correlation light cone
    _lightcone_movie(times, conn, rmax, v_lr, quick)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    C = np.array([conn[r] for r in range(1, rmax)])
    im = axes[0].imshow(C, origin="lower", aspect="auto",
                        extent=[times[0], times[-1], 1, rmax - 1], cmap="magma")
    axes[0].plot(times, np.clip(v_lr * times, 0, rmax - 1), "c--", label="LR cone")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("r"); axes[0].set_title(r"$|C^{zz}_c(r,t)|$"); axes[0].legend()
    fig.colorbar(im, ax=axes[0])
    axes[1].plot(times, ent, "o-")
    axes[1].set_xlabel("t"); axes[1].set_ylabel(r"$S_{N/2}(t)$"); axes[1].set_title("entanglement growth")
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase5_quench.png")); plt.close(fig)


def _lightcone_movie(times, conn, rmax, v_lr, quick):
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        C = np.array([conn[r] for r in range(1, rmax)])
        fig, ax = plt.subplots(figsize=(5, 4))
        rs = np.arange(1, rmax)
        line, = ax.plot(rs, C[:, 0], "o-")
        cone = ax.axvline(0, color="r", ls="--")
        ax.set_ylim(0, max(C.max() * 1.1, 1e-3)); ax.set_xlabel("distance r"); ax.set_ylabel(r"$|C^{zz}(r,t)|$")
        ttl = ax.set_title("t = 0.00")

        def upd(f):
            line.set_ydata(C[:, f]); cone.set_xdata([v_lr * times[f]] * 2)
            ttl.set_text(f"t = {times[f]:.2f}")
            return line, cone, ttl
        anim = FuncAnimation(fig, upd, frames=len(times), blit=False)
        path = os.path.join(MEDIA, "phase5_lightcone.mp4")
        try:
            anim.save(path, writer=FFMpegWriter(fps=12))
        except Exception:
            anim.save(path.replace(".mp4", ".gif"), writer=PillowWriter(fps=12))
        plt.close(fig)
    except Exception as e:
        print(f"    (movie skipped: {e})")


# ======================================================================= P6
def phase6(quick=False):
    print("\n--- Phase 6: Hubbard Mott physics + Hydrogen-chain dissociation ---")
    # Mott gap via a chemical-potential scan at fixed U (particle-hole symmetric)
    N = 16 if not quick else 10
    U = 4.0
    mus = np.linspace(-0.2, 3.4, 13 if not quick else 9)
    fillings = []
    _, _, _, n_up, n_dn, _ = models._hubbard_operators()
    ntot_op = (n_up + n_dn).astype(complex)
    for mi, mu in enumerate(mus):
        res = dmrg.run_dmrg(models.hubbard_mpo(N, t=1.0, U=U, mu=mu), N, 4, chi_max=48,
                            n_sweeps=10, tol=1e-8, chi_schedule=[16, 32, 48], verbose=False)
        fillings.append(res.mps.density_profile(ntot_op).mean())
        print(f"    mu-scan {mi + 1}/{len(mus)}:  mu={mu:.2f}  <n>={fillings[-1]:.3f}")
    fillings = np.array(fillings)
    plateau = np.abs(fillings - 1.0) < 0.03
    if plateau.any():
        gap_dmrg = mus[plateau].max() - mus[plateau].min()
    else:
        gap_dmrg = 0.0
    gap_liebwu = _lieb_wu_gap(U)
    check("Hubbard Mott plateau width ~ Lieb-Wu charge gap",
          abs(gap_dmrg - gap_liebwu) < max(0.4, 0.4 * gap_liebwu),
          f"plateau {gap_dmrg:.3f}  vs Lieb-Wu {gap_liebwu:.3f}")

    # spin vs charge: static structure -- charge is gapped (short-ranged),
    # spin is gapless (power-law). Compare decay of connected correlators.
    N2 = 20 if not quick else 12
    res = dmrg.run_dmrg(models.hubbard_mpo(N2, t=1.0, U=U, mu=U / 2), N2, 4, chi_max=60,
                        n_sweeps=14, tol=1e-8, chi_schedule=[16, 32, 48, 60], verbose=False)
    psi = res.mps
    _, _, Sz1, _, _, _ = models.spin_operators(0.5)
    sz_op = np.zeros((4, 4), dtype=complex); sz_op[1, 1] = 0.5; sz_op[2, 2] = -0.5
    c = N2 // 2
    rs = np.arange(1, min(10, N2 // 2 - 1))
    js = [c + int(r) for r in rs]
    dens = psi.density_profile(ntot_op)
    spin_c = np.abs(np.real(psi.correlators_from(sz_op, c, sz_op, js)))
    chg_raw = np.real(psi.correlators_from(ntot_op, c, ntot_op, js))
    chg_c = np.abs(chg_raw - dens[c] * dens[js])
    # charge decays faster (larger effective slope on a log scale)
    sl_spin = np.polyfit(np.log(rs[spin_c > 1e-9]), np.log(spin_c[spin_c > 1e-9]), 1)[0]
    sl_chg = np.polyfit(rs[chg_c > 1e-10], np.log(chg_c[chg_c > 1e-10]), 1)[0]
    check("charge correlations decay faster than spin (Mott: charge gapped)",
          sl_chg < 0 and (chg_c[-1] < spin_c[-1]),
          f"spin log-slope {sl_spin:.2f}  charge exp-slope {sl_chg:.2f}")

    # Hydrogen chain dissociation in a minimal basis (single-band Hubbard proxy):
    # H_n at spacing R -> Hubbard with R-dependent t(R), U(R) from Slater 1s overlaps.
    Rs = np.linspace(1.0, 5.0, 10 if not quick else 6)
    n_H = 6
    e_dmrg, e_rhf = [], []
    for R in Rs:
        t_R, U_R, e0_R = _hchain_params(R)
        mpo = models.hubbard_mpo(n_H, t=t_R, U=U_R, mu=U_R / 2)
        res = dmrg.run_dmrg(mpo, n_H, 4, chi_max=32, n_sweeps=12, tol=1e-8, verbose=False)
        # subtract the -mu N shift, add the per-atom on-site reference
        E = res.energy + (U_R / 2) * n_H + n_H * e0_R
        e_dmrg.append(E / n_H)
        e_rhf.append((_rhf_hchain(t_R, U_R, n_H) + (U_R / 2) * n_H) / n_H + e0_R)
    e_dmrg = np.array(e_dmrg); e_rhf = np.array(e_rhf)
    # DMRG must approach a finite separated-atom limit; RHF overshoots (rises)
    dmrg_flat = abs(e_dmrg[-1] - e_dmrg[-2]) < abs(e_dmrg[1] - e_dmrg[0])
    rhf_worse = (e_rhf[-1] - e_dmrg[-1]) > (e_rhf[len(Rs) // 2] - e_dmrg[len(Rs) // 2])
    check("H-chain: DMRG dissociates to a flat separated-atom limit",
          dmrg_flat, f"dE_tail = {e_dmrg[-1] - e_dmrg[-2]:.4f}")
    check("H-chain: restricted HF overshoots at large R where DMRG does not",
          rhf_worse, f"gap(RHF-DMRG): mid {e_rhf[len(Rs)//2]-e_dmrg[len(Rs)//2]:.3f} -> end {e_rhf[-1]-e_dmrg[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(mus, fillings, "o-"); axes[0].axhline(1.0, ls=":", color="k")
    axes[0].set_xlabel(r"$\mu$"); axes[0].set_ylabel(r"$\langle n\rangle$")
    axes[0].set_title(f"Mott plateau (U={U}): gap {gap_dmrg:.2f}")
    axes[1].semilogy(rs, spin_c, "o-", label="spin")
    axes[1].semilogy(rs, chg_c, "s-", label="charge")
    axes[1].set_xlabel("r"); axes[1].set_ylabel("|C(r)|"); axes[1].set_title("spin-charge separation"); axes[1].legend()
    axes[2].plot(Rs, e_dmrg, "o-", label="DMRG")
    axes[2].plot(Rs, e_rhf, "s--", label="restricted HF")
    axes[2].set_xlabel("R (Bohr)"); axes[2].set_ylabel("E / atom (Ha)")
    axes[2].set_title(f"H$_{n_H}$ chain dissociation"); axes[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, "phase6_hubbard.png")); plt.close(fig)


def _lieb_wu_gap(U, t=1.0):
    """Lieb-Wu charge gap of the 1D Hubbard model at half filling (units of t).
    Delta = U - 4t + 8t integral_0^infty J_1(w)/(w (1 + exp(w U/2t))) dw."""
    from scipy.integrate import quad
    from scipy.special import j1
    with np.errstate(over="ignore"):
        val, _ = quad(lambda w: j1(w) / (w * (1.0 + np.exp(np.minimum(w * U / (2 * t), 700)))),
                      0, np.inf, limit=200)
    return U - 4 * t + 8 * t * val


def _hchain_params(R):
    """Crude minimal-basis parameters for a hydrogen chain at spacing R (Bohr):
    hopping t(R) from the 1s-1s hopping integral, on-site U(R) screened Coulomb,
    per-atom reference e0 (isolated H = -0.5 Ha)."""
    S = (1 + R + R ** 2 / 3) * np.exp(-R)          # 1s overlap
    t = 0.5 * (1 + R) * np.exp(-R) + 0.4 * S       # effective hopping, ~decays as e^-R
    U = 0.625 / np.sqrt(1 + (R / 2.5) ** 2)        # screened on-site repulsion -> 0.625 Ha at R->0-ish
    U = max(U, 0.35)
    return t, U, -0.5


def _rhf_hchain(t, U, n):
    """Restricted-HF energy of the half-filled single-band Hubbard chain:
    fill the lowest n/2 tight-binding orbitals (per spin) and add the mean-field
    U <n_up><n_dn> = U/4 per site."""
    eps = np.sort([-2 * t * np.cos(np.pi * m / (n + 1)) for m in range(1, n + 1)])
    E_band = 2 * np.sum(eps[: n // 2])
    return E_band + U * 0.25 * n - (U / 2) * n     # last term cancels the mu shift added by caller


# =======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    phases = {"1": phase1, "2": phase2, "3": phase3, "4": phase4, "5": phase5, "6": phase6}
    t0 = time.time()
    if args.phase == "all":
        for p in ("1", "2", "3", "4", "5", "6"):
            phases[p](args.quick)
    else:
        phases[args.phase](args.quick)
    ok = summary()
    print(f"\n  total wall time {time.time() - t0:.1f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
