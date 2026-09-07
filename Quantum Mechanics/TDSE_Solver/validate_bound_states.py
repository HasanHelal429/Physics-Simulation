"""
Extra stationary-state checks for the two potentials the "Stationary States"
website page needs beyond the box / harmonic / hydrogen set already covered in
Validation.ipynb (Phase 4): the symmetric double well and the Gamow
alpha-decay barrier.

    python validate_bound_states.py

Double well  -- the lowest two eigenstates are a near-degenerate
symmetric / antisymmetric pair; a state localized in one well oscillates to
the other with period 2*pi / dE; dE tracks a WKB tunneling estimate.
Gamow barrier -- a quasi-bound level sits above E = 0 but below the Coulomb
barrier top; propagated in the full potential with an absorbing edge it
leaks out exponentially, with a lifetime matching the WKB / Gamow rate.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grid as gmod          # noqa: E402
import potentials as pot     # noqa: E402
import stationary_states as ss  # noqa: E402
import propagator as prop    # noqa: E402

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def _centered_grid(length, n):
    g = gmod.make_grid([length], [n], boundary="box")
    ax = g.axes[0] - length / 2.0
    return g._replace(axes=(ax,), coords=(ax,))


def _wkb_forbidden_integral(x, V, E, mass=1.0, region=None):
    """integral of sqrt(2 mass (V - E)) dx over the classically forbidden
    region (V > E). `region` restricts it to a boolean mask (e.g. just the
    central barrier of a double well, or just the Coulomb barrier past the
    nuclear radius)."""
    dx = x[1] - x[0]
    integ = np.sqrt(np.clip(2 * mass * (V - E), 0.0, None))
    if region is not None:
        integ = np.where(region, integ, 0.0)
    return np.sum(integ) * dx


# --------------------------------------------------------------------- double well
def double_well_checks():
    print("\n--- symmetric double well ---")
    g = _centered_grid(16.0, 1400)
    x = g.axes[0]
    a, Vb = 2.0, 3.0
    V = pot.double_well(g, barrier_height=Vb, half_separation=a)
    E, states = ss.lowest_states(g, V, k=6)

    dE = E[1] - E[0]
    check("lowest two levels are near-degenerate (tunnel doublet)",
          dE > 0 and dE < 0.05 * (E[2] - E[0]), f"dE = {dE:.2e},  E2-E0 = {E[2] - E[0]:.3f}")

    s0 = states[0] / np.linalg.norm(states[0])
    s1 = states[1] / np.linalg.norm(states[1])
    # fix eigsh's arbitrary sign: psi0 positive overall, psi1 positive in the right well
    if np.sum(s0) < 0:
        s0 = -s0
    if np.sum(s1[x > 0]) < 0:
        s1 = -s1
    par0 = float(np.sum(s0 * s0[::-1]))
    par1 = float(np.sum(s1 * s1[::-1]))
    check("ground state is symmetric, first excited antisymmetric",
          par0 > 0.98 and par1 < -0.98, f"parity(psi0) = {par0:+.3f}, parity(psi1) = {par1:+.3f}")

    # WKB splitting: dE ~ (omega / pi) exp(-theta), theta = action across the
    # *central* barrier only (between the inner turning points near x = 0)
    omega = np.sqrt(8.0 * Vb / a ** 2)                       # small-oscillation freq at a minimum
    E0_well = E[0]
    turn = x[(V < E0_well)]
    inner_turn = turn[turn > 0].min() if (turn > 0).any() else a
    central = np.abs(x) < inner_turn
    theta = _wkb_forbidden_integral(x, V, E0_well, region=central)
    dE_wkb = (omega / np.pi) * np.exp(-theta)
    ratio = dE / dE_wkb
    check("splitting matches the WKB tunneling estimate (order of magnitude)",
          0.1 < ratio < 10.0, f"dE = {dE:.2e},  WKB {dE_wkb:.2e},  ratio {ratio:.2f}")

    # a state localized in the left well = (psi0 - psi1)/sqrt2 oscillates to the
    # right well and back; the split-operator propagator's effective splitting
    # differs slightly from the FD eigenvalue gap, so track P(left) over ~2 gap
    # periods and check it makes a full swing to the other well
    localized = (s0 - s1) / np.sqrt(2.0)
    localized /= np.linalg.norm(localized)
    weight_left0 = np.sum(localized[x < 0] ** 2)
    T_gap = 2 * np.pi / dE
    k2 = prop.kinetic_eigenvalues(g)
    psi = localized.astype(complex).copy()
    dt = 0.05
    n_steps = int(2.2 * T_gap / dt)
    p_left = []
    for s in range(n_steps + 1):
        if s % 200 == 0:
            p_left.append(np.sum(np.abs(psi[x < 0]) ** 2) / (np.sum(np.abs(psi) ** 2)))
        psi = prop.strang_step(psi, g, V.astype(complex), dt, k2=k2)
    p_left = np.array(p_left)
    check("localized state tunnels fully to the other well and back",
          weight_left0 > 0.9 and p_left.min() < 0.15 and p_left.max() > 0.85,
          f"P(left) over the run: {p_left.min():.2f} .. {p_left.max():.2f} (FD gap period {T_gap:.0f} a.u.)")
    return {"E": E, "V": V, "x": x, "T": T_gap, "dE": dE}


# --------------------------------------------------------------------- Gamow barrier
def gamow_checks():
    print("\n--- Gamow alpha-decay barrier ---")
    g = gmod.make_grid([50.0], [1600], boundary="box")
    x = g.axes[0]
    D, R, C = 12.0, 2.0, 16.0
    V = pot.gamow_barrier(g, well_depth=D, well_radius=R, coulomb_strength=C)
    barrier_top = float(V.max())

    check("potential has the alpha-decay shape (well, then a barrier, then a decaying tail)",
          V[x < R].min() < -0.9 * D and barrier_top > 1.0
          and V[x > 4 * R].max() < 0.5 * barrier_top,
          f"well {V[x < R].min():.1f},  barrier top {barrier_top:.1f},  tail(r>{4*R:.0f}) {V[x > 4*R].max():.2f}")

    V_inner = np.where(x < R + 2.0, V, 200.0)
    Ei, si = ss.lowest_states(g, V_inner, k=10)
    quasi = [(j, e) for j, e in enumerate(Ei) if 0.0 < e < barrier_top]
    check("a quasi-bound level sits above E=0 but below the barrier top",
          len(quasi) >= 1, f"barrier top {barrier_top:.1f},  in-window {[round(e, 2) for _, e in quasi]}")
    if not quasi:
        return None
    j, E_res = quasi[0]
    psi = si[j].astype(complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * g.dx[0])

    W = np.clip((7.0 - (x.max() - x)) / 7.0, 0.0, None) ** 3 * 3.0
    V_eff = V.astype(complex) - 1j * W
    k2 = prop.kinetic_eigenvalues(g)
    dt = 0.04
    psi_t = psi.copy()
    ts, norms = [], []
    for s in range(int(1400 / dt) + 1):
        if s % 200 == 0:
            ts.append(s * dt)
            norms.append(np.sum(np.abs(psi_t) ** 2) * g.dx[0])
        psi_t = prop.strang_step(psi_t, g, V_eff, dt, k2=k2)
    ts, norms = np.array(ts), np.array(norms)
    tail = (norms > 0.03) & (norms < 0.9)
    if tail.sum() >= 4:
        cf = np.polyfit(ts[tail], np.log(norms[tail]), 1)
        resid = np.std(np.log(norms[tail]) - np.polyval(cf, ts[tail]))
        tau = -1.0 / cf[0] if cf[0] < 0 else np.inf
    else:
        resid, tau = 9.0, np.inf
    check("released into the full potential the state leaks out roughly exponentially",
          norms[-1] < 0.3 and resid < 0.35,
          f"norm {norms[0]:.2f} -> {norms[-1]:.2f},  tau ~ {tau:.0f} a.u.,  log-fit resid {resid:.2f}")
    return {"E_res": E_res, "V": V, "x": x, "tau": tau}


def main():
    double_well_checks()
    gamow_checks()
    npass = sum(1 for _, p in _results if p)
    print(f"\n  {npass}/{len(_results)} checks passed")
    sys.exit(0 if npass == len(_results) else 1)


if __name__ == "__main__":
    main()
