"""
Diffusion Monte Carlo: project out the fixed-node ground state by evolving the
imaginary-time Schrodinger equation as an importance-sampled drift-diffusion-
branching random walk on a walker population.

Follows Umrigar-Nightingale-Runge:
* drift + diffuse each walker, with a Metropolis accept/reject on the move
  (kills most of the time-step error),
* fixed-node: any move that changes the sign of Psi_T is rejected,
* multiply the walker weight by the branching factor
  ``exp(-dtau (S(R)+S(R'))/2)`` with ``S = E_L - E_T``,
* stochastic-reconfiguration (comb) population control back to the target,
* update ``E_T`` with a gentle feedback on ``ln(W/W_target)``.

The energy estimator is the weight-averaged local energy (the mixed estimator,
exact for the energy within fixed node). Time-step error is removed by running
a few ``dtau`` and extrapolating to 0.
"""

import numpy as np

from vmc import _clip_drift, metropolis_step, equilibrate, reblock_error


def _sign_logpsi(psi, R):
    """(sign of Psi_T, ln|Psi_T|) for each walker -- sign from the Slater
    determinants (Jastrow is strictly positive)."""
    R = np.asarray(R, float)
    W = R.shape[0]
    sign = np.ones(W)
    logabs = np.zeros(W)
    for orbs, sl in ((psi.orbs_up, slice(0, psi.n_up)),
                     (psi.orbs_dn, slice(psi.n_up, psi.n_elec))):
        if len(orbs) == 0:
            continue
        M, _, _ = psi._slater_block(R[:, sl], orbs)
        s, ld = np.linalg.slogdet(M)
        sign *= s
        logabs += ld
    info = psi.evaluate(R)
    logabs = logabs  # determinant part only for sign; full logpsi in info
    return sign, info


def dmc_run(psi, system, rng, dtau=0.01, n_blocks=40, steps_per_block=40,
            n_walkers=400, n_equil_blocks=8, E_ref=None, feedback=0.1,
            verbose=True):
    """Returns a dict with the extrapolated (per-dtau) energy, its error, the
    per-block energy series, and the final walker population.
    """
    R = system.initial_walkers(n_walkers, seed=rng.integers(1 << 30))
    R, tau_vmc = equilibrate(psi, R, rng, n_steps=250, tau=max(dtau, 0.03))
    # VMC-distribute a bit more so we start near |Psi_T|^2
    for _ in range(200):
        R, _ = metropolis_step(psi, R, tau_vmc, rng)

    sign, info = _sign_logpsi(psi, R)
    EL = -0.5 * info["lap_over_psi"] + psi.potential(R)
    weight = np.ones(R.shape[0])
    E_T = float(np.mean(EL)) if E_ref is None else float(E_ref)

    block_energies = []
    W_target = float(n_walkers)
    all_local = []

    total_blocks = n_equil_blocks + n_blocks
    for b in range(total_blocks):
        block_num, block_den = 0.0, 0.0
        for _ in range(steps_per_block):
            R, weight, EL, sign, acc = _dmc_step(psi, R, weight, EL, sign, E_T, dtau, rng)
            wsum = weight.sum()
            block_num += np.sum(weight * EL)
            block_den += wsum
            # population control: comb when weight mass strays
            if wsum > 1.6 * W_target or wsum < 0.6 * W_target or (R.shape[0] > 2.5 * n_walkers):
                R, weight, EL, sign = _reconfigure(R, weight, EL, sign, n_walkers, rng)
            E_T = E_T - feedback / dtau * np.log(max(weight.sum(), 1e-6) / W_target) * dtau
        E_block = block_num / block_den
        if b >= n_equil_blocks:
            block_energies.append(E_block)
            all_local.append(E_block)
        # trailing-average trial energy keeps things stable
        E_T = 0.5 * E_T + 0.5 * E_block
        if verbose and (b % 5 == 0 or b == total_blocks - 1):
            tag = "eq" if b < n_equil_blocks else "  "
            print(f"  {tag} block {b:3d}  E={E_block:.5f}  E_T={E_T:.5f}  "
                  f"Nw={R.shape[0]}  Wsum={weight.sum():.0f}")

    be = np.array(block_energies)
    return {
        "energy": float(be.mean()),
        "error": reblock_error(be),
        "block_energies": be,
        "dtau": dtau,
        "R": R,
        "weight": weight,
    }


def _dmc_step(psi, R, weight, EL_old, sign_old, E_T, dtau, rng):
    W = R.shape[0]
    info = psi.evaluate(R)
    v = _clip_drift(info["grad_ln_psi"], dtau)
    eta = rng.standard_normal(R.shape)
    R_new = R + dtau * v + np.sqrt(dtau) * eta

    info_n = psi.evaluate(R_new)
    vn = _clip_drift(info_n["grad_ln_psi"], dtau)

    # Metropolis accept/reject on the drift-diffusion move
    fwd = np.sum((R_new - R - dtau * v) ** 2, axis=(1, 2))
    bwd = np.sum((R - R_new - dtau * vn) ** 2, axis=(1, 2))
    log_g = (fwd - bwd) / (2.0 * dtau)
    log_ratio = 2.0 * (info_n["logpsi"] - info["logpsi"]) + log_g

    sign_new, info_sn = _sign_logpsi(psi, R_new)
    node_cross = sign_new * sign_old < 0
    accept = (np.log(rng.random(W)) < log_ratio) & (~node_cross)

    R_out = np.where(accept[:, None, None], R_new, R)
    EL_new = -0.5 * info_n["lap_over_psi"] + psi.potential(R_new)
    EL_out = np.where(accept, EL_new, EL_old)
    sign_out = np.where(accept, sign_new, sign_old)

    # branching factor with the effective time step (UNR): scale by accept prob
    p_acc = np.clip(np.exp(np.minimum(log_ratio, 0.0)), 0.0, 1.0)
    tau_eff = dtau * np.where(accept, 1.0, p_acc) / np.maximum(p_acc, 1e-6)
    tau_eff = np.where(np.isfinite(tau_eff), tau_eff, dtau)
    tau_eff = np.clip(tau_eff, 0.0, 4.0 * dtau)
    S_old = E_T - EL_old
    S_new = E_T - EL_out
    weight_out = weight * np.exp(0.5 * (S_old + S_new) * tau_eff)
    weight_out = np.clip(weight_out, 1e-8, 20.0)
    return R_out, weight_out, EL_out, sign_out, float(np.mean(accept))


def _reconfigure(R, weight, EL, sign, n_target, rng):
    """Stochastic reconfiguration (Sorella comb): resample ``n_target`` walkers
    with probability proportional to weight, then reset weights to the mean."""
    w = np.asarray(weight, float)
    w = np.clip(w, 1e-12, None)
    p = w / w.sum()
    positions = (rng.random() + np.arange(n_target)) / n_target
    cdf = np.cumsum(p)
    idx = np.searchsorted(cdf, positions)
    idx = np.clip(idx, 0, len(w) - 1)
    mean_w = w.sum() / n_target
    return R[idx].copy(), np.full(n_target, mean_w), EL[idx].copy(), sign[idx].copy()


def dmc_timestep_extrapolation(psi, system, rng, dtaus=(0.02, 0.01, 0.005),
                               verbose=True, **kw):
    """Run DMC at each ``dtau`` and linearly extrapolate the energy to
    ``dtau -> 0``. Returns ``(E0, err0, per_tau)``."""
    pts = []
    for dt in dtaus:
        if verbose:
            print(f"--- DMC dtau = {dt} ---")
        out = dmc_run(psi, system, rng, dtau=dt, verbose=verbose, **kw)
        pts.append((dt, out["energy"], out["error"]))
        if verbose:
            print(f"    dtau={dt}: E = {out['energy']:.5f} +/- {out['error']:.5f}")
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    e = np.array([p[2] for p in pts])
    A = np.vstack([np.ones_like(x), x]).T
    w = 1.0 / np.maximum(e, 1e-6) ** 2
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    coef = cov @ (A.T @ (w * y))
    return float(coef[0]), float(np.sqrt(cov[0, 0])), pts
