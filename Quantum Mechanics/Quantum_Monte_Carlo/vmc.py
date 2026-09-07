"""
Variational Monte Carlo: sample ``|Psi_T|^2`` with a drift-guided
(importance-sampled Langevin) Metropolis walk, estimate ``<E_L>`` with a
reblocked error bar, and optimize the Jastrow parameters by correlated-sample
variance minimization.

Walkers are propagated as a batch of shape ``(W, n_elec, 3)``.
"""

import numpy as np


# ------------------------------------------------------------------ sampler
def metropolis_step(psi, R, tau, rng):
    """One drift-Metropolis move of *all* electrons per walker.
    Proposal:  R' = R + tau * v(R) + sqrt(2 tau) eta   (D = 1/2, so
    D*tau = tau/2 drift and variance 2 D tau = tau -- written with the
    standard QMC convention where ``tau`` already absorbs D).
    Returns ``(R_new, accept_fraction)``.
    """
    W = R.shape[0]
    info = psi.evaluate(R)
    v = info["grad_ln_psi"]
    v = _clip_drift(v, tau)
    eta = rng.standard_normal(R.shape)
    R_prop = R + tau * v + np.sqrt(2.0 * tau) * eta

    info_p = psi.evaluate(R_prop)
    vp = _clip_drift(info_p["grad_ln_psi"], tau)

    # Green's-function ratio  G(R<-R') / G(R->R')
    fwd = np.sum((R_prop - R - tau * v) ** 2, axis=(1, 2))
    bwd = np.sum((R - R_prop - tau * vp) ** 2, axis=(1, 2))
    log_g = (fwd - bwd) / (4.0 * tau)
    log_ratio = 2.0 * (info_p["logpsi"] - info["logpsi"]) + log_g

    accept = np.log(rng.random(W)) < log_ratio
    R_new = np.where(accept[:, None, None], R_prop, R)
    return R_new, float(np.mean(accept))


def _clip_drift(v, tau):
    """Cap the drift so a single step can't overshoot near a node
    (Umrigar-style): |v| <- v * (-1 + sqrt(1 + 2 a |v|^2 tau)) / (a |v|^2 tau)."""
    a = 1.0
    v2 = np.sum(v ** 2, axis=-1, keepdims=True)
    v2 = np.where(v2 > 1e-12, v2, 1e-12)
    scale = (-1.0 + np.sqrt(1.0 + 2.0 * a * v2 * tau)) / (a * v2 * tau)
    return v * scale


def equilibrate(psi, R, rng, n_steps=200, tau=0.05, target=0.55):
    """Burn-in with automatic ``tau`` adaptation toward `target` acceptance."""
    acc = 0.0
    for _ in range(n_steps):
        R, acc = metropolis_step(psi, R, tau, rng)
        if acc < target - 0.1:
            tau *= 0.9
        elif acc > target + 0.1:
            tau *= 1.1
        tau = float(np.clip(tau, 1e-4, 1.0))
    return R, tau


def sample_energy(psi, R, rng, n_steps=2000, tau=0.05, decorr=1, return_series=False):
    """Accumulate the local energy over ``n_steps`` walker moves.
    Returns ``(mean, stderr, extra)`` with a reblocked stderr; ``extra`` has
    the per-step walker-mean series and the acceptance."""
    series = []
    accs = []
    for step in range(n_steps):
        R, acc = metropolis_step(psi, R, tau, rng)
        accs.append(acc)
        if step % decorr == 0:
            EL = psi.local_energy(R)
            series.append(EL.copy())
    E = np.array(series)                       # (n_samp, W)
    mean = float(E.mean())
    stderr = reblock_error(E.mean(axis=1))     # error of the walker-averaged series
    extra = {"series": E, "acceptance": float(np.mean(accs)), "R": R,
             "variance": float(E.var())}
    if return_series:
        extra["walker_mean_series"] = E.mean(axis=1)
    return mean, stderr, extra


# ------------------------------------------------------------------ reblocking
def reblock_error(x):
    """Flyvbjerg-Petersen blocking: repeatedly halve the series by pairwise
    averaging and take the plateau of the naive standard error."""
    x = np.asarray(x, float)
    x = x[: len(x) - (len(x) % 2)] if len(x) % 2 else x
    errs = []
    while len(x) >= 4:
        err = np.std(x, ddof=1) / np.sqrt(len(x))
        errs.append(err)
        if len(x) % 2:
            x = x[:-1]
        x = 0.5 * (x[0::2] + x[1::2])
    return float(max(errs)) if errs else float("nan")


def autocorr_time(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return 1.0
    tau = 1.0
    for k in range(1, min(n // 2, 500)):
        c = np.dot(x[:-k], x[k:]) / (n - k) / var
        if c <= 0:
            break
        tau += 2.0 * c
    return tau


# ------------------------------------------------------------------ optimization
def optimize_jastrow(psi, system, rng, n_walkers=400, n_opt=6,
                     sample_steps=800, tau=0.06, verbose=True,
                     bounds=((0.05, 8.0), (0.05, 8.0), (-1.0, 2.0))):
    """Correlated-sample variance minimization of the Pade parameters
    ``[b_ee, b_en, c_en]``. On each outer step: draw a fixed sample from the
    *current* ``|Psi_T|^2``, then minimize the reweighted variance of
    ``E_L(p)`` over that frozen sample (Nelder-Mead), accept if the true
    (re-sampled) energy did not rise.
    """
    from scipy.optimize import minimize

    R = system.initial_walkers(n_walkers, seed=rng.integers(1 << 30))
    R, tau = equilibrate(psi, R, rng, n_steps=300, tau=tau)

    best_p = psi.jas.params.copy()
    best_E, best_err, extra = sample_energy(psi, R, rng, n_steps=sample_steps, tau=tau)
    R = extra["R"]
    history = [(best_p.copy(), best_E, best_err)]
    if verbose:
        print(f"  opt 0: p={_fmt(best_p)}  E={best_E:.5f} +/- {best_err:.5f}  "
              f"var={extra['variance']:.4f}  acc={extra['acceptance']:.2f}")

    for it in range(1, n_opt + 1):
        # frozen sample from current Psi_T^2
        R, tau = equilibrate(psi, R, rng, n_steps=120, tau=tau)
        config_bank = []
        for _ in range(24):
            R, _ = metropolis_step(psi, R, tau, rng)
            config_bank.append(R.copy())
        bank = np.concatenate(config_bank, axis=0)          # (24W, n_elec, 3)
        logpsi0 = psi.log_psi(bank)

        def objective(p):
            saved = psi.jas.params.copy()
            psi.jas.params = np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds])
            info = psi.evaluate(bank)
            EL = -0.5 * info["lap_over_psi"] + psi.potential(bank)
            w = np.exp(2.0 * (info["logpsi"] - logpsi0))
            w = w / np.mean(w)
            Emean = np.sum(w * EL) / np.sum(w)
            var = np.sum(w * (EL - Emean) ** 2) / np.sum(w)
            psi.jas.params = saved
            return var + 0.15 * max(Emean - best_E, 0.0)     # small energy tilt

        res = minimize(objective, psi.jas.params, method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-5})
        trial_p = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])

        psi.jas.params = trial_p
        R, tau = equilibrate(psi, R, rng, n_steps=120, tau=tau)
        E, err, extra = sample_energy(psi, R, rng, n_steps=sample_steps, tau=tau)
        R = extra["R"]
        if verbose:
            print(f"  opt {it}: p={_fmt(trial_p)}  E={E:.5f} +/- {err:.5f}  "
                  f"var={extra['variance']:.4f}")
        if E <= best_E + 2 * (err + best_err):
            best_p, best_E, best_err = trial_p.copy(), E, err
        else:
            psi.jas.params = best_p
        history.append((psi.jas.params.copy(), E, err))

    psi.jas.params = best_p
    return {"params": best_p, "energy": best_E, "error": best_err, "history": history}


def _fmt(p):
    return "[" + ", ".join(f"{x:.3f}" for x in p) + "]"
