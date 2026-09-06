"""Linear-response analysis of a delta-kick rt-TDDFT run: the dipole signal
d_a(t) -> dynamic polarizability alpha(omega) -> dipole strength function
S(omega) and photoabsorption cross section sigma(omega), with the
Thomas-Reiche-Kuhn f-sum rule as a built-in correctness check.

Conventions (atomic units, hbar = m = e = 1):

  kick        psi_j -> exp(i k x_a) psi_j
  moment      d_a(t) = integral x_a * n(r, t) d^3r        (geometric first moment)
  alpha_aa(w) = (1/k) integral_0^inf e^{i w t} e^{-t/tau} [d_a(t) - d_a(0)] dt
  S(w)        = (2 w / pi) Im alpha_aa(w)                  (per Cartesian axis)
  sigma(w)    = (4 pi w / c) Im alpha_aa(w)                (c = 137.036)

The kick imparts momentum +k to every electron, so d_a(t) - d_a(0) has
positive initial slope k*N_e; with the +(1/k) prefactor and the e^{+iwt}
transform this makes Im alpha > 0 (absorption) at resonances, and then
integral_0^inf S(w) dw = N_e  (TRK sum rule, per axis).
"""

import numpy as np

C_LIGHT = 137.035999084
HA_TO_EV = 27.211386245988


def moment_observer(occ, coords, dx, axis=0):
    """An observer fn(psi, t) -> d_axis(t) for propagate(observers=...)."""
    c = coords[axis]
    occ = np.asarray(occ)

    def obs(psi, t):
        n = np.einsum("j,j...->...", occ, np.abs(psi) ** 2)
        return float(np.sum(c * n) * dx ** 3)

    return obs


def polarizability(t, d, k, tau=None):
    """alpha(omega) from a uniformly-sampled dipole trace.

    t, d: 1-D arrays, t[0] = 0, uniform spacing.
    k:    kick strength.
    tau:  exponential damping time (default: 0.4 * (t[-1] - t[0])). Sets an
          artificial Lorentzian linewidth 2/tau on every peak.

    Returns (omega, alpha) for omega >= 0.
    """
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    dt = t[1] - t[0]
    T = t[-1] - t[0]
    if tau is None:
        tau = 0.4 * T

    signal = (d - d[0]) * np.exp(-(t - t[0]) / tau)

    # zero-pad to the next power of two for FFT resolution/speed
    n = 1
    while n < 4 * len(signal):
        n *= 2
    sig = np.zeros(n)
    sig[: len(signal)] = signal

    # integral_0^inf e^{+i w t} f(t) dt  ~  dt * sum_n f_n e^{+i w_k t_n}
    #                                    =  dt * N * ifft(f)_k
    G = np.fft.ifft(sig) * n * dt
    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt)

    pos = omega >= 0
    omega = omega[pos]
    alpha = (1.0 / k) * G[pos]
    return omega, alpha


def strength_function(omega, alpha):
    """S(omega) = (2 omega / pi) Im alpha(omega)  (dipole oscillator-strength
    density, per Cartesian axis; integrates to N_e)."""
    return (2.0 * omega / np.pi) * np.imag(alpha)


def cross_section(omega, alpha):
    """sigma(omega) = (4 pi omega / c) Im alpha(omega)  (photoabsorption cross
    section, Gaussian atomic units; in Bohr^2)."""
    return (4.0 * np.pi * omega / C_LIGHT) * np.imag(alpha)


def sum_rule(omega, S, w_max=None):
    """integral_0^{w_max} S(omega) domega -- should equal N_e (TRK). Returns
    the scalar integral; also useful as the running N_eff(omega) via
    np.cumsum(S) * domega."""
    m = np.ones_like(omega, bool) if w_max is None else (omega <= w_max)
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))  # np>=2 renamed it
    return float(trapezoid(S[m], omega[m]))


def kick_spectrum(psi0, grid, occ, V_nuc, dt, T, k=0.01, axis=0, method="lda",
                  w_max=4.0):
    """Full pipeline for one delta-kick: kick psi0 along `axis`, propagate for
    T (a.u.) at step dt, FFT the induced moment. Returns (omega, alpha, S)
    truncated to omega < w_max.
    """
    import perturb
    import propagate as prop

    coords = grid[1:4]
    dx = grid[4]
    psi = perturb.dipole_kick(psi0, coords, k, axis=axis)
    res = prop.propagate(psi, grid, int(round(T / dt)), dt, occ=occ, V_nuc=V_nuc,
                         method=method, record_every=1,
                         observers={"d": moment_observer(occ, coords, dx, axis=axis)})
    w, alpha = polarizability(res["t"], res["obs"]["d"], k)
    S = strength_function(w, alpha)
    m = w < w_max
    return w[m], alpha[m], S[m]


def hhg_spectrum(t, d, window="hann"):
    """High-harmonic spectrum from a dipole trace d(t) driven by a laser.

    Emits the dipole-acceleration spectrum |a(w)|^2 with a(t) = d''(t)
    (the acceleration form suppresses the low-frequency drive and is the
    standard HHG observable). d(t) is differenced twice, windowed, zero-
    padded, and FFT'd.

    window: "hann" | "none" | a callable t -> array.
    Returns (omega, power), omega >= 0.
    """
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    dt = t[1] - t[0]
    a = np.gradient(np.gradient(d, dt), dt)
    if window == "hann":
        w = np.hanning(len(a))
    elif window in ("none", None):
        w = np.ones(len(a))
    elif callable(window):
        w = np.asarray(window(t), float)
    else:
        raise ValueError(f"unknown window {window!r}")
    aw = a * w

    n = 1
    while n < 4 * len(aw):
        n *= 2
    A = np.fft.rfft(aw, n=n)
    omega = 2 * np.pi * np.fft.rfftfreq(n, d=dt)
    return omega, np.abs(A) ** 2


def harmonic_peak(omega, power, harmonic_order, omega_L, half_width=3):
    """Max of `power` in a small window around omega = harmonic_order * omega_L."""
    i = int(np.argmin(np.abs(omega - harmonic_order * omega_L)))
    lo, hi = max(0, i - half_width), i + half_width + 1
    return float(np.max(power[lo:hi]))


def peaks(omega, S, n=5, w_lo=0.05, w_hi=None, min_prominence=None):
    """The n most prominent local maxima of S(omega) in (w_lo, w_hi),
    as a list of (omega, S) sorted by descending S. Pure-numpy, no scipy."""
    if w_hi is None:
        w_hi = omega[-1]
    band = (omega > w_lo) & (omega < w_hi)
    w, s = omega[band], S[band]
    loc = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    idx = np.where(loc)[0] + 1
    if min_prominence is not None:
        idx = idx[s[idx] > min_prominence]
    idx = idx[np.argsort(s[idx])[::-1][:n]]
    return [(float(w[i]), float(s[i])) for i in idx]
