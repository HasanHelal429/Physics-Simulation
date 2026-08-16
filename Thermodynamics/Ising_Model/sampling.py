"""Autocorrelation-aware measurement infrastructure for Ising Monte Carlo
time series: burn-in detection from independent-chain agreement, integrated
autocorrelation time via FFT + automatic windowing (Sokal's method), and
autocorrelation-corrected error bars. Pure numpy -- operates on time series
already pulled off the GPU.
"""
import numpy as np


def estimate_burnin(traces, tol_sigma=2.0, tail_frac=0.5):
    """traces: (n_chains, n_steps), independent chains of the same scalar
    observable started from independent initial conditions.

    Returns the smallest step index t0 such that the mean of traces[:, t0:]
    agrees with the mean of the run's second half to within tol_sigma
    chain-to-chain standard errors -- the point past which the running
    estimate is statistically indistinguishable from its long-run
    asymptote, rather than an arbitrary fixed fraction of the run.
    """
    n_chains, n_steps = traces.shape
    ref_mean = traces[:, int(n_steps * tail_frac):].mean()
    for t0 in range(n_steps - 1):
        chain_means = traces[:, t0:].mean(axis=1)
        sem = chain_means.std(ddof=1) / np.sqrt(n_chains)
        if sem == 0 or abs(chain_means.mean() - ref_mean) < tol_sigma * sem:
            return t0
    return n_steps // 2


def autocorrelation_function(x):
    """Normalized autocorrelation function rho(t), rho(0)=1, via FFT
    (zero-padded to avoid circular-correlation wraparound)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.fft(x, n=nfft)
    acov = np.fft.ifft(f * np.conjugate(f))[:n].real
    return acov / acov[0]


def integrated_autocorrelation_time(x, c=5.0):
    """Integrated autocorrelation time tau_int via Sokal's automatic
    windowing: tau_int(M) = 0.5 + sum_{t=1}^M rho(t), take the smallest
    window M with M >= c*tau_int(M) (standard emcee-style estimator)."""
    rho = autocorrelation_function(x)
    taus = 2.0 * np.cumsum(rho) - 1.0
    valid = np.arange(len(taus)) < c * taus
    window = np.argmin(valid) if np.any(valid) else len(taus) - 1
    return max(taus[window], 1.0)


def mean_with_error(x, c=5.0):
    """Sample mean and autocorrelation-corrected standard error of a single
    time series, using effective sample size N/(2*tau_int)."""
    x = np.asarray(x, dtype=float)
    tau = integrated_autocorrelation_time(x, c=c)
    n_eff = len(x) / (2.0 * tau)
    sem = x.std(ddof=1) / np.sqrt(max(n_eff, 1.0))
    return x.mean(), sem, tau
