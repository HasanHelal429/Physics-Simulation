"""Exact reference solutions for the 2D Ising model (J=1), used as
validation ground truth -- exact enumeration for small finite lattices,
Onsager's closed-form results for the thermodynamic limit.
"""
import numpy as np


def onsager_Tc():
    """Exact critical temperature, B=0, J=1, k=1."""
    return 2.0 / np.log(1.0 + np.sqrt(2.0))


def onsager_magnetization(T):
    """Exact spontaneous magnetization per site below T_c (B=0, J=1), 0 above."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    beta = 1.0 / T
    s = np.sinh(2.0 * beta)
    with np.errstate(invalid='ignore'):
        m = np.where(s > 1.0, (1.0 - s ** -4.0) ** 0.125, 0.0)
    return m if m.shape != (1,) else m[0]


def exact_enumeration(L, T, B=0.0):
    """Brute-force exact statistical mechanics of an LxL periodic Ising
    lattice by summing over all 2**(L*L) configurations. Only tractable for
    small L (L=4 -> 65536 states is fast; L=5 -> ~33M states is slow but
    feasible; L>=6 is not).

    Returns a dict with per-site <E>, <M>, <|M|>, specific heat C, and
    susceptibility chi (both per-site, i.e. divided by L*L).
    """
    n = L * L
    if n > 25:
        raise ValueError(f"L={L} too large for exact enumeration (n={n} > 25)")

    idx = np.arange(1 << n, dtype=np.uint64)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
    spins = (2 * bits - 1).reshape(-1, L, L).astype(np.float64)

    nn = (np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
          + np.roll(spins, 1, axis=2) + np.roll(spins, -1, axis=2))
    interaction = -0.5 * np.sum(spins * nn, axis=(1, 2))
    field = -B * np.sum(spins, axis=(1, 2))
    E = interaction + field
    M = np.sum(spins, axis=(1, 2))

    beta = 1.0 / T
    w = np.exp(-beta * (E - E.min()))
    Z = w.sum()
    p = w / Z

    E_mean = np.sum(p * E)
    E2_mean = np.sum(p * E ** 2)
    M_mean = np.sum(p * M)
    M2_mean = np.sum(p * M ** 2)
    absM_mean = np.sum(p * np.abs(M))

    C = (E2_mean - E_mean ** 2) / T ** 2 / n
    chi = (M2_mean - M_mean ** 2) / T / n

    return dict(E=E_mean / n, M=M_mean / n, absM=absM_mean / n, C=C, chi=chi)
