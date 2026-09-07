"""
System definitions for the QMC solvers: nuclei, electron spin counts, and the
single-particle orbital set that builds the Slater determinants. Also the
homogeneous-electron-gas periodic box.

An `orbital` is a callable ``f(dr) -> (val, grad, lap)`` where ``dr`` is the
electron position *relative to this orbital's centre*, shapes ``(N,3) ->
(N,), (N,3), (N,)``. Atomic orbitals come straight from
`HF_solver.hydrogenic.orbital_value_grad_lap`; molecular orbitals are fixed
linear combinations of them.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "HF_solver"))

import hydrogenic as hy  # noqa: E402


# --------------------------------------------------------------------------
def atomic_orbital(n, l, m, zeta, center):
    """Hydrogen-like orbital with effective exponent `zeta`, centred at
    `center` (length-3). Uses the analytic value/grad/lap. The exponent is
    passed as ``Z`` so the electron-nucleus cusp is ``-zeta``."""
    center = np.asarray(center, float)

    def f(pos):
        dr = np.asarray(pos, float) - center
        return hy.orbital_value_grad_lap(n, l, m, dr, Z=zeta)

    return f


def mo_lcao(components):
    """Molecular orbital = sum_k coeff_k * atomic_orbital_k. `components` is a
    list of ``(coeff, orbital_callable)``."""
    def f(pos):
        val = None
        grad = None
        lap = None
        for c, orb in components:
            v, g, la = orb(pos)
            if val is None:
                val, grad, lap = c * v, c * g, c * la
            else:
                val = val + c * v
                grad = grad + c * g
                lap = lap + c * la
        return val, grad, lap
    return f


# --------------------------------------------------------------------------
class System:
    """nuclei: list of ``(Z, x, y, z)``. n_up/n_dn: electron counts.
    orbitals_up/orbitals_dn: orbital callables (len == n_up / n_dn)."""

    def __init__(self, name, nuclei, n_up, n_dn, orbitals_up, orbitals_dn,
                 exact_energy=None, hf_energy=None):
        self.name = name
        self.nuclei = [(float(Z), np.array([x, y, z], float)) for Z, x, y, z in nuclei]
        self.n_up = n_up
        self.n_dn = n_dn
        self.n_elec = n_up + n_dn
        self.orbitals_up = orbitals_up
        self.orbitals_dn = orbitals_dn
        self.exact_energy = exact_energy
        self.hf_energy = hf_energy

    @property
    def nuclear_repulsion(self):
        e = 0.0
        for i in range(len(self.nuclei)):
            Zi, Ri = self.nuclei[i]
            for j in range(i + 1, len(self.nuclei)):
                Zj, Rj = self.nuclei[j]
                e += Zi * Zj / np.linalg.norm(Ri - Rj)
        return e

    def initial_walkers(self, n_walkers, seed=0):
        """Gaussian cloud of electron positions near the nuclei."""
        rng = np.random.default_rng(seed)
        centers = np.array([R for _, R in self.nuclei])
        pos = np.zeros((n_walkers, self.n_elec, 3))
        for e in range(self.n_elec):
            c = centers[e % len(centers)]
            pos[:, e, :] = c + rng.normal(scale=0.8, size=(n_walkers, 3))
        return pos


# --------------------------------------------------------------------------
def hydrogen(zeta=1.0):
    o = atomic_orbital(1, 0, 0, zeta, [0, 0, 0])
    return System("H", [(1, 0, 0, 0)], 1, 0, [o], [], exact_energy=-0.5, hf_energy=-0.5)


def helium(zeta=1.6875):
    o = atomic_orbital(1, 0, 0, zeta, [0, 0, 0])
    return System("He", [(2, 0, 0, 0)], 1, 1, [o], [o],
                  exact_energy=-2.903724, hf_energy=-2.861680)


def _slater_2s_orthogonalized(z1s, z2s, center):
    """A Slater-type 2s ``(r - r0) e^{-z2s r}`` made exactly orthogonal to the
    1s ``e^{-z1s r}`` by choosing the polynomial coefficient ``r0`` analytically
    (⟨1s|2s⟩ = 0). Keeps an independent, physically diffuse 2s exponent while
    guaranteeing a well-conditioned Slater matrix and a sensible nodal surface."""
    center = np.asarray(center, float)
    zc = z1s + z2s                     # combined exponent in <1s|2s>
    # <e^{-z1s r} | r^k e^{-z2s r}> d^3r ~ integral r^{k+2} e^{-zc r} dr = (k+2)!/zc^{k+3}
    r0 = (6.0 / zc ** 4) / (2.0 / zc ** 3)   # = 3/zc, so <1s | (r-r0) e^{-z2s r}> = 0

    def f(pos):
        dr = np.asarray(pos, float) - center
        r = np.sqrt(np.sum(dr ** 2, axis=-1))
        rs = np.where(r > 1e-12, r, 1e-12)
        e = np.exp(-z2s * r)
        val = (r - r0) * e
        # d/dr[(r - r0) e^{-z2s r}] = e (1 - z2s (r - r0))
        dfr = e * (1.0 - z2s * (r - r0))
        d2fr = e * (-z2s - z2s * (1.0 - z2s * (r - r0)))
        grad = (dfr / rs)[..., None] * dr
        lap = d2fr + 2.0 * dfr / rs
        return val, grad, lap
    return f


def lithium(z1s=2.69, z2s=0.72):
    c = [0, 0, 0]
    o1s = atomic_orbital(1, 0, 0, z1s, c)
    o2s = _slater_2s_orthogonalized(z1s, z2s, c)
    return System("Li", [(3, 0, 0, 0)], 2, 1, [o1s, o2s], [o1s],
                  exact_energy=-7.47806, hf_energy=-7.43273)


def beryllium(z1s=3.68, z2s=1.12):
    c = [0, 0, 0]
    o1s = atomic_orbital(1, 0, 0, z1s, c)
    o2s = _slater_2s_orthogonalized(z1s, z2s, c)
    return System("Be", [(4, 0, 0, 0)], 2, 2, [o1s, o2s], [o1s, o2s],
                  exact_energy=-14.66736, hf_energy=-14.573023)


def h2(R=1.4011, zeta=1.20):
    cA = [0, 0, -R / 2]
    cB = [0, 0, +R / 2]
    a = atomic_orbital(1, 0, 0, zeta, cA)
    b = atomic_orbital(1, 0, 0, zeta, cB)
    sigma_g = mo_lcao([(1.0, a), (1.0, b)])
    return System("H2", [(1, 0, 0, -R / 2), (1, 0, 0, R / 2)], 1, 1,
                  [sigma_g], [sigma_g], exact_energy=-1.17447, hf_energy=-1.1336)


def lih(R=3.015, z_li1s=2.69, z_li2s=0.80, z_h=0.95):
    cLi = [0, 0, 0]
    cH = [0, 0, R]
    li1s = atomic_orbital(1, 0, 0, z_li1s, cLi)
    li2s = _slater_2s_orthogonalized(z_li1s, z_li2s, cLi)
    h1s = atomic_orbital(1, 0, 0, z_h, cH)
    core = li1s
    bond = mo_lcao([(0.40, li2s), (0.70, h1s)])
    return System("LiH", [(3, 0, 0, 0), (1, 0, 0, R)], 2, 2,
                  [core, bond], [core, bond],
                  exact_energy=-8.070548, hf_energy=-7.98737)


REGISTRY = {
    "H": hydrogen, "He": helium, "Li": lithium, "Be": beryllium,
    "H2": h2, "LiH": lih,
}


# --------------------------------------------------------------------------
class ElectronGas:
    """N unpolarized electrons in a cubic box of side L with the minimum-image
    convention and an Ewald-free Yukawa-screened interaction proxy for the
    finite single-k-point estimate (see Electron_Gas notebook / Phase 6 note
    on the finite-size caveat). Slater determinant = plane waves filling the
    lowest closed shell."""

    def __init__(self, n_elec=14, rs=1.0):
        self.n_elec = n_elec
        self.rs = rs
        self.n_up = self.n_dn = n_elec // 2
        vol = n_elec * (4 / 3) * np.pi * rs ** 3
        self.L = vol ** (1 / 3)
        self.k_vectors = self._fill_shell(self.n_up)

    def _fill_shell(self, n_orb):
        b = 2 * np.pi / self.L
        rng = range(-3, 4)
        ks = np.array([[i, j, k] for i in rng for j in rng for k in rng], float) * b
        ks = ks[np.argsort(np.sum(ks ** 2, axis=1))]
        return ks[:n_orb]

    def initial_walkers(self, n_walkers, seed=0):
        rng = np.random.default_rng(seed)
        return rng.uniform(0, self.L, size=(n_walkers, self.n_elec, 3))
