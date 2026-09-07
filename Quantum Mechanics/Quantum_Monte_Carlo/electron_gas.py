"""
Homogeneous electron gas -- a reduced single-k-point VMC estimate of the
correlation energy per electron, to compare against the Ceperley-Alder points
the PZ81 correlation functional (used throughout the DFT solvers in this repo)
was fit to.

Slater determinant: the lowest closed plane-wave shell, written with the real
combinations {1, cos(k.r), sin(k.r)} so the determinant stays real. Jastrow:
a periodic RPA-style two-body factor with the electron-electron cusp built in.
Interaction: Ewald pair potential + Madelung self-term.

Finite-size caveat: a single k point (no twist averaging), N = 14 -- the
absolute numbers carry a real finite-size error; the trend in r_s and the
shape of g(r) are the robust outputs.
"""

import numpy as np
from scipy.special import erfc as _erfc

# ------------------------------------------------------------------ Ewald
class Ewald:
    def __init__(self, L, kappa=None, g_max=4):
        self.L = L
        self.kappa = kappa or 7.0 / L
        b = 2 * np.pi / L
        gs = np.array([[i, j, k] for i in range(-g_max, g_max + 1)
                       for j in range(-g_max, g_max + 1)
                       for k in range(-g_max, g_max + 1)], float) * b
        g2 = np.sum(gs ** 2, axis=1)
        keep = (g2 > 1e-9) & (g2 < (g_max * b) ** 2 + 1e-9)
        self.G = gs[keep]
        self.G2 = g2[keep]
        self.vol = L ** 3
        self.recip_pref = 4 * np.pi / self.vol * np.exp(-self.G2 / (4 * self.kappa ** 2)) / self.G2
        # Madelung constant (energy of one charge in its own periodic array + bg)
        self.xi = self._madelung(g_max)

    def _madelung(self, r_max=5):
        L, kappa = self.L, self.kappa
        # real space
        s = 0.0
        for i in range(-r_max, r_max + 1):
            for j in range(-r_max, r_max + 1):
                for k in range(-r_max, r_max + 1):
                    if i == j == k == 0:
                        continue
                    R = np.linalg.norm([i, j, k]) * L
                    s += _erfc(kappa * R) / R
        recip = np.sum(self.recip_pref)
        return s + recip - np.pi / (kappa ** 2 * self.vol) - 2 * kappa / np.sqrt(np.pi)

    def pair_potential(self, dr):
        """Ewald pair potential v(dr) for displacement(s) dr, shape (...,3)."""
        dr = np.asarray(dr, float)
        r = np.sqrt(np.sum(dr ** 2, axis=-1))
        r = np.where(r > 1e-10, r, 1e-10)
        real = _erfc(self.kappa * r) / r
        # reciprocal
        phase = np.tensordot(dr, self.G.T, axes=(-1, 0))     # (..., nG)
        recip = np.sum(self.recip_pref * np.cos(phase), axis=-1)
        return real + recip - np.pi / (self.kappa ** 2 * self.vol)


# ------------------------------------------------------------------ trial WF
class HEGWavefunction:
    def __init__(self, gas, jas_A=0.5, jas_F=None):
        self.L = gas.L
        self.n_up = gas.n_up
        self.n_dn = gas.n_dn
        self.n_elec = gas.n_elec
        self.jas_A = jas_A
        self.jas_F = jas_F or gas.L / 4
        self.ewald = Ewald(gas.L)
        self.rs = gas.rs
        # distinct plane-wave directions (one per +/- pair): build a real
        # {1, cos(k.r), sin(k.r)} basis with exactly n_up entries.
        b = 2 * np.pi / self.L
        shells = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                for kk in range(-3, 4):
                    v = np.array([i, j, kk], float)
                    if v @ v < 1e-9:
                        continue
                    if (i, j, kk) > (-i, -j, -kk):        # keep one of the +/- pair
                        shells.append(v * b)
        shells.sort(key=lambda v: v @ v)
        need = (self.n_up - 1 + 1) // 2                   # cos+sin per direction
        self._dirs = shells[:need]

    def _orbs(self, pos):
        """pos (W, n, 3) -> list of n (value, grad, lap) columns."""
        W, n = pos.shape[:2]
        cols = [(np.ones((W, n)), np.zeros((W, n, 3)), np.zeros((W, n)))]
        for k in self._dirs:
            if len(cols) >= n:
                break
            ph = pos @ k
            k2 = float(k @ k)
            cols.append((np.cos(ph), -np.sin(ph)[..., None] * k, -k2 * np.cos(ph)))
            if len(cols) >= n:
                break
            cols.append((np.sin(ph), np.cos(ph)[..., None] * k, -k2 * np.sin(ph)))
        return cols[:n]

    def _det_part(self, pos):
        cols = self._orbs(pos)
        W, n = pos.shape[:2]
        M = np.empty((W, n, n))
        G = np.empty((W, n, n, 3))
        Lp = np.empty((W, n, n))
        for j, (v, g, la) in enumerate(cols):
            M[:, :, j] = v; G[:, :, j, :] = g; Lp[:, :, j] = la
        Minv = np.linalg.inv(M)
        _, logdet = np.linalg.slogdet(M)
        gi = np.einsum("wikd,wki->wid", G, Minv)
        li = np.einsum("wik,wki->wi", Lp, Minv) - np.sum(gi ** 2, axis=-1)
        return logdet, gi, li

    def _min_image(self, dr):
        return dr - self.L * np.round(dr / self.L)

    def _u(self, r):
        """RPA-style pair factor, cut off at the box half-length so it is
        periodic:  u(r) = -A (1 - e^{-r/F}) * (1 - r/rc)^2  for r < rc, else 0.
        As r -> 0:  u ~ -(A/F) r + ...  -> u'(0) = -A/F.  The cusp coefficient
        (1/2 unlike spins in the HEG spin-averaged Jastrow) fixes A/F = 1/2, so
        F is the only shape parameter and A = F/2.  (Sign: electrons repel, so
        the pair factor *lowers* |Psi|^2 at contact -> u(0^+) decreasing.)"""
        rc = 0.5 * self.L
        A = 0.5 * self.jas_F
        F = self.jas_F
        x = np.clip(1.0 - r / rc, 0.0, None)
        env = 1.0 - np.exp(-r / F)
        u = -A * env * x ** 2
        # derivatives
        denv = np.exp(-r / F) / F
        d2env = -np.exp(-r / F) / F ** 2
        dx = np.where(r < rc, -1.0 / rc, 0.0)
        du = -A * (denv * x ** 2 + env * 2 * x * dx)
        d2u = -A * (d2env * x ** 2 + 2 * denv * 2 * x * dx + env * 2 * dx ** 2)
        return u, du, d2u

    def _jastrow(self, R):
        dr = self._min_image(R[:, :, None, :] - R[:, None, :, :])
        r = np.sqrt(np.sum(dr ** 2, axis=-1))
        W = R.shape[0]
        iu = np.triu_indices(self.n_elec, k=1)
        u_pairs, _, _ = self._u(r[:, iu[0], iu[1]])
        Jval = np.sum(u_pairs, axis=1)
        grad = np.zeros((W, self.n_elec, 3))
        lap = np.zeros((W, self.n_elec))
        safe = np.where(r > 1e-10, r, 1.0)
        for i in range(self.n_elec):
            js = [j for j in range(self.n_elec) if j != i]
            rj = r[:, i, js]; rs = safe[:, i, js]
            _, du, d2u = self._u(rj)
            rhat = dr[:, i, js, :] / rs[:, :, None]
            grad[:, i, :] += np.sum(du[:, :, None] * rhat, axis=1)
            lap[:, i] += np.sum(d2u + 2.0 * du / rs, axis=1)
        return Jval, grad, lap

    def evaluate(self, R):
        R = np.asarray(R, float)
        if R.ndim == 2:
            R = R[None]
        ldu, gu, lu = self._det_part(R[:, :self.n_up])
        ldd, gd, ld = self._det_part(R[:, self.n_up:])
        grad_det = np.concatenate([gu, gd], axis=1)
        lap_det = np.concatenate([lu, ld], axis=1)
        Jv, gJ, lJ = self._jastrow(R)
        grad = grad_det + gJ
        lap_over_psi = np.sum(lap_det + lJ + np.sum(grad ** 2, axis=-1), axis=-1)
        return {"logpsi": ldu + ldd + Jv, "grad_ln_psi": grad, "lap_over_psi": lap_over_psi}

    def log_psi(self, R):
        return self.evaluate(R)["logpsi"]

    def potential(self, R):
        R = np.asarray(R, float)
        if R.ndim == 2:
            R = R[None]
        W = R.shape[0]
        V = np.zeros(W)
        iu = np.triu_indices(self.n_elec, k=1)
        dr = self._min_image(R[:, iu[0], :] - R[:, iu[1], :])
        V += np.sum(self.ewald.pair_potential(dr), axis=1)
        V += 0.5 * self.n_elec * self.ewald.xi           # Madelung self + background
        return V

    def local_energy(self, R):
        info = self.evaluate(R)
        return -0.5 * info["lap_over_psi"] + self.potential(R)


# ------------------------------------------------------------------ driver
def heg_energy(rs, n_elec=14, quick=False):
    """Returns a dict: VMC energy/electron with the (F-optimized) Jastrow and
    with no Jastrow, the analytic HF (kinetic+exchange) reference, the
    correlation energy per electron, and g(r)."""
    import vmc as vmcmod
    import estimators as est
    from systems import ElectronGas

    gas = ElectronGas(n_elec=n_elec, rs=rs)
    rng = np.random.default_rng(int(rs * 100) + 1)
    steps = 700 if not quick else 400
    nw = 90 if not quick else 60

    # analytic HF reference per electron (Ha)
    kf = (9 * np.pi / 4) ** (1 / 3) / rs
    e_ts = 0.3 * kf ** 2
    e_x = -3.0 / (4 * np.pi) * kf

    # determinant only (no Jastrow) -> should reproduce e_ts + e_x
    psi0 = HEGWavefunction(gas, jas_F=1e6)              # F huge -> u ~ 0
    R = gas.initial_walkers(nw, seed=1)
    R, tau = vmcmod.equilibrate(psi0, R, rng, 150, 0.02 * gas.L)
    E_det, err_det, _ = vmcmod.sample_energy(psi0, R, rng, steps, tau, decorr=6)
    E_det /= n_elec

    # optimize the single Jastrow shape parameter F
    best = (None, np.inf)
    for F in (np.array([0.8, 1.6, 2.8]) * rs):
        psi = HEGWavefunction(gas, jas_F=float(F))
        Rk = gas.initial_walkers(nw, seed=2)
        Rk, tk = vmcmod.equilibrate(psi, Rk, rng, 120, 0.02 * gas.L)
        Ek, _, _ = vmcmod.sample_energy(psi, Rk, rng, steps // 2, tk, decorr=6)
        if Ek / n_elec < best[1]:
            best = (float(F), Ek / n_elec)
    F_opt = best[0]

    psi = HEGWavefunction(gas, jas_F=F_opt)
    R = gas.initial_walkers(nw + 30, seed=3)
    R, tau = vmcmod.equilibrate(psi, R, rng, 180, 0.02 * gas.L)
    E, err, ex = vmcmod.sample_energy(psi, R, rng, steps, tau, decorr=6)
    E_per = E / n_elec

    r, g = est.pair_correlation(ex["R"], L=gas.L, n_bins=40)
    return {
        "e_vmc": E_per, "e_det": E_det, "e_hf": e_ts + e_x, "e_x": e_x,
        "e_c": E_per - (e_ts + e_x), "F_opt": F_opt,
        "g_r": g, "r": r,
    }
