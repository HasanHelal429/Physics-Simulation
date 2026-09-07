"""
Slater-Jastrow trial wavefunction, **batched over walkers**

    Psi_T(R) = D_up(R_up) * D_dn(R_dn)
               * exp[ sum_{i<j} u(r_ij) + sum_{i,I} chi(r_iI) ]

Every configuration array has shape ``(W, n_elec, 3)`` (``W`` walkers); all the
returned quantities carry the same leading ``W`` axis. This is what makes VMC
and DMC fast enough to run many systems -- the per-walker Python loop is gone.

Provided:
* ``log_psi(R)``                 ln|Psi_T|                      -> (W,)
* ``grad_ln_psi(R)``             grad Psi / Psi (drift)         -> (W, n_elec, 3)
* ``lap_over_psi(R)``            sum_i lap_i Psi / Psi          -> (W,)
* ``local_energy(R)``            E_L = (H Psi_T)/Psi_T          -> (W,)
* ``move_ratio(R, e, new_re)``   |Psi(R')/Psi(R)|^2 for moving one electron

Spin layout: electrons ``0..n_up-1`` up, the rest down. The Jastrow couples
all pairs; the electron-electron cusp fixes ``u'(0)`` (1/2 antiparallel, 1/4
parallel), so only the Pade denominators and the e-n amplitude are variational.
"""

import numpy as np


# ------------------------------------------------------------------ Jastrow
class PadeJastrow:
    """u_pair(r) = cusp * r / (1 + b_ee r) ;  chi_en(r) = -c_en r / (1 + b_en r).

    ``params`` = ``[b_ee, b_en, c_en]``.  ``b_ee > 0`` keeps u bounded; the
    cusp coefficients are held fixed by the cusp conditions.
    """

    def __init__(self, b_ee=1.0, b_en=1.0, c_en=0.0):
        self.b_ee, self.b_en, self.c_en = float(b_ee), float(b_en), float(c_en)

    @property
    def params(self):
        return np.array([self.b_ee, self.b_en, self.c_en])

    @params.setter
    def params(self, p):
        self.b_ee, self.b_en, self.c_en = [float(x) for x in p]

    def copy(self):
        return PadeJastrow(self.b_ee, self.b_en, self.c_en)

    def u(self, r, cusp):
        return cusp * r / (1.0 + self.b_ee * r)

    def u_deriv(self, r, cusp):
        denom = 1.0 + self.b_ee * r
        return cusp / denom ** 2, -2.0 * cusp * self.b_ee / denom ** 3

    def chi(self, r):
        if self.c_en == 0.0:
            return np.zeros_like(r)
        return -self.c_en * r / (1.0 + self.b_en * r)

    def chi_deriv(self, r):
        if self.c_en == 0.0:
            z = np.zeros_like(r)
            return z, z
        denom = 1.0 + self.b_en * r
        return -self.c_en / denom ** 2, 2.0 * self.c_en * self.b_en / denom ** 3


# ------------------------------------------------------------------ trial WF
class SlaterJastrow:
    def __init__(self, system, jastrow=None):
        self.sys = system
        self.n_up, self.n_dn = system.n_up, system.n_dn
        self.n_elec = system.n_elec
        self.orbs_up, self.orbs_dn = system.orbitals_up, system.orbitals_dn
        self.jas = jastrow or PadeJastrow()
        # cusp coefficient matrix (n_elec, n_elec)
        spin = np.array([0] * self.n_up + [1] * self.n_dn)
        same = spin[:, None] == spin[None, :]
        self.cusp = np.where(same, 0.25, 0.5)
        self.nuc_Z = np.array([Z for Z, _ in system.nuclei])
        self.nuc_R = np.array([R for _, R in system.nuclei])   # (n_nuc, 3)

    # ---------- Slater blocks ----------
    def _slater_block(self, pos, orbs):
        """pos: (W, n, 3). Returns M (W,n,n), G (W,n,n,3), Lp (W,n,n)."""
        W, n = pos.shape[0], pos.shape[1]
        M = np.empty((W, n, n))
        G = np.empty((W, n, n, 3))
        Lp = np.empty((W, n, n))
        for k, orb in enumerate(orbs):
            v, g, la = orb(pos)                 # (W,n), (W,n,3), (W,n)
            M[:, :, k] = v
            G[:, :, k, :] = g
            Lp[:, :, k] = la
        return M, G, Lp

    def _det_derivs(self, pos, orbs):
        """Return logdet (W,), grad_ln_det (W,n,3), lap_ln_det (W,n)."""
        M, G, Lp = self._slater_block(pos, orbs)
        Minv = np.linalg.inv(M)                 # (W,n,n)
        _, logdet = np.linalg.slogdet(M)
        # grad_i ln|det| = sum_k G[w,i,k,:] Minv[w,k,i]
        gi = np.einsum("wikd,wki->wid", G, Minv)
        li = np.einsum("wik,wki->wi", Lp, Minv) - np.sum(gi ** 2, axis=-1)
        return logdet, gi, li

    # ---------- Jastrow ----------
    def _pairs(self, R):
        diff = R[:, :, None, :] - R[:, None, :, :]       # (W,i,j,3)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))       # (W,i,j)
        return diff, dist

    def _jastrow_all(self, R):
        """Returns J (W,), grad_J (W,n,3), lap_J (W,n)."""
        W = R.shape[0]
        diff, dist = self._pairs(R)
        iu = np.triu_indices(self.n_elec, k=1)
        # value
        rr = dist[:, iu[0], iu[1]]                        # (W, npairs)
        cusp = self.cusp[iu]                              # (npairs,)
        Jval = np.sum(self.jas.u(rr, cusp[None, :]), axis=1)
        # gradient / laplacian wrt each electron
        grad = np.zeros((W, self.n_elec, 3))
        lap = np.zeros((W, self.n_elec))
        safe = np.where(dist > 1e-12, dist, 1.0)
        for i in range(self.n_elec):
            js = [j for j in range(self.n_elec) if j != i]
            r = dist[:, i, js]                            # (W, n-1)
            rs = safe[:, i, js]
            du, d2u = self.jas.u_deriv(r, self.cusp[i, js][None, :])
            rhat = diff[:, i, js, :] / rs[:, :, None]
            grad[:, i, :] += np.sum(du[:, :, None] * rhat, axis=1)
            lap[:, i] += np.sum(d2u + 2.0 * du / rs, axis=1)
        # electron-nucleus
        if self.jas.c_en != 0.0:
            d = R[:, :, None, :] - self.nuc_R[None, None, :, :]   # (W,i,I,3)
            rn = np.sqrt(np.sum(d ** 2, axis=-1))
            rn_s = np.where(rn > 1e-12, rn, 1.0)
            Jval = Jval + np.sum(self.jas.chi(rn), axis=(1, 2))
            dchi, d2chi = self.jas.chi_deriv(rn)
            rhat = d / rn_s[..., None]
            grad += np.sum(dchi[..., None] * rhat, axis=2)
            lap += np.sum(d2chi + 2.0 * dchi / rn_s, axis=2)
        return Jval, grad, lap

    # ---------- assembled quantities ----------
    def evaluate(self, R):
        R = np.asarray(R, float)
        if R.ndim == 2:
            R = R[None]
        W = R.shape[0]
        grad_ln_det = np.zeros((W, self.n_elec, 3))
        lap_ln_det = np.zeros((W, self.n_elec))
        logdet = np.zeros(W)
        for orbs, sl in ((self.orbs_up, slice(0, self.n_up)),
                         (self.orbs_dn, slice(self.n_up, self.n_elec))):
            if len(orbs) == 0:
                continue
            ld, gi, li = self._det_derivs(R[:, sl], orbs)
            logdet += ld
            grad_ln_det[:, sl] = gi
            lap_ln_det[:, sl] = li
        Jval, grad_J, lap_J = self._jastrow_all(R)
        grad_ln_psi = grad_ln_det + grad_J
        lap_over_psi = np.sum(lap_ln_det + lap_J + np.sum(grad_ln_psi ** 2, axis=-1), axis=-1)
        return {
            "logpsi": logdet + Jval,
            "grad_ln_psi": grad_ln_psi,
            "lap_over_psi": lap_over_psi,
        }

    def log_psi(self, R):
        return self.evaluate(R)["logpsi"]

    def grad_ln_psi(self, R):
        return self.evaluate(R)["grad_ln_psi"]

    def potential(self, R):
        R = np.asarray(R, float)
        if R.ndim == 2:
            R = R[None]
        W = R.shape[0]
        d = R[:, :, None, :] - self.nuc_R[None, None, :, :]
        rn = np.sqrt(np.sum(d ** 2, axis=-1))
        V = -np.sum(self.nuc_Z[None, None, :] / np.maximum(rn, 1e-12), axis=(1, 2))
        _, dist = self._pairs(R)
        iu = np.triu_indices(self.n_elec, k=1)
        V += np.sum(1.0 / np.maximum(dist[:, iu[0], iu[1]], 1e-12), axis=1)
        V += self.sys.nuclear_repulsion
        return V

    def local_energy(self, R):
        info = self.evaluate(R)
        return -0.5 * info["lap_over_psi"] + self.potential(R)

    def move_ratio(self, R, e, new_pos_e):
        """|Psi(R')/Psi(R)|^2 where electron ``e`` moves to ``new_pos_e``
        (shape (W,3)). Direct re-evaluation -- robust; correct for the drift
        importance sampling in vmc.py."""
        R2 = R.copy()
        R2[:, e, :] = new_pos_e
        return np.exp(2.0 * (self.log_psi(R2) - self.log_psi(R)))
